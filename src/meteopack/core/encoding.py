"""Integer storage for fields that arrive as integers.

Radar imagery is written as integer pixel counts plus a linear calibration:
``value = scale * stored + offset``, with two reserved codes for "measured
nothing here" and "outside the image". Decoding that to float32 at pack time
doubles the store, loses exactness, and throws away the distinction between the
two reserved codes — and it is not even what the data is. So the canonical
store keeps the integer form, with a calibration of *its own* choosing, and
every source frame is re-encoded onto it.

Two ideas carry this module.

*The canonical calibration is fixed by configuration, not inherited.* A product
spans years of upstream generations whose calibrations differ; pinning one
target means every frame in the store decodes with the same two numbers, and a
reader never has to ask which generation a frame came from in order to read it.

*Re-encoding goes through a lookup table.* Both sides are integers of at most
16 bits, so there are at most 65536 distinct inputs. Building the table costs
65536 operations once per frame; applying it is one fancy-index over 535,500
pixels. Doing the arithmetic per pixel instead is ~8x the work and introduces
float rounding that varies with the order of operations. The table also makes
the sentinel rules total: every input code maps somewhere explicit, including
the ones no document mentions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Beyond this many distinct input codes a lookup table stops being the cheap
#: option and the arithmetic is done directly. 16-bit imagery sits well inside.
_MAX_LUT = 1 << 16


@dataclass(frozen=True, slots=True)
class Encoding:
    """How one variable is stored: dtype, calibration and the two sentinels.

    `missing` and `out_of_image` are distinct on purpose. "The radar measured
    nothing at this pixel" and "this pixel is outside the radar image" are
    different facts, and a consumer deciding whether to inpaint, mask or ignore
    needs to tell them apart. Collapsing both to NaN is lossy in a way that
    cannot be undone later.
    """

    dtype: np.dtype
    scale: float
    offset: float
    missing: int
    out_of_image: int

    @classmethod
    def from_config(cls, spec: dict | None, *, name: str = "") -> Encoding | None:
        """Build from a product config's `encoding:` block, or None for float."""
        if not spec:
            return None
        dtype = np.dtype(str(spec.get("dtype", "uint16")))
        if dtype.kind not in "ui":
            raise ValueError(f"{name}: integer encoding needs an integer dtype, got {dtype}")
        missing, out_of_image = default_sentinels(dtype)
        return cls(
            dtype=dtype,
            scale=float(spec.get("scale", 1.0)),
            offset=float(spec.get("offset", 0.0)),
            missing=int(spec.get("missing", missing)),
            out_of_image=int(spec.get("out_of_image", out_of_image)),
        )

    @property
    def formula(self) -> str:
        """The calibration, in the form KNMI writes it, for the array attrs."""
        return f"GEO={self.scale:.6f}*PV+{self.offset:.6f}"

    @property
    def reserved(self) -> frozenset[int]:
        return frozenset({self.missing, self.out_of_image})

    def decode(self, stored: np.ndarray) -> np.ndarray:
        """Integer codes -> float32 with NaN for both sentinels.

        For readers. The store keeps the integers; nothing in the pipeline
        round-trips through this.
        """
        raw = np.asarray(stored)
        out = raw.astype("float32") * np.float32(self.scale) + np.float32(self.offset)
        out[np.isin(raw, np.asarray(sorted(self.reserved), dtype=raw.dtype))] = np.nan
        return out


def default_sentinels(dtype: np.dtype) -> tuple[int, int]:
    """`(missing, out_of_image)` = the top two codes of an integer dtype.

    uint16 -> (65534, 65535), uint8 -> (254, 255). The same convention the
    radar_v3 stores use, so a reader that knows one knows the other.
    """
    info = np.iinfo(dtype)
    return int(info.max) - 1, int(info.max)


def encode(
    values: np.ndarray,
    target: Encoding,
    *,
    source_scale: float,
    source_offset: float,
    source_missing: frozenset[int] | set[int] = frozenset(),
    source_out_of_image: frozenset[int] | set[int] = frozenset(),
) -> np.ndarray:
    """Re-encode integer pixel counts from one calibration onto `target`.

    `source_missing` and `source_out_of_image` are the codes the *file* reserves,
    which differ by generation; whatever they are, they come out as the target's
    two sentinels. Everything else is decoded with the source calibration and
    re-encoded with the target's, rounded to nearest, and clipped away from the
    reserved codes so a real value can never be mistaken for a sentinel.
    """
    raw = np.asarray(values)
    too_wide = raw.dtype.kind in "ui" and (
        int(np.iinfo(raw.dtype).max) - int(np.iinfo(raw.dtype).min) + 1 > _MAX_LUT
    )
    if raw.dtype.kind not in "ui" or too_wide:
        # Float pixels, or integers too wide to tabulate. Same rules, done
        # directly: the sentinels are matched against the stored codes, and
        # everything else is decoded with the source calibration first.
        return _encode_direct(raw, target, source_scale, source_offset,
                              source_missing, source_out_of_image)

    info = np.iinfo(raw.dtype)
    span = int(info.max) - int(info.min) + 1

    codes = np.arange(int(info.min), int(info.max) + 1, dtype="int64")
    table = np.empty(span, dtype=target.dtype)

    out_mask = _code_mask(codes, source_out_of_image)
    missing_mask = _code_mask(codes, source_missing) & ~out_mask
    valid = ~(out_mask | missing_mask)

    table[out_mask] = target.out_of_image
    table[missing_mask] = target.missing
    if valid.any():
        table[valid] = _quantise(
            codes[valid].astype("float64") * source_scale + source_offset, target
        )

    return table[raw.astype("int64") - int(info.min)]


def _encode_direct(
    values: np.ndarray,
    target: Encoding,
    source_scale: float,
    source_offset: float,
    source_missing,
    source_out_of_image,
) -> np.ndarray:
    """The general path: no lookup table, so each pixel is converted directly.

    Reached for float pixels and for integers wider than 16 bits. The rules are
    the ones the table encodes, and a non-finite pixel counts as missing - a
    NaN is the file saying it has no value here, which is exactly what
    `missing` means.
    """
    raw = np.asarray(values)
    out = np.empty(raw.shape, dtype=target.dtype)
    out_mask = _code_mask(raw, source_out_of_image)
    missing_mask = _code_mask(raw, source_missing) & ~out_mask
    if raw.dtype.kind == "f":
        missing_mask = missing_mask | ~np.isfinite(raw)
    valid = ~(out_mask | missing_mask)
    out[out_mask] = target.out_of_image
    out[missing_mask] = target.missing
    if valid.any():
        decoded = raw[valid].astype("float64") * source_scale + source_offset
        out[valid] = _quantise(decoded, target)
    return out


def _code_mask(values: np.ndarray, codes) -> np.ndarray:
    if not codes:
        return np.zeros(np.shape(values), dtype=bool)
    wanted = np.asarray(sorted(int(c) for c in codes), dtype="int64")
    return np.isin(np.asarray(values).astype("int64", copy=False), wanted)


def _quantise(decoded: np.ndarray, target: Encoding) -> np.ndarray:
    """Physical values -> target codes, rounded, clipped clear of the sentinels.

    The usable range stops below the lower of the two sentinels. A value that
    would land on one is pulled down to the last ordinary code rather than
    written as a sentinel: a saturated pixel is wrong by one count, while a
    pixel silently reported as missing is wrong in kind.
    """
    info = np.iinfo(target.dtype)
    ceiling = min(target.reserved) - 1
    codes = np.rint((decoded - target.offset) / target.scale)
    below = codes < info.min
    above = codes > ceiling
    if below.any() or above.any():
        logger.debug(
            "%d value(s) below and %d above the encodable range were clipped",
            int(below.sum()), int(above.sum()),
        )
    return np.clip(codes, info.min, ceiling).astype(target.dtype)
