"""Tests for the KNMI 5-minute radar composite source — the first gridded one.

Nothing here needs the real archive. The fixtures are synthetic HDF5 frames
built with h5py into `tmp_path` on the `example_tiny` 8x6 grid from the real
registry, tarred the way KNMI tars them, so the whole chain — tar member ->
timestamp, calibration formula, the two sentinels, the grid assertion, the
layout fingerprint — runs against files this module actually opens rather than
against mocks.

The calibration formulas, `image_geo_parameter` names and sentinel values are
verbatim from four product generations observed in the live archive on
2026-10-31; the arrays around them are tiny and made up. Two tests at the end
are guarded on the archive existing and read one real tar's index and first
frame, which is what proves the synthetic fixtures are shaped like the data.

What the missing-value tests are defending, because getting it backwards is
silent: `calibration_out_of_image` is NaN, a `calibration_missing_data` of 0 is
a real **0 mm of no echo**, and a `calibration_missing_data` of 65534 is really
missing. One archive, both conventions.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
import yaml

from meteopack.core import grids
from meteopack.core.config import ProductConfig, SourceConfig, StoreConfig, Variable
from meteopack.core.encoding import Encoding
from meteopack.core.store import gridded
from meteopack.core.timeline import Timeline
from meteopack.sources.base import RawFile, Source, TimeWindow
from meteopack.sources.knmi_radar import (
    ImageSpec,
    KnmiRadar,
    SchemaDriftError,
    _encode_image,
    calibrate,
    member_time,
    parse_calibration,
    product_fingerprint,
    representable_sentinel,
    schema_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = REPO_ROOT / "configs" / "sources" / "knmi_radar.yaml"
PRODUCT_YAML = REPO_ROOT / "configs" / "products" / "knmi_radar" / "rtcor_5min.yaml"

ARCHIVE = Path(
    "/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi/nl_rdr_data_rtcor_5m_tar"
)

#: The real registry's test grid. Real, loadable, and small enough to eyeball.
GRID_ID = "example_tiny"
NY, NX = grids.get(GRID_ID).shape
CORNERS = np.asarray(grids.get(GRID_ID).corners_lonlat, dtype="float32").ravel()

CADENCE = 300
DAY_START = timedelta(hours=8, minutes=5)

# Verbatim from the archive. Note the two spacings and the doubled sign.
PRECIP_FORMULA = "GEO = 0.010000 * PV + 0.000000"
ACCUM_FORMULA = "GEO=0.010000*PV+0.000000"
QUALITY_FORMULA = "GEO=0.004*PV+0"
ADJUST_FORMULA = "GEO=0.001*PV+-32.768"

OUT_OF_IMAGE = 65535
LEGACY_MISSING = 0       # pre-2023: collides with a dry pixel, so it means no echo
MODERN_MISSING = 65534   # 2023 on: outside the data range, so it is really missing


# --------------------------------------------------------------------- fixtures


def frame_bytes(
    images: list[tuple[str, str, str, int, int, np.ndarray]],
    *,
    rows: int = NY,
    columns: int = NX,
    corners: np.ndarray | None = None,
) -> bytes:
    """One synthetic KNMI HDF5 frame, laid out the way the real ones are."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        geo = f.create_group("geographic")
        geo.attrs["geo_number_rows"] = np.array([rows], dtype="int32")
        geo.attrs["geo_number_columns"] = np.array([columns], dtype="int32")
        geo.attrs["geo_product_corners"] = (
            CORNERS if corners is None else np.asarray(corners, dtype="float32")
        )
        for group, parameter, formula, missing, out_of_image, data in images:
            node = f.create_group(group)
            node.attrs["image_geo_parameter"] = np.bytes_(parameter)
            calibration = node.create_group("calibration")
            calibration.attrs["calibration_formulas"] = np.bytes_(formula)
            calibration.attrs["calibration_missing_data"] = np.array([missing], dtype="int32")
            calibration.attrs["calibration_out_of_image"] = np.array(
                [out_of_image], dtype="int32"
            )
            node.create_dataset("image_data", data=data)
    return buf.getvalue()


def precip_frame(counts: np.ndarray, *, legacy: bool = True, quality: np.ndarray | None = None):
    """A frame in the pre-2023 (`legacy`) or post-2023 generation."""
    if legacy:
        images = [("image1", "PRECIP_[MM]", PRECIP_FORMULA, LEGACY_MISSING, OUT_OF_IMAGE, counts)]
    else:
        images = [
            ("image1", "ACCUMULATION_[MM]", ACCUM_FORMULA, MODERN_MISSING, OUT_OF_IMAGE, counts)
        ]
    if quality is not None:
        images.append(("image2", "QUALITY_[-]", QUALITY_FORMULA, 254, 255, quality))
    return images


def write_tar(path: Path, members: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for name, blob in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    return path


def tar_name(day: str, *, start: str = "080500", end_day: str | None = None,
             end: str = "080000") -> str:
    following = end_day or (
        datetime.strptime(day, "%Y%m%d") + timedelta(days=1)
    ).strftime("%Y%m%d")
    return f"RAD25_OPER_R___TARRRT__L2__{day}T{start}_{following}T{end}_0001.tar"


def day_tar(root: Path, day: str, frames: int = 3, **counts_kwargs) -> Path:
    """One tar of `frames` consecutive 5-minute members starting at 08:05."""
    first = datetime.strptime(day, "%Y%m%d").replace(tzinfo=UTC) + DAY_START
    members = {}
    for i in range(frames):
        when = first + timedelta(seconds=CADENCE * i)
        counts = np.full((NY, NX), i, dtype="uint16")
        members[f"RAD_NL25_RAC_RT_{when:%Y%m%d%H%M}.h5"] = frame_bytes(
            precip_frame(counts, **counts_kwargs)
        )
    return write_tar(root / tar_name(day), members)


def make_cfg(**extra) -> ProductConfig:
    base = {
        "day_start": "08:05",
        "missing_data_policy": "no_echo_if_zero",
    }
    base.update(extra)
    return ProductConfig(
        source="knmi_radar",
        product="rtcor_5min",
        cadence_seconds=CADENCE,
        grid_id=GRID_ID,
        store=StoreConfig(
            kind="gridded",
            chunks={"time": 1, "y": NY, "x": NX},
            shard={"time": 288, "y": NY, "x": NX},
            compression_level=1,
        ),
        variables=[
            Variable(
                name="precipitation_amount",
                source_key="PRECIP_[MM]|ACCUMULATION_[MM]",
                units="mm",
                long_name="Radar precipitation accumulation",
                valid_min=0.0,
                valid_max=200.0,
                no_echo=0.0,
            ),
            Variable(
                name="quality",
                source_key="QUALITY_[-]",
                units="1",
                valid_min=0.0,
                valid_max=1.0,
            ),
        ],
        extra=base,
    )


def raw_of(path: Path, key: str = "knmi_radar/rtcor_5min/20190601") -> RawFile:
    return RawFile(path=path, sha256="", n_bytes=path.stat().st_size, unit_key=key)


# ------------------------------------------------------------------- the contract


def test_implements_the_source_protocol() -> None:
    assert isinstance(KnmiRadar(), Source)
    assert KnmiRadar.name == "knmi_radar"


def test_configs_load_into_the_config_models() -> None:
    source = SourceConfig(**yaml.safe_load(SOURCE_YAML.read_text()))
    product = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert source.name == product.source == "knmi_radar"
    assert product.store.kind == "gridded"
    assert product.grid_id == "knmi_1km_stereo"
    assert product.cadence_seconds == 300
    assert product.store.chunks == {"time": 1, "y": 765, "x": 700}
    assert product.store.shard == {"time": 288, "y": 765, "x": 700}
    assert (product.raw_layout.archive, product.raw_layout.granularity) == ("tar", "day")
    # The grid the product claims really is the 765x700 one the frames assert against.
    assert grids.get(product.grid_id).shape == (765, 700)


def test_pinned_fingerprint_matches_the_pinned_layouts() -> None:
    """Editing extra.known_layouts without re-pinning the hash must fail loudly."""
    product = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    layouts = product.extra["known_layouts"]
    # 11 layouts, from a census over the whole archive rather than a sample.
    assert len(layouts) == len(set(layouts)) == 11
    assert product_fingerprint(layouts) == product.schema_fingerprint


def test_the_source_exposes_the_hash_the_pins_are_checked_against() -> None:
    """`pack` writes refreshed pins; they have to hash the way the check does.

    The report used a generic helper that joins names on a NUL while this
    source joins layouts on a "|", so a fingerprint copied out of the report
    failed the consistency check it existed to satisfy.
    """
    assert KnmiRadar.product_fingerprint(["b", "a"]) == product_fingerprint(["a", "b"])


# -------------------------------------------------------------------- calibration


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        # All four verbatim from the archive: two spacings, a doubled sign, and
        # an exponent.
        (PRECIP_FORMULA, (0.01, 0.0)),
        (ACCUM_FORMULA, (0.01, 0.0)),
        (ADJUST_FORMULA, (0.001, -32.768)),
        ("GEO = 1.52593e-05 * PV + 0", (1.52593e-05, 0.0)),
        ("GEO=0.16*PV+-20.24", (0.16, -20.24)),
        ("GEO = 2 * PV - 5", (2.0, -5.0)),
    ],
)
def test_parse_calibration(formula: str, expected: tuple[float, float]) -> None:
    scale, offset = parse_calibration(formula)
    assert (pytest.approx(scale), pytest.approx(offset)) == expected


def test_unreadable_calibration_is_an_error_not_a_default() -> None:
    with pytest.raises(SchemaDriftError, match="calibration formula"):
        parse_calibration("REFLECTIVITY = magic(PV)")


def test_scale_comes_from_the_file_not_from_a_constant(tmp_path: Path) -> None:
    """A frame calibrated at 0.5 must read back at 0.5, not at radar's usual 0.01."""
    counts = np.array([[1, 2, 3, 4, 5, 6]] * NY, dtype="uint16")
    blob = frame_bytes(
        [("image1", "PRECIP_[MM]", "GEO = 0.500000 * PV + 1.250000", 0, OUT_OF_IMAGE, counts)]
    )
    path = write_tar(tmp_path / tar_name("20190601"), {"RAD_NL25_RAC_RT_201906010805.h5": blob})
    ds = KnmiRadar().parse(raw_of(path), make_cfg())
    # 0.5 * PV + 1.25, and emphatically not 0.01 * PV.
    np.testing.assert_allclose(ds["precipitation_amount"].values[0], counts * 0.5 + 1.25)


# ---------------------------------------------------------------- missing values


def _single_frame_values(tmp_path: Path, counts: np.ndarray, *, legacy: bool, **cfg_extra):
    blob = frame_bytes(precip_frame(counts, legacy=legacy))
    path = write_tar(
        tmp_path / tar_name("20190601"), {"RAD_NL25_RAC_RT_201906010805.h5": blob}
    )
    ds = KnmiRadar().parse(raw_of(path), make_cfg(**cfg_extra))
    return ds["precipitation_amount"].values[0]


def test_out_of_image_sentinel_becomes_nan(tmp_path: Path) -> None:
    counts = np.full((NY, NX), 7, dtype="uint16")
    counts[0, 0] = OUT_OF_IMAGE
    values = _single_frame_values(tmp_path, counts, legacy=True)
    assert np.isnan(values[0, 0])
    assert values[1, 1] == pytest.approx(0.07)


def test_no_echo_zero_is_zero_millimetres_not_missing(tmp_path: Path) -> None:
    """The whole archive is mostly zeros. Reading them as NaN loses everything."""
    counts = np.zeros((NY, NX), dtype="uint16")
    counts[0, 0] = OUT_OF_IMAGE
    values = _single_frame_values(tmp_path, counts, legacy=True)
    assert np.isnan(values[0, 0])
    dry = values[1:, :]
    assert not np.isnan(dry).any()
    np.testing.assert_array_equal(dry, np.zeros_like(dry))


def test_modern_missing_sentinel_is_nan_while_zero_stays_zero(tmp_path: Path) -> None:
    """From 2023 the sentinel moved to 65534 and 0 became an ordinary value."""
    counts = np.zeros((NY, NX), dtype="uint16")
    counts[0, 0] = OUT_OF_IMAGE
    counts[0, 1] = MODERN_MISSING
    counts[1, 0] = 250
    values = _single_frame_values(tmp_path, counts, legacy=False)
    assert np.isnan(values[0, 0])
    assert np.isnan(values[0, 1])
    assert values[1, 0] == pytest.approx(2.5)
    assert values[2, 2] == 0.0  # a dry pixel is still a real zero


def test_missing_policy_can_be_forced_either_way(tmp_path: Path) -> None:
    counts = np.zeros((NY, NX), dtype="uint16")
    forced_missing = _single_frame_values(
        tmp_path, counts, legacy=True, missing_data_policy="missing"
    )
    assert np.isnan(forced_missing).all()

    counts_modern = np.full((NY, NX), MODERN_MISSING, dtype="uint16")
    forced_echo = _single_frame_values(
        tmp_path / "b", counts_modern, legacy=False, missing_data_policy="no_echo"
    )
    np.testing.assert_array_equal(forced_echo, np.zeros_like(forced_echo))


# The three calibrations the echotop archive actually uses, censused on real
# frames: PV=0 is 39-49% of every image, PV=1 is never used, so PV=0 is "no echo"
# throughout. Only the first decodes PV=0 to 0 km.
ECHOTOP_GENERATIONS = [
    ("2008", 0.062992, 0.0),
    ("2017", 0.062992, -0.062992),
    ("2019", 0.102362, -0.102362),
]
ECHOTOP_TARGET = Encoding(np.dtype("uint8"), 0.102362, -0.102362, 254, 255)


@pytest.mark.parametrize(("year", "scale", "offset"), ECHOTOP_GENERATIONS)
def test_no_echo_echo_top_is_zero_km_in_every_generation(year, scale, offset) -> None:
    """The bug this guards: no echo became -0.102 km once KNMI shifted the offset.

    The fix was to give the no-echo code the variable's stated value instead of
    letting the file's calibration decode it - the arithmetic is right only where
    the offset happens to be 0.
    """
    spec = ImageSpec("image1", "HEIGHT_[KM]", scale, offset, 0.0, 255.0, "uint8", (NY, NX))
    counts = np.zeros((NY, NX), dtype="uint8")
    counts[0, 0] = 255
    counts[0, 1] = 100
    stored = _encode_image(counts, spec, ECHOTOP_TARGET, policy="no_echo_if_zero", no_echo=0.0)
    km = ECHOTOP_TARGET.decode(stored)
    assert np.isnan(km[0, 0])                                   # outside the image
    assert km[0, 1] == pytest.approx(100 * scale + offset, abs=0.0513)  # half a target count
    dry = km[1:, :]
    assert (dry == 0.0).all(), f"{year}: no echo decoded to {np.unique(dry)}"


@pytest.mark.parametrize("offset", [-31.5, -32.0])
def test_no_echo_reflectivity_is_one_value_across_generations(offset) -> None:
    """2008 frames decode PV=0 to -31.5 dBZ and 2019+ to -32.0.

    Nothing range-checks that - both are inside [-32, 95] - so it would have
    stayed a silent split of one physical fact across two numbers.
    """
    target = Encoding(np.dtype("uint8"), 0.5, -32.0, 254, 255)
    spec = ImageSpec("image1", "REFLECTIVITY_[DBZ]", 0.5, offset, 0.0, 255.0, "uint8", (NY, NX))
    stored = _encode_image(np.zeros((NY, NX), "uint8"), spec, target,
                           policy="no_echo_if_zero", no_echo=-32.0)
    assert (target.decode(stored) == -32.0).all()


def test_a_variable_with_no_no_echo_state_honours_the_zero_sentinel() -> None:
    """Quality and adjustment fields have no 'no echo'. Their code 0 is missing.

    Decoding it instead gave adjustment -32.768 dB, clipped to the floor of the
    store's range: a plausible-looking number for a pixel the file says is empty.
    """
    target = Encoding(np.dtype("uint8"), 0.16, -20.24, 254, 255)
    spec = ImageSpec("image2", "ADJUSTMENT_FACTOR_[DB]", 0.001, -32.768, 0.0, 65535.0,
                     "uint16", (NY, NX))
    stored = _encode_image(np.zeros((NY, NX), "uint16"), spec, target,
                           policy="no_echo_if_zero", no_echo=None)
    assert (stored == target.missing).all()


def test_a_no_echo_policy_without_a_no_echo_value_is_refused(tmp_path: Path) -> None:
    """Forgetting `no_echo:` must not quietly erase the dry mask."""
    cfg = make_cfg()
    cfg.variables[0].no_echo = None
    with pytest.raises(ValueError, match="no_echo"):
        KnmiRadar()._missing_policy(cfg)


def test_the_float_path_uses_the_same_no_echo_value() -> None:
    spec = ImageSpec("image1", "HEIGHT_[KM]", 0.102362, -0.102362, 0.0, 255.0, "uint8", (NY, NX))
    out = calibrate(np.zeros((NY, NX), "uint8"), spec, policy="no_echo_if_zero", no_echo=0.0)
    assert (out == 0.0).all()
    honoured = calibrate(np.zeros((NY, NX), "uint8"), spec, policy="no_echo_if_zero",
                         no_echo=None)
    assert np.isnan(honoured).all()


def test_unknown_missing_policy_is_rejected() -> None:
    spec = ImageSpec("image1", "PRECIP_[MM]", 0.01, 0.0, 0.0, 65535.0, "uint16", (NY, NX))
    with pytest.raises(ValueError, match="missing_data_policy"):
        calibrate(np.zeros((NY, NX), "uint16"), spec, policy="whatever")


@pytest.mark.parametrize(
    ("sentinel", "dtype", "expected"),
    [
        (65535.0, "uint16", 65535.0),   # representable: left alone
        (65535.0, "uint8", 255.0),      # the real case: clamped to the dtype maximum
        (65534.0, "uint8", 254.0),      # keeps its distance from the top, not folded onto 255
        (254.0, "uint8", 254.0),
        (None, "uint8", None),
        (-1.0, "uint8", None),          # no pixel can carry it, so there is no mask
        (65535.0, "float32", 65535.0),  # not an integer image: not ours to clamp
    ],
)
def test_representable_sentinel(sentinel, dtype: str, expected) -> None:
    assert representable_sentinel(sentinel, np.dtype(dtype)) == expected


def test_uint8_frame_quoting_the_uint16_sentinel_is_still_masked(tmp_path: Path) -> None:
    """Really in the archive: ten frames of 2019-06-01 are uint8 inside a uint16
    day, and they keep quoting out_of_image = 65535. Read literally, ~104 000
    out-of-image pixels come back as 2.55 mm of rain over the whole North Sea."""
    counts = np.zeros((NY, NX), dtype="uint8")
    counts[0, :] = 255  # the footprint mask, written at the uint8 maximum
    counts[1, 0] = 100
    blob = frame_bytes(
        [("image1", "PRECIP_[MM]", PRECIP_FORMULA, LEGACY_MISSING, OUT_OF_IMAGE, counts)]
    )
    path = write_tar(tmp_path / tar_name("20190601"), {"RAD_NL25_RAC_RT_201906010805.h5": blob})
    values = KnmiRadar().parse(raw_of(path), make_cfg(known_layouts=[]))[
        "precipitation_amount"
    ].values[0]
    assert np.isnan(values[0]).all()
    assert values[1, 0] == pytest.approx(1.0)
    assert values[2, 2] == 0.0  # and a dry pixel is still a real zero


def test_calibrate_resolves_sentinels_before_scaling() -> None:
    """65535 * 0.01 would be a perfectly plausible 655 mm. It must never appear."""
    spec = ImageSpec("image1", "PRECIP_[MM]", 0.01, 0.0, 0.0, 65535.0, "uint16", (1, 3))
    out = calibrate(np.array([[0, 100, 65535]], dtype="uint16"), spec)
    assert out.dtype == np.float32
    assert out[0, 0] == 0.0
    assert out[0, 1] == pytest.approx(1.0)
    assert np.isnan(out[0, 2])


# ----------------------------------------------------------------- grid assertion


def test_frame_shape_must_match_the_grid(tmp_path: Path) -> None:
    blob = frame_bytes(
        precip_frame(np.zeros((NY + 1, NX), dtype="uint16")), rows=NY + 1, columns=NX
    )
    path = write_tar(tmp_path / tar_name("20190601"), {"RAD_NL25_RAC_RT_201906010805.h5": blob})
    with pytest.raises(SchemaDriftError, match=r"grid example_tiny is \(8, 6\)"):
        KnmiRadar().parse(raw_of(path), make_cfg())


def test_declared_row_and_column_counts_must_match_the_grid(tmp_path: Path) -> None:
    blob = frame_bytes(precip_frame(np.zeros((NY, NX), dtype="uint16")), rows=999)
    path = write_tar(tmp_path / tar_name("20190601"), {"RAD_NL25_RAC_RT_201906010805.h5": blob})
    with pytest.raises(SchemaDriftError, match="declares 999x6"):
        KnmiRadar().parse(raw_of(path), make_cfg())


def test_product_corners_must_match_the_grid(tmp_path: Path) -> None:
    moved = CORNERS.copy()
    moved[0] += 1.0
    blob = frame_bytes(precip_frame(np.zeros((NY, NX), dtype="uint16")), corners=moved)
    path = write_tar(tmp_path / tar_name("20190601"), {"RAD_NL25_RAC_RT_201906010805.h5": blob})
    with pytest.raises(SchemaDriftError, match="geo_product_corners"):
        KnmiRadar().parse(raw_of(path), make_cfg())


def test_float32_corner_wobble_is_tolerated(tmp_path: Path) -> None:
    """The real corners differ in the 5th decimal between generations."""
    wobbled = CORNERS + np.float32(2e-5)
    blob = frame_bytes(precip_frame(np.zeros((NY, NX), dtype="uint16")), corners=wobbled)
    path = write_tar(tmp_path / tar_name("20190601"), {"RAD_NL25_RAC_RT_201906010805.h5": blob})
    assert KnmiRadar().parse(raw_of(path), make_cfg()).sizes["time"] == 1


# ----------------------------------------------------------------- member naming


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RAD_NL25_RAC_RT_201906010805.h5", datetime(2019, 6, 1, 8, 5, tzinfo=UTC)),
        ("RAD_NL25_RAC_RT_201906020800.h5", datetime(2019, 6, 2, 8, 0, tzinfo=UTC)),
        ("./RAD_NL25_RAC_RT_202604220800.h5", datetime(2026, 4, 22, 8, 0, tzinfo=UTC)),
        ("RAD_NL25_RAC_RT_20190601080.h5", None),
        ("RAD_NL25_RAC_RT_201913010805.h5", None),  # month 13
        ("readme.txt", None),
    ],
)
def test_member_time(name: str, expected: datetime | None) -> None:
    assert member_time(name) == expected


def test_non_radar_members_are_ignored(tmp_path: Path) -> None:
    blob = frame_bytes(precip_frame(np.zeros((NY, NX), dtype="uint16")))
    path = write_tar(
        tmp_path / tar_name("20190601"),
        {"RAD_NL25_RAC_RT_201906010805.h5": blob, "MANIFEST.txt": b"not a frame"},
    )
    assert KnmiRadar().parse(raw_of(path), make_cfg()).sizes["time"] == 1


# ------------------------------------------------------------------------- list


def test_list_uses_the_0805_to_0800_day_and_sets_valid_time(tmp_path: Path) -> None:
    for day in ("20190601", "20190602", "20190603"):
        day_tar(tmp_path, day)
    source = KnmiRadar(archive_root=tmp_path)
    window = TimeWindow(datetime(2019, 6, 1, tzinfo=UTC), datetime(2019, 6, 5, tzinfo=UTC))
    units = source.list(window, make_cfg())

    assert [u.key for u in units] == [
        "knmi_radar/rtcor_5min/20190601",
        "knmi_radar/rtcor_5min/20190602",
        "knmi_radar/rtcor_5min/20190603",
    ]
    # Not midnight: the first frame of a KNMI radar day is 08:05 UTC.
    assert units[0].valid_time == datetime(2019, 6, 1, 8, 5, tzinfo=UTC)
    meta = units[0].meta or {}
    assert meta["covers_from"] == "2019-06-01T08:05:00+00:00"
    # The name's end token is the LAST frame (08:00), so the half-open end is 08:05.
    assert meta["covers_to"] == "2019-06-02T08:05:00+00:00"
    assert units[0].url.startswith("file://")


def test_every_unit_carries_valid_time(tmp_path: Path) -> None:
    """The planner refuses a gridded unit without one: shards must not be split."""
    for day in ("20190601", "20190602"):
        day_tar(tmp_path, day)
    write_tar(tmp_path / tar_name("20190603", start="0080000"), {})  # the archived typo
    units = KnmiRadar(archive_root=tmp_path).list(
        TimeWindow(datetime(2019, 6, 1, tzinfo=UTC), datetime(2019, 6, 10, tzinfo=UTC)),
        make_cfg(),
    )
    assert len(units) == 3
    assert all(u.valid_time is not None for u in units)
    assert all(u.valid_time.tzinfo is UTC for u in units)


def test_malformed_filename_falls_back_to_the_configured_day_start(tmp_path: Path) -> None:
    """`20181220T0080000` is really in the archive. It must not take out the day."""
    write_tar(tmp_path / tar_name("20181220", start="0080000", end="075500"), {})
    units = KnmiRadar(archive_root=tmp_path).list(
        TimeWindow(datetime(2018, 12, 1, tzinfo=UTC), datetime(2019, 1, 1, tzinfo=UTC)),
        make_cfg(),
    )
    assert len(units) == 1
    assert units[0].key == "knmi_radar/rtcor_5min/20181220"
    assert units[0].valid_time == datetime(2018, 12, 20, 8, 5, tzinfo=UTC)
    # Only the start token is broken; the readable end token still bounds the day.
    assert (units[0].meta or {})["covers_to"] == "2018-12-21T08:00:00+00:00"


def test_list_reads_the_older_filename_convention_from_the_name(tmp_path: Path) -> None:
    """Two 2018 tars still run 08:00 -> 07:55. Neither is assumed to start at 08:05."""
    write_tar(tmp_path / tar_name("20181219", start="080000", end="075500"), {})
    units = KnmiRadar(archive_root=tmp_path).list(
        TimeWindow(datetime(2018, 12, 1, tzinfo=UTC), datetime(2019, 1, 1, tzinfo=UTC)),
        make_cfg(),
    )
    assert units[0].valid_time == datetime(2018, 12, 19, 8, 0, tzinfo=UTC)
    assert (units[0].meta or {})["covers_to"] == "2018-12-20T08:00:00+00:00"


def test_list_clips_to_the_window_on_the_0805_boundary(tmp_path: Path) -> None:
    for day in ("20190601", "20190602", "20190603"):
        day_tar(tmp_path, day)
    source = KnmiRadar(archive_root=tmp_path)
    cfg = make_cfg()

    # A window ending at 08:05 on the 2nd ends exactly where the 2nd's tar begins.
    units = source.list(
        TimeWindow(datetime(2019, 6, 1, tzinfo=UTC), datetime(2019, 6, 2, 8, 5, tzinfo=UTC)), cfg
    )
    assert [u.key for u in units] == ["knmi_radar/rtcor_5min/20190601"]

    # A window starting one minute later still needs the 1st's tar: that day runs on.
    units = source.list(
        TimeWindow(datetime(2019, 6, 2, tzinfo=UTC), datetime(2019, 6, 3, tzinfo=UTC)), cfg
    )
    assert [u.key for u in units] == [
        "knmi_radar/rtcor_5min/20190601",
        "knmi_radar/rtcor_5min/20190602",
    ]


def test_list_ignores_foreign_files(tmp_path: Path) -> None:
    day_tar(tmp_path, "20190601")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "RAD25_OPER_R___OTHER__L2__20190601T080500_20190602T080000_0001.tar").touch()
    units = KnmiRadar(archive_root=tmp_path).list(
        TimeWindow(datetime(2019, 6, 1, tzinfo=UTC), datetime(2019, 6, 3, tzinfo=UTC)),
        make_cfg(),
    )
    assert len(units) == 1


def test_archive_root_resolution_order(tmp_path: Path) -> None:
    source = SourceConfig(name="knmi_radar", extra={"archive_root": str(tmp_path / "from_source")})
    src = KnmiRadar(source)
    assert src.archive_root(make_cfg()) == tmp_path / "from_source"
    assert src.archive_root(make_cfg(archive_root=str(tmp_path / "from_product"))) == (
        tmp_path / "from_product"
    )
    assert KnmiRadar(source, archive_root=tmp_path / "explicit").archive_root(make_cfg()) == (
        tmp_path / "explicit"
    )
    with pytest.raises(ValueError, match="archive root"):
        KnmiRadar().archive_root(make_cfg())


# ------------------------------------------------------------------------ fetch


def test_fetch_publishes_hashes_and_is_idempotent(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    raw_root = tmp_path / "raw"
    path = day_tar(archive, "20190601")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    source = KnmiRadar(
        archive_root=archive,
        source=SourceConfig(name="knmi_radar", base_url=str(archive),
                            extra={"checksum": "sha256"}),
    )
    unit = source.list(
        TimeWindow(datetime(2019, 6, 1, tzinfo=UTC), datetime(2019, 6, 3, tzinfo=UTC)), make_cfg()
    )[0]

    raw = source.fetch(unit, raw_root)
    assert raw is not None
    assert raw.path == raw_root / "2019" / path.name  # one directory per year
    assert raw.sha256 == expected
    assert raw.n_bytes == path.stat().st_size
    assert raw.unit_key == unit.key
    # Hardlinked by default: the raw layer costs no bytes.
    assert raw.path.stat().st_ino == path.stat().st_ino
    assert not list(raw.path.parent.glob("*.part"))

    again = source.fetch(unit, raw_root)
    assert again is not None
    assert (again.path, again.sha256, again.n_bytes) == (raw.path, raw.sha256, raw.n_bytes)


def test_fetch_can_copy_instead_of_linking(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    path = day_tar(archive, "20190601")
    source = KnmiRadar(SourceConfig(name="knmi_radar", extra={"link_mode": "copy"}),
                       archive_root=archive)
    unit = source.list(
        TimeWindow(datetime(2019, 6, 1, tzinfo=UTC), datetime(2019, 6, 3, tzinfo=UTC)), make_cfg()
    )[0]
    raw = source.fetch(unit, tmp_path / "raw")
    assert raw is not None
    assert raw.path.stat().st_ino != path.stat().st_ino
    assert raw.path.read_bytes() == path.read_bytes()


def test_fetch_returns_none_when_the_tar_has_vanished(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    path = day_tar(archive, "20190601")
    source = KnmiRadar(archive_root=archive)
    unit = source.list(
        TimeWindow(datetime(2019, 6, 1, tzinfo=UTC), datetime(2019, 6, 3, tzinfo=UTC)), make_cfg()
    )[0]
    path.unlink()
    assert source.fetch(unit, tmp_path / "raw") is None


# ------------------------------------------------------------------------ parse


def test_parse_returns_a_gridded_dataset(tmp_path: Path) -> None:
    path = day_tar(tmp_path, "20190601", frames=4)
    ds = KnmiRadar().parse(raw_of(path), make_cfg())

    assert isinstance(ds, xr.Dataset)
    assert ds["precipitation_amount"].dims == ("time", "y", "x")
    assert ds["precipitation_amount"].shape == (4, NY, NX)
    assert ds["precipitation_amount"].dtype == np.float32
    assert ds["precipitation_amount"].attrs["units"] == "mm"
    assert list(ds["time"].values[:2]) == [
        np.datetime64("2019-06-01T08:05:00", "ns"),
        np.datetime64("2019-06-01T08:10:00", "ns"),
    ]
    # Frame i was filled with count i, calibrated at 0.01.
    np.testing.assert_allclose(ds["precipitation_amount"].values[3], 0.03, rtol=1e-6)


def test_parse_orders_frames_by_timestamp_not_by_tar_order(tmp_path: Path) -> None:
    frames = {
        "RAD_NL25_RAC_RT_201906010815.h5": frame_bytes(
            precip_frame(np.full((NY, NX), 3, dtype="uint16"))
        ),
        "RAD_NL25_RAC_RT_201906010805.h5": frame_bytes(
            precip_frame(np.full((NY, NX), 1, dtype="uint16"))
        ),
    }
    path = write_tar(tmp_path / tar_name("20190601"), frames)
    ds = KnmiRadar().parse(raw_of(path), make_cfg())
    assert ds["time"].values[0] == np.datetime64("2019-06-01T08:05:00", "ns")
    np.testing.assert_allclose(ds["precipitation_amount"].values[0], 0.01, rtol=1e-6)
    np.testing.assert_allclose(ds["precipitation_amount"].values[1], 0.03, rtol=1e-6)


def test_one_canonical_variable_spans_the_mid_archive_rename(tmp_path: Path) -> None:
    """PRECIP_[MM] and ACCUMULATION_[MM] are the same field under two names."""
    frames = {
        "RAD_NL25_RAC_RT_201906010805.h5": frame_bytes(
            precip_frame(np.full((NY, NX), 1, dtype="uint16"), legacy=True)
        ),
        "RAD_NL25_RAC_RT_201906010810.h5": frame_bytes(
            precip_frame(np.full((NY, NX), 2, dtype="uint16"), legacy=False)
        ),
    }
    path = write_tar(tmp_path / tar_name("20190601"), frames)
    ds = KnmiRadar().parse(raw_of(path), make_cfg(known_layouts=[]))
    values = ds["precipitation_amount"].values
    np.testing.assert_allclose(values[0], 0.01, rtol=1e-6)
    np.testing.assert_allclose(values[1], 0.02, rtol=1e-6)


def test_quality_is_carried_when_the_product_provides_one(tmp_path: Path) -> None:
    quality = np.full((NY, NX), 125, dtype="uint8")
    quality[0, 0] = 255  # out of image
    quality[0, 1] = 254  # missing, and not zero, so genuinely missing
    blob = frame_bytes(
        precip_frame(np.zeros((NY, NX), dtype="uint16"), legacy=False, quality=quality)
    )
    path = write_tar(tmp_path / tar_name("20230601"), {"RAD_NL25_RAC_RT_202306010805.h5": blob})
    ds = KnmiRadar().parse(raw_of(path), make_cfg())
    values = ds["quality"].values[0]
    assert np.isnan(values[0, 0])
    assert np.isnan(values[0, 1])
    assert values[1, 1] == pytest.approx(0.5)  # 125 * 0.004


def test_a_variable_with_no_image_is_dropped_not_written_as_nan(tmp_path: Path) -> None:
    """Pre-2023 frames have no quality image. Writing NaN would erase real data."""
    path = day_tar(tmp_path, "20190601")
    ds = KnmiRadar().parse(raw_of(path), make_cfg())
    assert "precipitation_amount" in ds.data_vars
    assert "quality" not in ds.data_vars


def test_a_frame_with_none_of_the_configured_images_is_drift(tmp_path: Path) -> None:
    blob = frame_bytes(
        [("image1", "REFLECTIVITY_[DBZ]", "GEO=0.5*PV+-32", 0, OUT_OF_IMAGE,
          np.zeros((NY, NX), dtype="uint16"))]
    )
    path = write_tar(tmp_path / tar_name("20190601"), {"RAD_NL25_RAC_RT_201906010805.h5": blob})
    with pytest.raises(SchemaDriftError, match="none of the configured images"):
        KnmiRadar().parse(raw_of(path), make_cfg(known_layouts=[]))


def test_an_empty_tar_parses_to_an_empty_dataset(tmp_path: Path) -> None:
    path = write_tar(tmp_path / tar_name("20190601"), {})
    ds = KnmiRadar().parse(raw_of(path), make_cfg())
    assert ds.sizes.get("time", 0) == 0


def test_valid_range_is_applied_in_canonical_units(tmp_path: Path) -> None:
    counts = np.full((NY, NX), 30000, dtype="uint16")  # 300 mm in five minutes
    values = _single_frame_values(tmp_path, counts, legacy=True)
    assert np.isnan(values).all()


# ------------------------------------------------------------- drift and scanning


def test_schema_fingerprint_is_stable_and_order_independent() -> None:
    a = ["PRECIP_[MM]|uint16|765x700", "QUALITY_[-]|uint8|765x700"]
    assert schema_fingerprint(a) == schema_fingerprint(list(reversed(a)))
    assert schema_fingerprint(a) == schema_fingerprint([*a, "", "  "])
    assert schema_fingerprint(a) != schema_fingerprint([a[0]])
    assert len(schema_fingerprint(a)) == 16


def test_fingerprint_moves_when_the_calibration_moves() -> None:
    base = ImageSpec("image1", "QUALITY_[-]", 0.004, 0.0, 254.0, 255.0, "uint8", (765, 700))
    rescaled = ImageSpec(
        "image1", "QUALITY_[-]", 1.52593e-05, 0.0, 0.0, 65535.0, "uint16", (765, 700)
    )
    assert schema_fingerprint([base.descriptor]) != schema_fingerprint([rescaled.descriptor])


def test_fingerprint_ignores_the_image_group_index() -> None:
    """image2 is the quality field in one generation and the adjustment in another."""
    a = ImageSpec("image2", "QUALITY_[-]", 0.004, 0.0, 254.0, 255.0, "uint8", (765, 700))
    b = ImageSpec("image3", "QUALITY_[-]", 0.004, 0.0, 254.0, 255.0, "uint8", (765, 700))
    assert schema_fingerprint([a.descriptor]) == schema_fingerprint([b.descriptor])


def test_an_unpinned_layout_is_reported_without_losing_the_day(tmp_path: Path) -> None:
    """Drift must be loud, and must not cost the data.

    Raising here used to discard the whole tar - 287 good frames for one odd
    one. On the real archive that lost 5,508 frames across 19 days, and the
    layouts involved were legitimate product generations: a uint8
    `PRECIP_[MM]` and a uint16 `QUALITY_[-]` at a different scale. Images are
    addressed by `image_geo_parameter`, so an unpinned layout still reads
    correctly; what is unknown is whether something else moved. So the frames
    are packed, the drift is recorded on the dataset, and `pack` fails on it.
    """
    path = day_tar(tmp_path, "20190601")
    ds = KnmiRadar().parse(raw_of(path), make_cfg(known_layouts=["0000000000000000"]))

    assert ds.sizes["time"] > 0, "the frames must survive the drift report"
    drift = ds.attrs.get("schema_drift")
    assert drift, "drift must be recorded on the dataset"
    assert any("not in the pinned set" in d for d in drift), drift


def test_a_config_that_disagrees_with_itself_still_raises(tmp_path: Path) -> None:
    """A different kind of failure, and it keeps the old treatment.

    An unpinned layout is a fact about the data. A `known_layouts` list that
    does not hash to `schema_fingerprint` is a fact about the config - somebody
    widened the allow-list without re-pinning - and nothing should be written
    against it.
    """
    path = day_tar(tmp_path, "20190601")
    cfg = make_cfg(known_layouts=["0000000000000000"]).model_copy(
        update={"schema_fingerprint": "deadbeefdeadbeef"}
    )
    with pytest.raises(SchemaDriftError, match="config is inconsistent"):
        KnmiRadar().parse(raw_of(path), cfg)


def test_a_pinned_layout_passes(tmp_path: Path) -> None:
    path = day_tar(tmp_path, "20190601")
    source = KnmiRadar()
    report = source.scan([raw_of(path)])
    pinned = report["fingerprints"]
    assert len(pinned) == 1
    ds = source.parse(raw_of(path), make_cfg(known_layouts=pinned))
    assert ds.sizes["time"] == 3


def test_known_layouts_and_schema_fingerprint_must_agree(tmp_path: Path) -> None:
    path = day_tar(tmp_path, "20190601")
    cfg = make_cfg(known_layouts=["0000000000000000"])
    cfg = cfg.model_copy(update={"schema_fingerprint": "deadbeefdeadbeef"})
    with pytest.raises(SchemaDriftError, match="config is inconsistent"):
        KnmiRadar().parse(raw_of(path), cfg)


def test_scan_reports_every_layout_and_survives_a_bad_file(tmp_path: Path) -> None:
    legacy = day_tar(tmp_path / "a", "20190601")
    modern = day_tar(tmp_path / "b", "20230601", legacy=False,
                     quality=np.zeros((NY, NX), dtype="uint8"))
    broken = tmp_path / "c" / tar_name("20240101")
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a tar at all")

    report = KnmiRadar().scan([raw_of(p) for p in (legacy, modern, broken)])
    assert report["source"] == "knmi_radar"
    assert report["n_files"] == 3
    assert report["n_variants"] == 2
    assert len(report["errors"]) == 1
    assert "c" in report["errors"][0]["path"]
    assert sorted(report["variants"][0]["parameters"]) == ["PRECIP_[MM]"]
    assert sorted(report["variants"][1]["parameters"]) == ["ACCUMULATION_[MM]", "QUALITY_[-]"]
    assert report["variants"][0]["rows"] == NY
    assert report["fingerprint"] == product_fingerprint(report["fingerprints"])


# ------------------------------------------------------- the gridded store accepts it


def test_parse_output_round_trips_through_the_real_gridded_store(tmp_path: Path) -> None:
    """The point of the exercise: a gridded source feeding `gridded.write` unchanged."""
    archive, root = tmp_path / "archive", tmp_path / "data"
    path = day_tar(archive, "20190601", frames=4)
    cfg = make_cfg()

    # The timeline starts at 08:05, which is what makes one tar one shard.
    timeline = Timeline(
        start=datetime(2019, 6, 1, 8, 5, tzinfo=UTC),
        end=datetime(2019, 6, 3, 8, 5, tzinfo=UTC),
        cadence=timedelta(seconds=CADENCE),
    )
    gridded.initialise(cfg, root, timeline, variables=["precipitation_amount"])

    ds = KnmiRadar().parse(raw_of(path), cfg)
    gridded.write(ds, cfg, root)

    back = gridded.read(cfg.store_id, root, time_range=(timeline.start, timeline.end))
    values = back["precipitation_amount"].values
    np.testing.assert_allclose(values[:4], ds["precipitation_amount"].values, rtol=1e-6)
    assert np.isnan(values[4:]).all()  # the rest of the axis is untouched

    info = gridded.info(cfg.store_id, root)
    assert info["shape"] == (len(timeline), NY, NX)
    assert info["chunks"] == (1, NY, NX)


def test_a_tar_lands_inside_one_shard_when_the_axis_starts_at_the_day_start() -> None:
    """A midnight axis would split every tar across two shards, and gridded.write
    is explicit that two tasks writing one shard lose frames."""
    cfg = make_cfg()
    first = datetime(2019, 6, 1, 8, 5, tzinfo=UTC)
    aligned = Timeline(first, first + timedelta(days=4), timedelta(seconds=CADENCE))
    frames_per_tar = 86_400 // CADENCE
    rows = aligned.index_of(
        [np.datetime64(first.replace(tzinfo=None) + timedelta(seconds=CADENCE * i), "ns")
         for i in range(frames_per_tar)]
    )
    shard = cfg.store.shard["time"]
    assert {int(r) // shard for r in rows} == {0}

    midnight = datetime(2019, 6, 1, tzinfo=UTC)
    unaligned = Timeline(midnight, midnight + timedelta(days=4), timedelta(seconds=CADENCE))
    rows = unaligned.index_of(
        [np.datetime64(first.replace(tzinfo=None) + timedelta(seconds=CADENCE * i), "ns")
         for i in range(frames_per_tar)]
    )
    assert len({int(r) // shard for r in rows}) == 2


# ------------------------------------------------------------- the real archive

pytestmark_archive = pytest.mark.skipif(
    not ARCHIVE.is_dir(), reason="the local KNMI radar archive is not mounted here"
)


@pytestmark_archive
def test_real_archive_listing_covers_the_window_and_survives_the_typo() -> None:
    """The archive really does contain `20181220T0080000`."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    source = KnmiRadar(archive_root=ARCHIVE)
    # 2018-12-19 is the first day in the archive, so the window deliberately
    # starts before it: a window reaching past the archive is not an error.
    units = source.list(
        TimeWindow(datetime(2018, 12, 17, tzinfo=UTC), datetime(2018, 12, 24, tzinfo=UTC)), cfg
    )
    assert [u.key.rsplit("/", 1)[-1] for u in units] == [
        "20181219", "20181220", "20181221", "20181222", "20181223",
    ]
    assert all(u.valid_time is not None for u in units)
    typo = next(u for u in units if u.key.endswith("20181220"))
    assert (typo.meta or {})["filename"].endswith("20181220T0080000_20181221T075500_0001.tar")
    assert typo.valid_time == datetime(2018, 12, 20, 8, 5, tzinfo=UTC)  # fallback, not a crash


@pytestmark_archive
def test_real_frames_match_the_pinned_layouts_and_the_declared_grid() -> None:
    """One real tar, one real frame: the fixtures above are shaped like the data."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    name = "RAD25_OPER_R___TARRRT__L2__20190601T080500_20190602T080000_0001.tar"
    tar = ARCHIVE / name
    if not tar.exists():  # pragma: no cover - depends on which days are mirrored
        pytest.skip(f"{name} is not in this copy of the archive")

    report = KnmiRadar().scan([raw_of(tar)], members_per_file=1)
    assert report["errors"] == []
    variant = report["variants"][0]
    assert variant["fingerprint"] in set(cfg.extra["known_layouts"])
    assert (variant["rows"], variant["columns"]) == grids.get(cfg.grid_id).shape == (765, 700)
    assert variant["parameters"] == ["ADJUSTMENT_FACTOR_[DB]", "PRECIP_[MM]"]
    # The scale is read from the file, and in this generation it really is 0.01.
    precip = next(i for i in variant["images"] if i.startswith("PRECIP_[MM]"))
    assert "scale=0.01" in precip and "out_of_image=65535.0" in precip
    assert "missing=0.0" in precip  # i.e. no echo, not missing


def test_a_frame_with_no_images_is_corrupt_not_drift(tmp_path, monkeypatch):
    """Several 2019 tars hold a readable HDF5 file with no image groups.

    Treating that as schema drift failed the whole tar and lost its other 287
    good frames; radar_v3 classified it as corrupt and carried on. The store
    has an is_corrupt_frame array precisely so this can be recorded rather than
    raised.
    """
    import numpy as np

    from meteopack.sources import knmi_radar

    assert knmi_radar.layout_id({}) == knmi_radar.LAYOUT_UNKNOWN
    source = inspect.getsource(knmi_radar.KnmiRadar.parse)
    assert "if not specs:" in source, "an imageless frame must be handled before the layout check"
    assert "LAYOUT_CORRUPT" in source
    assert isinstance(np.int16(knmi_radar.LAYOUT_CORRUPT), np.int16)


def test_the_digest_is_a_choice_because_hashing_is_a_full_read(tmp_path: Path) -> None:
    """`link_mode: hardlink` moves no bytes - the raw file and the archive file
    are the same inode - so hashing buys nothing against transfer error and
    costs a complete read. On the 127 GB radar archive that is ~36 minutes at
    the 60 MiB/s this filesystem gives. It still guards against later silent
    corruption, so it stays available; it is just no longer mandatory.
    """
    archive = tmp_path / "archive"
    path = day_tar(archive, "20190601")
    window = TimeWindow(datetime(2019, 6, 1, tzinfo=UTC), datetime(2019, 6, 3, tzinfo=UTC))

    def fetch_with(checksum: str, into: str):
        src = KnmiRadar(
            archive_root=archive,
            source=SourceConfig(name="knmi_radar", base_url=str(archive),
                                extra={"checksum": checksum}),
        )
        return src.fetch(src.list(window, make_cfg())[0], tmp_path / into)

    hashed = fetch_with("sha256", "a")
    assert hashed.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    cheap = fetch_with("size", "b")
    assert cheap.sha256.startswith("size:"), cheap.sha256
    assert str(path.stat().st_size) in cheap.sha256
    assert cheap.n_bytes == path.stat().st_size

    # The default is the cheap one: a published archive is re-fetched often.
    default = KnmiRadar(archive_root=archive)
    raw = default.fetch(default.list(window, make_cfg())[0], tmp_path / "c")
    assert raw.sha256.startswith("size:")

    with pytest.raises(ValueError, match="checksum must be"):
        fetch_with("md5", "d")


def test_parse_stage_summarises_gridded_files_instead_of_concatenating(tmp_path, capsys):
    """`meteopack parse` on a radar source handed xarray Datasets to polars.concat."""
    from types import SimpleNamespace

    from meteopack.cli import _parse_gridded

    counts = np.full((NY, NX), 7, dtype="uint16")
    paths = [
        write_tar(tmp_path / tar_name(day), {
            f"RAD_NL25_RAC_RT_{day}0805.h5": frame_bytes(precip_frame(counts, legacy=True)),
            f"RAD_NL25_RAC_RT_{day}0810.h5": frame_bytes(precip_frame(counts, legacy=True)),
        })
        for day in ("20190601", "20190602")
    ]
    ctx = SimpleNamespace(src=KnmiRadar(), cfg=make_cfg())
    assert _parse_gridded(ctx, paths) == 0
    out = capsys.readouterr().out
    assert out.count("2 frame(s)") == 2, out


@pytest.mark.parametrize(("parameter", "code"), [("HEIGHT_[KM]", 6), ("REFLECTIVITY_[DBZ]", 7)])
def test_echotop_and_reflectivity_frames_have_a_layout(parameter, code):
    """Every echotop frame in the store used to be 2, 'unknown'."""
    from meteopack.sources.knmi_radar import LAYOUT_NAMES, layout_id

    spec = ImageSpec("image1", parameter, 0.5, -32.0, 0.0, 255.0, "uint8", (NY, NX))
    assert layout_id({parameter: spec}) == code
    assert LAYOUT_NAMES[code] != "unknown"


def test_a_frame_with_a_calibration_listed_as_corrupt_is_withheld(tmp_path):
    """692 echotop frames carry a placeholder calibration (scale 1, or the
    reflectivity formula) over an image with no echo pixel at all, between
    neighbours with tens of thousands. Decoded, they claimed "no echo top
    anywhere"; they are failed products, and the store must say so.
    """
    from meteopack.sources.knmi_radar import (
        LAYOUT_CORRUPT,
        _image_specs,
        product_fingerprint,
    )

    counts = np.full((NY, NX), 7, dtype="uint16")
    good = frame_bytes(precip_frame(counts, legacy=True))
    broken = frame_bytes([("image1", "PRECIP_[MM]", "GEO = 1.000000 * PV + 0.000000",
                           0, OUT_OF_IMAGE, np.zeros((NY, NX), dtype="uint16"))])
    with h5py.File(io.BytesIO(broken), "r") as f:
        bad_fp = schema_fingerprint(sorted(s.descriptor for s in _image_specs(f).values()))
    with h5py.File(io.BytesIO(good), "r") as f:
        good_fp = schema_fingerprint(sorted(s.descriptor for s in _image_specs(f).values()))
    path = write_tar(tmp_path / tar_name("20190601"), {
        "RAD_NL25_RAC_RT_201906010805.h5": good, "RAD_NL25_RAC_RT_201906010810.h5": broken})

    cfg = make_cfg(known_layouts=[good_fp], corrupt_layouts=[bad_fp])
    cfg.schema_fingerprint = product_fingerprint([good_fp, bad_fp])
    ds = KnmiRadar().parse(raw_of(path), cfg)
    assert ds["is_corrupt_frame"].values.tolist() == [False, True]
    assert ds["source_layout_id"].values[1] == LAYOUT_CORRUPT
    assert np.isnan(ds["precipitation_amount"].values[1]).all()      # withheld, not "no echo"
    assert ds["precipitation_amount"].values[0, 1, 1] == pytest.approx(0.07)
    assert not ds.attrs.get("schema_drift"), "a layout listed as corrupt is not drift"
