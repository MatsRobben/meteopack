"""Integer canonical encoding: exactness, sentinels, and the two kinds of gap."""
from __future__ import annotations

import numpy as np
import pytest

from meteopack.core.encoding import Encoding, default_sentinels, encode

RTCOR = Encoding(dtype=np.dtype("uint16"), scale=0.01, offset=0.0,
                 missing=65534, out_of_image=65535)
QUALITY = Encoding(dtype=np.dtype("uint8"), scale=0.004, offset=0.0,
                   missing=254, out_of_image=255)


def test_default_sentinels_are_the_top_two_codes():
    assert default_sentinels(np.dtype("uint16")) == (65534, 65535)
    assert default_sentinels(np.dtype("uint8")) == (254, 255)


def test_same_calibration_round_trips_exactly():
    """The common case: the file already uses the canonical calibration.

    Every ordinary code must come back as itself. Anything less means the store
    is lossier than the file it was built from, which is the whole objection to
    decoding to float32.
    """
    codes = np.arange(0, 65534, dtype="uint16")
    out = encode(codes, RTCOR, source_scale=0.01, source_offset=0.0)
    np.testing.assert_array_equal(out, codes)


def test_a_coarser_source_calibration_lands_on_the_right_codes():
    """uint8 at 0.1 mm into uint16 at 0.01 mm: 1 count becomes 10."""
    codes = np.array([0, 1, 2, 17], dtype="uint8")
    out = encode(codes, RTCOR, source_scale=0.1, source_offset=0.0)
    np.testing.assert_array_equal(out, np.array([0, 10, 20, 170], dtype="uint16"))


def test_an_offset_calibration_is_carried_across():
    target = Encoding(dtype=np.dtype("uint8"), scale=0.16, offset=-20.24,
                      missing=254, out_of_image=255)
    codes = np.array([0, 1, 100], dtype="uint8")
    out = encode(codes, target, source_scale=0.16, source_offset=-20.24)
    np.testing.assert_array_equal(out, codes)
    np.testing.assert_allclose(target.decode(out)[:3], [-20.24, -20.08, -4.24], atol=1e-5)


def test_the_two_sentinels_stay_apart():
    """`missing` and `out_of_image` are different facts and must not merge.

    Collapsing both to one value (or to NaN) cannot be undone, and it is the
    difference between a pixel to inpaint and a pixel that does not exist.
    """
    codes = np.array([0, 5, 200, 201], dtype="uint8")
    out = encode(codes, RTCOR, source_scale=0.01, source_offset=0.0,
                 source_missing={200}, source_out_of_image={201})
    assert out[2] == RTCOR.missing
    assert out[3] == RTCOR.out_of_image
    assert out[0] == 0 and out[1] == 5

    decoded = RTCOR.decode(out)
    assert np.isnan(decoded[2]) and np.isnan(decoded[3])
    assert decoded[0] == 0.0


def test_a_zero_missing_code_can_be_excluded_by_the_caller():
    """Older rtcor advertises calibration_missing_data=0 while 0 means dry.

    Honouring it literally erases the dry radar mask - every no-rain pixel in
    the country - so the source decides, and this checks both readings work.
    """
    codes = np.array([0, 1, 2], dtype="uint16")

    literal = encode(codes, RTCOR, source_scale=0.01, source_offset=0.0,
                     source_missing={0})
    assert literal[0] == RTCOR.missing

    kept = encode(codes, RTCOR, source_scale=0.01, source_offset=0.0,
                  source_missing=set())
    assert kept[0] == 0


def test_a_value_is_never_quietly_turned_into_a_sentinel():
    """Saturation must clip to the last ordinary code, not to `missing`.

    A pixel wrong by one count is a small error; a pixel reported as missing is
    a different kind of claim.
    """
    huge = np.array([60000, 65000], dtype="uint16")
    out = encode(huge, RTCOR, source_scale=1.0, source_offset=0.0)
    assert out.max() == min(RTCOR.reserved) - 1 == 65533
    assert not np.isin(out, list(RTCOR.reserved)).any()


def test_out_of_image_wins_when_a_file_reserves_one_code_for_both():
    codes = np.array([7], dtype="uint8")
    out = encode(codes, QUALITY, source_scale=0.004, source_offset=0.0,
                 source_missing={7}, source_out_of_image={7})
    assert out[0] == QUALITY.out_of_image


def test_encoding_from_config_defaults_the_sentinels():
    enc = Encoding.from_config({"dtype": "uint16", "scale": 0.01}, name="x")
    assert (enc.missing, enc.out_of_image) == (65534, 65535)
    assert enc.formula == "GEO=0.010000*PV+0.000000"
    assert Encoding.from_config(None) is None
    with pytest.raises(ValueError, match="integer dtype"):
        Encoding.from_config({"dtype": "float32"}, name="x")


def test_the_lookup_table_matches_direct_arithmetic_over_every_code():
    """The table is an optimisation; it must agree with the obvious method."""
    codes = np.arange(0, 256, dtype="uint8")
    out = encode(codes, RTCOR, source_scale=0.1, source_offset=-1.0,
                 source_missing={254}, source_out_of_image={255})
    expect = np.rint((codes.astype("float64") * 0.1 - 1.0) / 0.01).astype("int64")
    expect = np.clip(expect, 0, 65533).astype("uint16")
    expect[254] = RTCOR.missing
    expect[255] = RTCOR.out_of_image
    np.testing.assert_array_equal(out, expect)


def test_float_pixels_take_the_same_rules_as_integer_ones():
    """Some generations write float imagery. It must encode, not raise.

    A NaN pixel is the file saying it has no value here, which is what
    `missing` means; it must not become 0 mm of rain.
    """
    values = np.array([0.0, 1.0, 2.0, np.nan, -999.0], dtype="float32")
    out = encode(values, RTCOR, source_scale=0.01, source_offset=0.0,
                 source_missing={-999})
    np.testing.assert_array_equal(out[:3], np.array([0, 1, 2], dtype="uint16"))
    assert out[3] == RTCOR.missing
    assert out[4] == RTCOR.missing
    assert np.isnan(RTCOR.decode(out)[3])


def test_wide_integers_do_not_build_a_four_billion_entry_table():
    values = np.array([0, 100, 12345], dtype="int32")
    out = encode(values, RTCOR, source_scale=0.01, source_offset=0.0)
    np.testing.assert_array_equal(out, np.array([0, 100, 12345], dtype="uint16"))
