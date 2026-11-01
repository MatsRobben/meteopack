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


def test_near_tie_values_are_resolved_in_float64_not_float32():
    """Where this deliberately differs from the radar_v3 pipeline.

    That pipeline did the conversion in float32. Re-encoding the 2018 gauge
    adjustment factor (source ``GEO=0.001*PV-32.768``, target 0.16/-20.24) puts
    one input code in every 160 at a value that is a .5 midpoint in exact
    arithmetic, and whether it rounds up or down is then decided by
    representation error - 1e-7 in float32 against 1e-14 in float64.

    Checked against the real store: 531 of 535,500 cells in one frame differ by
    exactly one count, every one at such a near-tie, and both answers sit 0.5
    counts (0.08 dB) from the true value. Neither is more accurate for those
    pixels. float64 is used anyway because it reproduces the intended value
    1e7 times more closely everywhere else, and because the result does not
    depend on the order the arithmetic happens to be done in.
    """
    target = Encoding(dtype=np.dtype("uint8"), scale=0.16, offset=-20.24,
                      missing=254, out_of_image=255)
    # code = (0.001*PV - 32.768 + 20.24) / 0.16 = 0.00625*PV - 78.3
    midpoints = np.array([12608, 12768, 12928, 13088], dtype="uint16")
    exact = 0.00625 * midpoints.astype("float64") - 78.3
    assert np.allclose(exact - np.floor(exact), 0.5, atol=1e-9), "fixture is not near the ties"

    out = encode(midpoints, target, source_scale=0.001, source_offset=-32.768)

    # The contract is the two-step route the implementation takes: decode with
    # the source calibration, re-encode with the target's, round with np.rint.
    decoded = midpoints.astype("float64") * 0.001 + (-32.768)
    expected = np.rint((decoded - target.offset) / target.scale).astype("uint8")
    np.testing.assert_array_equal(out, expected)

    # And here is the point, visible in the fixture itself: collapsing those
    # two steps algebraically into `0.00625 * PV - 78.3` gives a different
    # answer for every one of these codes, because neither 78.3 nor 0.16 is
    # representable and the two routes land on opposite sides of the tie. At a
    # mathematical tie the result is decided by representation, whichever
    # precision is used - which is why matching radar_v3 cell for cell here is
    # not a meaningful goal.
    shortcut = np.rint(0.00625 * midpoints.astype("float64") - 78.3).astype("uint8")
    assert not np.array_equal(out, shortcut)
    assert np.abs(out.astype("float64") - exact).max() <= 0.5 + 1e-9
    assert np.abs(shortcut.astype("float64") - exact).max() <= 0.5 + 1e-9


def test_a_modern_rtcor_frame_re_encodes_to_itself():
    """The canonical calibration is the product's own, so the 2023+ generation
    is a no-op. Verified against radar_v3 on real data: 12.85 million cells per
    variable over 24 frames, zero differing.
    """
    codes = np.array([0, 1, 1234, 65533, 65534, 65535], dtype="uint16")
    out = encode(codes, RTCOR, source_scale=0.01, source_offset=0.0,
                 source_missing={65534}, source_out_of_image={65535})
    np.testing.assert_array_equal(out, codes)


def test_cf_standard_names_are_only_claimed_where_they_are_real():
    """A standard_name the CF table does not define is worse than none.

    Qualified variants - air_temperature_min, soil_temperature_10cm,
    precipitation_amount_1h - are not CF standard names: CF expresses those
    with cell methods and coordinates. They get no standard_name rather than an
    invented one.
    """
    from meteopack.core.config import Variable

    def var(name, **kw):
        return Variable(name=name, source_key="x", units="1", **kw)

    assert var("air_temperature").cf_standard_name == "air_temperature"
    assert var("air_temperature_min").cf_standard_name is None
    assert var("soil_temperature").cf_standard_name == "soil_temperature"
    assert var("soil_temperature_10cm").cf_standard_name is None
    assert var("ir108_brightness_temperature").cf_standard_name is None

    # An alias: our shorter column name, CF's longer attribute.
    flux = var("surface_downwelling_shortwave_flux")
    assert flux.cf_standard_name == "surface_downwelling_shortwave_flux_in_air"

    # An explicit one always wins.
    assert var("anything", standard_name="toa_outgoing_shortwave_flux").cf_standard_name == (
        "toa_outgoing_shortwave_flux")


def test_every_configured_variable_names_its_units():
    """Columns of numbers with no units is what the points stores were."""
    import glob

    import yaml

    from meteopack.core.config import Variable

    for path in sorted(glob.glob("configs/products/*/*.yaml")):
        with open(path) as handle:
            spec_file = yaml.safe_load(handle)
        for spec in spec_file.get("variables") or []:
            var = Variable(**spec)
            assert var.units is not None, f"{path}: {var.name} has no units"
            assert var.cf_attrs["long_name"], f"{path}: {var.name} has no long_name"
