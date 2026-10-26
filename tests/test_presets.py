"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung."""

import pytest

import koi_constants as C
import koi_evaluation as ev
import koi_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert p["world"] in C.WORLDS and p["t"] in C.T_OPTIONS and C.A_MIN <= p["a"] <= C.A_MAX and p["share"] in C.SHARES and p["beta"] in C.BETA_OPTIONS, name
        assert C.GAMMA_MIN <= p["gamma"] <= C.GAMMA_MAX and C.COUPLING_MIN <= p["coupling"] <= C.COUPLING_MAX and C.DRIVER_MIN <= p["driver"] <= C.DRIVER_MAX, name
        assert p["classes"] in C.CLASS_OPTIONS and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()
    assert next(iter(C.PRESETS)) == "Standardfall (Voreinstellung)"


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_differ_and_cover_all_three_worlds():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values) and {p["world"] for p in C.PRESETS.values()} == set(C.WORLDS)


@pytest.mark.parametrize("key, raw, expected", [
    ("world_select", "coupled", "coupled"), ("world_select", "stern", None), ("t_select", "260", 260), ("t_select", "100", None),
    ("a_slider", "9", C.A_MAX), ("a_slider", "-1", C.A_MIN), ("a_slider", "0.337", 0.34), ("a_slider", "abc", None),
    ("share_select", "0.5", 0.5), ("share_select", "0.3", None), ("beta_select", "1.5", 1.5), ("beta_select", "3", None),
    ("gamma_slider", "2", C.GAMMA_MAX), ("coupling_slider", "-3", C.COUPLING_MIN), ("driver_slider", "9", C.DRIVER_MAX),
    ("classes_select", "5", 5), ("classes_select", "4", None), ("seed_input", "99999999", C.SEED_MAX), ("seed_input", "-5", 0),
])
def test_permalink_values_are_cleaned(key, raw, expected):
    assert P.clean_value(P.SETTING_SPECS[key], raw) == expected
