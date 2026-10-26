"""Auswertung: Zeilen der Hauptansicht je Welt, Zellen des Rasters, Hilfsfunktionen."""

import math

import numpy as np
import pytest

import koi_algorithm as K
import koi_constants as C
import koi_evaluation as ev
from prs_scenario import VRng


def test_defaults_are_the_documented_standard_case():
    s = ev.Settings()
    assert (s.world, s.t, s.a, s.share, s.beta, s.gamma, s.coupling, s.driver, s.classes, s.seed) == ("ecm", 104, 0.2, 1.0, 1.0, 0.0, 0.2, 0.8, 3, 35)


def test_settings_are_keyword_only_and_hashable():
    with pytest.raises(TypeError):
        ev.Settings("ecm", 104)
    assert hash(ev.Settings()) == hash(ev.Settings()) and ev.Settings(t=52) != ev.Settings()


def test_make_series_gives_one_row_of_the_requested_length_and_rejects_unknown_worlds():
    for w in C.WORLDS:
        x, y = ev.make_series(ev.Settings(world=w, t=52))
        assert x.shape == y.shape == (1, 52)
    with pytest.raises(ValueError):
        ev.make_series(ev.Settings(world="stern"))


def test_the_ecm_analysis_has_the_documented_rows_and_matches_direct_calls():
    s = ev.Settings(world="ecm", t=104, a=0.3, share=0.5, beta=1.5, seed=4)
    a = ev.analyse(s)
    assert [r.key for r in a.rows] == ["naive", "diff", "df_x", "df_y", "eg", "beta", "alpha_y", "alpha_x", "spread"]
    x, y = ev.make_series(s)
    t, u, b = K.engle_granger(x, y)
    assert a.row("eg").value == t[0] and a.row("eg").threshold == K.critical_values(104)["eg"][1] and a.row("beta").value == b[0] and np.array_equal(a.spread, u[0])
    ty, ay, tx, ax = K.ecm_alpha(x, y, u)
    assert a.row("alpha_y").value == ay[0] and a.row("alpha_x").value == ax[0] and a.extra["t_alpha_y"] == ty[0]
    assert a.row("df_x").threshold == K.critical_values(104)["df"][1] and a.row("naive").value == K.naive_test(x, y)[0][0]
    assert a.row("spread").truth == f"{ev.de(math.sqrt(K.exact_spread_variance(0.3, 1.5, 0.0)))} (exakt, stationär)" and a.row("nichts") is None


def test_the_truth_column_names_what_holds_in_the_world():
    a = ev.analyse(ev.Settings(a=0.0))
    assert a.row("naive").truth == "keine Beziehung" and a.row("eg").truth == "nicht kointegriert" and a.row("spread").truth == "wandert (a = 0)" and a.row("alpha_y").truth == "α_y = 0,00"
    b = ev.analyse(ev.Settings(a=0.2, share=0.5, beta=2.0, gamma=0.3))
    assert b.row("naive").truth == "Beziehung vorhanden" and b.row("eg").truth == "kointegriert" and b.row("alpha_y").truth == "α_y = -0,10" and b.row("alpha_x").truth == "α_x = 0,05"
    assert b.row("diff").truth == "kurzfristige Kopplung γ = 0,3" and b.row("beta").truth == "β = 2,0"


def test_the_directed_worlds_have_six_rows_and_the_identity_holds():
    for w in ("coupled", "lagged"):
        a = ev.analyse(ev.Settings(world=w, classes=5))
        assert [r.key for r in a.rows] == ["granger_xy", "granger_yx", "te_xy", "te_yx", "plugin_xy", "plugin_yx"] and a.spread is None and a.cv is None
        assert a.row("te_xy").value == pytest.approx(0.5 * math.log1p(a.row("granger_xy").value / (a.extra["n"] - 3)), abs=1e-12)
        assert a.row("granger_xy").threshold == K.f_critical(a.extra["n"] - 3) and "5 Klassen" in a.row("plugin_xy").name and 0.0 < a.extra["plugin_xy_p"] <= 1.0
    assert ev.analyse(ev.Settings(world="coupled", coupling=0.4)).row("te_xy").truth == f"exakt {ev.de(K.exact_te(0.4), 4)} (+ {ev.de(K.gauss_te_bias(103), 4)} Überschuss im Mittel)"
    assert ev.analyse(ev.Settings(world="lagged")).row("te_xy").truth == "–" and ev.analyse(ev.Settings(world="lagged")).row("granger_xy").truth == "keine Wirkung von x auf y"


def test_the_analysis_is_reproducible_including_the_permutation_test():
    a, b = ev.analyse(ev.Settings(world="coupled", seed=9)), ev.analyse(ev.Settings(world="coupled", seed=9))
    assert [r.value for r in a.rows] == [r.value for r in b.rows] and a.extra["plugin_xy_p"] == b.extra["plugin_xy_p"]
    assert [r.value for r in ev.analyse(ev.Settings(world="coupled", seed=10)).rows] != [r.value for r in a.rows]


def test_de_and_pct_format_with_decimal_comma_and_dash():
    assert ev.de(0.5) == "0,50" and ev.de(None) == "–" and ev.de(float("nan")) == "–" and ev.de(float("inf")) == "–" and ev.de(-3.1669, 2) == "-3,17" and ev.pct(0.1234) == "12,3 %" and ev.pct(1.0, 0) == "100 %"


def test_cells_have_the_documented_structure_and_are_deterministic():
    kw = {"seeds": (5,), "batches": 1, "size": 40}
    p = ev.power_cell(104, 0.2, **kw)
    assert (p["T"], p["a"], p["runs"]) == (104, 0.2, 40) and 0.0 <= p["eg"] <= 1.0 and p["cut"] == K.critical_values(104)["eg"][1] and ev.power_cell(104, 0.2, **kw) == p
    n = ev.null_cell(52, **kw)
    assert n["runs"] == 40 and len(n["q"]) == 3 and n["q"][0] < n["q"][1] < n["q"][2] and n["cut_live"] == K.critical_values(52)["eg"][1]
    d = ev.diff_cell(0.5, 0.2, **kw)
    assert d["runs"] == 40 and d["diff"] > 0.8 and d["r_diff"] > 0.2
    ad = ev.adapt_cell(104, 0.2, 0.5, **kw)
    assert (ad["true_y"], ad["true_x"]) == (-0.1, pytest.approx(0.1)) and ad["runs"] == 40
    g = ev.granger_cell(0.4, **kw)
    assert g["runs"] == 40 and g["xy"] > g["yx"] and g["identity"] < 1e-12 and g["exact"] == K.exact_te(0.4) and g["bias"] == K.gauss_te_bias(103)
    b = ev.bias_cell(100, 3, seeds=(5,), size=30)
    assert b["runs"] == 30 and b["plugin"] > b["gauss"] > 0 and b["plugin_pred"] == K.plugin_te_bias(99, 3) and b["gauss_pred"] == K.gauss_te_bias(99)
    lg = ev.lagged_cell(0.8, **kw)
    assert lg["xy"] > 0.8 > 0.0 <= lg["yx"] and lg["runs"] == 40


def test_find_and_load_grid(tmp_path):
    cells = [{"T": 1, "a": 0.0}, {"T": 1, "a": 0.2}, {"T": 2, "a": 0.2}]
    assert ev.find(cells, T=2, a=0.2) is cells[2] and ev.find(cells, T=3) is None
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    bad = tmp_path / "kaputt.json"
    bad.write_text("{", encoding="utf-8")
    assert ev.load_grid(bad) is None
    good = tmp_path / "gut.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    assert ev.load_grid(good) == {"a": 1}


def test_batches_use_one_stream_per_seed_so_cells_do_not_depend_on_each_other():
    got = list(ev._batches(lambda size, rng: rng.uniform((size,)), (1, 2), 2, 3))
    assert len(got) == 4 and np.array_equal(got[0], VRng(1).uniform((3,))) and not np.array_equal(got[0], got[1]) and np.array_equal(got[2], VRng(2).uniform((3,)))
