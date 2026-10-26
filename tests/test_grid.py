"""Das vorgerechnete Raster (data/cointegration_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probe)."""

import pytest

import koi_algorithm as K
import koi_constants as C
import koi_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/cointegration_grid.json fehlt: python tools/make_grid.py"
    return g


def _assert_close(a, b, path="probe"):
    """Verschachtelte Wörter und Listen: Zahlen mit relativer Toleranz (Summationsreihenfolge je Plattform, DEMO-PLAYBOOK §4), alles andere exakt."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-10), path
    else:
        assert a == b, path


def test_probe_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Statistik oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    _assert_close(ev.grid_probe(), grid["probe"])


def test_grid_has_every_cell_exactly_once(grid):
    assert sorted((c["T"], c["a"]) for c in grid["power"]) == sorted((T, a) for T in C.E1_TS for a in C.E1_AS)
    assert sorted(c["T"] for c in grid["null"]) == sorted(C.E2_TS)
    assert sorted((c["gamma"], c["a"]) for c in grid["diff"]) == sorted((g, a) for g in C.E3_GAMMAS for a in C.E3_AS)
    assert sorted((c["T"], c["a"], c["share"]) for c in grid["adapt"]) == sorted((T, a, s) for T in C.E4_TS for a in C.E4_AS for s in C.E4_SHARES)
    assert sorted(c["c"] for c in grid["granger"]) == sorted(C.E5_COUPLINGS) and sorted((c["T"], c["k"]) for c in grid["bias"]) == sorted((T, k) for T in C.E6_TS for k in C.E6_KS)
    assert sorted(c["b"] for c in grid["lagged"]) == [0.0, 0.4, 0.8, 1.2]


def test_grid_meta_and_run_counts(grid):
    total = len(C.GRID_SEEDS) * C.BATCHES * C.BATCH
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS)
    for key in ("power", "null", "diff", "adapt", "granger", "lagged"):
        assert all(c["runs"] == total for c in grid[key]), key
    assert all(c["runs"] == len(C.GRID_SEEDS) * C.E6_BATCH for c in grid["bias"])


def test_shares_are_in_the_unit_interval_and_the_stored_exact_values_are_fresh(grid):
    for c in grid["power"]:
        assert 0.0 <= c["eg"] <= 1.0 and 0.0 <= c["naive"] <= 1.0 and c["cut"] == pytest.approx(K.critical_values(c["T"])["eg"][1], abs=1e-9)
    for c in grid["granger"]:
        assert c["exact"] == pytest.approx(K.exact_te(c["c"]), abs=1e-12) and c["bias"] == pytest.approx(K.gauss_te_bias(C.E5_T - 1), abs=1e-15) and c["identity"] < 1e-12
    for c in grid["adapt"]:
        assert (c["true_y"], c["true_x"]) == tuple(v + 0.0 for v in K.exact_alphas(c["a"], c["share"], 1.0)) or (c["true_y"], c["true_x"]) == K.exact_alphas(c["a"], c["share"], 1.0)
    for c in grid["bias"]:
        n = c["T"] - 1
        assert c["gauss_pred"] == K.gauss_te_bias(n) and c["plugin_pred"] == K.plugin_te_bias(n, c["k"])


def test_the_null_cells_agree_with_the_asymptotic_table_and_show_the_small_sample_excess(grid):
    q = {c["T"]: c for c in grid["null"]}
    assert q[520]["q"][1] == pytest.approx(-3.3377, abs=0.06) and q[26]["q"][1] < q[104]["q"][1] < q[520]["q"][1] + 0.1 and q[26]["alarm_table"] > 0.07 and abs(q[520]["alarm_table"] - 0.05) < 0.012
    assert all(abs(c["alarm_live"] - 0.05) < 0.02 for c in grid["null"])


def test_the_headline_numbers_of_the_experiments(grid):
    p = {(c["T"], c["a"]): c for c in grid["power"]}
    assert p[(104, 0.0)]["naive"] > 0.7 and p[(104, 0.0)]["eg"] < 0.07 and p[(104, 0.2)]["naive"] > 0.99 and 0.5 < p[(104, 0.2)]["eg"] < 0.75 and p[(260, 0.05)]["eg"] > p[(104, 0.05)]["eg"]
    d = {(c["gamma"], c["a"]): c for c in grid["diff"]}
    assert d[(0.0, 0.2)]["diff"] < 0.07 and d[(0.0, 0.2)]["eg"] > 0.5 and d[(0.5, 0.0)]["diff"] > 0.98 and d[(0.5, 0.0)]["eg"] < 0.07
    ad = {(c["T"], c["a"], c["share"]): c for c in grid["adapt"]}
    assert ad[(104, 0.2, 1.0)]["sig_y"] > 0.99 and ad[(104, 0.2, 1.0)]["sig_x"] < 0.1 and ad[(104, 0.2, 0.0)]["sig_x"] > 0.97 and ad[(104, 0.2, 0.0)]["sig_y"] > 0.1
    g = {c["c"]: c for c in grid["granger"]}
    assert g[0.2]["xy"] > 0.55 and abs(g[0.2]["yx"] - 0.05) < 0.02 and g[0.0]["xy"] == pytest.approx(0.05, abs=0.02)
    b = {(c["T"], c["k"]): c for c in grid["bias"]}
    assert b[(100, 3)]["plugin"] > 0.06 > K.exact_te(0.2) and b[(100, 3)]["plugin"] > 2 * K.exact_te(0.2)
    lg = {c["b"]: c for c in grid["lagged"]}
    assert lg[0.8]["xy"] > 0.99 and lg[0.8]["yx"] < 0.12 and lg[0.0]["xy"] == pytest.approx(0.05, abs=0.02)
