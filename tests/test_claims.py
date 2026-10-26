"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/cointegration_grid.json) oder über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 mit festen Seeds. Die Tabellen der README werden Zeile für Zeile in der Rundung der App (`_c`, `_p`) gegen die README geprüft. Die Zeilen-Bauer (`power_rows` usw.) und
`numbers()` dienen auch dazu, die README-Tabellen und -Zahlen zu erzeugen."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import koi_algorithm as K
import koi_constants as C
import koi_evaluation as ev

ROOT = Path(__file__).resolve().parent.parent
README_PATH = ROOT / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""


def _c(v, d):
    return f"{round(v, d) + 0.0:.{d}f}".replace(".", ",")                  # kein „-0,000“


def _p(v, d=1):
    return f"{100 * v:.{d}f}".replace(".", ",") + " %"


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


def cell(grid, group, **kw):
    c = ev.find(grid[group], **kw)
    assert c is not None, (group, kw)
    return c


# --- Zeilen-Bauer (Rundung der App) ------------------------------------------------------------------------------------------------------------

def power_rows(grid) -> list[str]:
    return [f"| {c['T']} | {_c(c['a'], 2)} | {_c(1 - c['a'], 2)} | {_p(c['eg'])} | {_p(c['naive'])} | {_p(c['df_x'])} |" for c in sorted(grid["power"], key=lambda c: (c["T"], c["a"]))]


def null_rows(grid) -> list[str]:
    return [f"| {c['T']} | {_c(c['q'][0], 3)} / {_c(c['q'][1], 3)} / {_c(c['q'][2], 3)} | {_p(c['alarm_table'])} | {_c(c['cut_live'], 3)} | {_p(c['alarm_live'])} | {_c(c['df_q05'], 3)} |"
            for c in sorted(grid["null"], key=lambda c: c["T"])]


def diff_rows(grid) -> list[str]:
    return [f"| {_c(c['gamma'], 1)} | {_c(c['a'], 1)} | {_p(c['diff'])} | {_p(c['eg'])} | {_c(c['r_diff'], 3)} |" for c in sorted(grid["diff"], key=lambda c: (c["gamma"], c["a"]))]


def adapt_rows(grid) -> list[str]:
    return [f"| {c['T']}, {_c(c['a'], 1)} | {C.SHARES[c['share']]} | {_p(c['sig_y'])} | {_p(c['sig_x'])} | {_c(c['alpha_y'], 3)} ({_c(c['true_y'], 3)}) | {_c(c['alpha_x'], 3)} ({_c(c['true_x'], 3)}) |"
            for c in sorted(grid["adapt"], key=lambda c: (c["T"], c["a"], -c["share"]))]


def granger_rows(grid) -> list[str]:
    return [f"| {_c(c['c'], 1)} | {_p(c['xy'])} | {_p(c['yx'])} | {_c(c['te_xy'], 4)} ({_c(c['exact'], 4)}) | {_c(c['te_yx'], 4)} |" for c in sorted(grid["granger"], key=lambda c: c["c"])]


def bias_rows(grid) -> list[str]:
    rows = []
    for T in C.E6_TS:
        c3, c5 = cell(grid, "bias", T=T, k=3), cell(grid, "bias", T=T, k=5)
        rows.append(f"| {T} | {_c(c3['gauss'], 4)} ({_c(c3['gauss_pred'], 4)}) | {_c(c3['plugin'], 4)} ({_c(c3['plugin_pred'], 4)}) | {_c(c5['plugin'], 4)} ({_c(c5['plugin_pred'], 4)}) |")
    return rows


def lagged_rows(grid) -> list[str]:
    return [f"| {_c(c['b'], 1)} | {_p(c['xy'])} | {_p(c['yx'])} |" for c in sorted(grid["lagged"], key=lambda c: c["b"])]


def numbers(grid) -> dict:
    """Die Zahlen im Fließtext der README (Name -> Zeichenkette in der Rundung der App)."""
    p = {(c["T"], c["a"]): c for c in grid["power"]}
    n = {c["T"]: c for c in grid["null"]}
    d = {(c["gamma"], c["a"]): c for c in grid["diff"]}
    ad = {(c["T"], c["a"], c["share"]): c for c in grid["adapt"]}
    g = {c["c"]: c for c in grid["granger"]}
    b = {(c["T"], c["k"]): c for c in grid["bias"]}
    lg = {c["b"]: c for c in grid["lagged"]}
    return {
        "n0_104": _p(p[(104, 0.0)]["naive"]), "e0_104": _p(p[(104, 0.0)]["eg"]), "n2_104": _p(p[(104, 0.2)]["naive"]), "e2_104": _p(p[(104, 0.2)]["eg"]), "e05_104": _p(p[(104, 0.05)]["eg"]),
        "e05_260": _p(p[(260, 0.05)]["eg"]), "e1_260": _p(p[(260, 0.1)]["eg"]), "n0_260": _p(p[(260, 0.0)]["naive"]), "e2_260": _p(p[(260, 0.2)]["eg"]), "df_x": _p(p[(104, 0.2)]["df_x"]),
        "q5_26": _c(n[26]["q"][1], 2), "q5_520": _c(n[520]["q"][1], 2), "alarm26": _p(n[26]["alarm_table"]), "alarm520": _p(n[520]["alarm_table"]),
        "live_min": _p(min(c["alarm_live"] for c in grid["null"])), "live_max": _p(max(c["alarm_live"] for c in grid["null"])),
        "d00_diff": _p(d[(0.0, 0.0)]["diff"]), "d02_diff": _p(d[(0.0, 0.2)]["diff"]), "d02_eg": _p(d[(0.0, 0.2)]["eg"]), "d50_diff": _p(d[(0.5, 0.0)]["diff"]), "d52_diff": _p(d[(0.5, 0.2)]["diff"]),
        "d52_eg": _p(d[(0.5, 0.2)]["eg"]), "d50_r": _c(d[(0.5, 0.0)]["r_diff"], 3),
        "a_y1": _p(ad[(104, 0.2, 1.0)]["sig_y"]), "a_x1": _p(ad[(104, 0.2, 1.0)]["sig_x"]), "a_y0": _p(ad[(104, 0.2, 0.0)]["sig_y"]), "a_x0": _p(ad[(104, 0.2, 0.0)]["sig_x"]),
        "a_ay1": _c(ad[(104, 0.2, 1.0)]["alpha_y"], 3), "a_ay5": _c(ad[(104, 0.2, 0.5)]["alpha_y"], 3), "a_y05": _p(ad[(104, 0.2, 0.5)]["sig_y"]), "a_x05": _p(ad[(104, 0.2, 0.5)]["sig_x"]),
        "g_xy": _p(g[0.2]["xy"]), "g_yx": _p(g[0.2]["yx"]), "g_te": _c(g[0.2]["te_xy"], 4), "g_ex": _c(g[0.2]["exact"], 4), "g_bias": _c(g[0.2]["bias"], 4), "g_te0": _c(g[0.0]["te_xy"], 4),
        "g_xy0": _p(g[0.0]["xy"]), "g_id": f"{max(c['identity'] for c in grid['granger']):.0e}",
        "b100_3": _c(b[(100, 3)]["plugin"], 4), "b100_pred": _c(b[(100, 3)]["plugin_pred"], 4), "b100_g": _c(b[(100, 3)]["gauss"], 4), "te_c02": _c(K.exact_te(0.2), 4), "b50_5": _c(b[(50, 5)]["plugin"], 4),
        "l_xy": _p(lg[0.8]["xy"]), "l_yx": _p(lg[0.8]["yx"]), "l_xy0": _p(lg[0.0]["xy"]), "grid_seconds": str(grid["meta"]["seconds"]), "runs": f"{cell(grid, 'power', T=104, a=0.0)['runs']:,}".replace(",", " "),
        "bias_runs": f"{cell(grid, 'bias', T=100, k=3)['runs']:,}".replace(",", " "),
    }


# --- Tabellen und Zahlen der README --------------------------------------------------------------------------------------------------------------

def test_claim_readme_exists_and_has_the_required_sections():
    assert README, "README.md fehlt"
    for heading in ("## Warum dieses Problem", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Ehrliche Grenzen", "## Verifikation", "## Dateistruktur", "## Bewusst nicht umgesetzt"):
        assert heading in README, heading
    assert "TODO" not in README.replace("<!-- TODO nach dem Deploy", "")


@pytest.mark.parametrize("builder", [power_rows, null_rows, diff_rows, adapt_rows, granger_rows, bias_rows, lagged_rows])
def test_claim_readme_tables_match_the_grid_row_by_row(grid, builder):
    for row in builder(grid):
        assert row in README, row


def test_claim_readme_numbers_in_the_text(grid):
    for name, text in numbers(grid).items():
        assert text in README, (name, text)


def test_claim_the_headline_findings_hold_in_the_grid(grid):
    p = {(c["T"], c["a"]): c for c in grid["power"]}
    assert p[(104, 0.0)]["naive"] > 0.7 and p[(104, 0.0)]["eg"] < 0.07 and p[(104, 0.2)]["naive"] > 0.99 and 0.5 < p[(104, 0.2)]["eg"] < 0.75
    assert p[(104, 0.05)]["eg"] < 0.1 and p[(260, 0.05)]["eg"] > 0.15
    d = {(c["gamma"], c["a"]): c for c in grid["diff"]}
    assert d[(0.0, 0.2)]["diff"] < 0.07 and d[(0.5, 0.0)]["diff"] > 0.98 and d[(0.5, 0.0)]["eg"] < 0.07
    ad = {(c["T"], c["a"], c["share"]): c for c in grid["adapt"]}
    assert ad[(104, 0.2, 1.0)]["sig_y"] > 0.99 and ad[(104, 0.2, 1.0)]["sig_x"] < 0.1 and ad[(104, 0.2, 0.0)]["sig_y"] > 0.1 and ad[(104, 0.2, 1.0)]["alpha_y"] < ad[(104, 0.2, 1.0)]["true_y"]
    lg = {c["b"]: c for c in grid["lagged"]}
    assert lg[0.8]["xy"] > 0.99 and lg[0.8]["yx"] < 0.12
    b = {(c["T"], c["k"]): c for c in grid["bias"]}
    assert b[(100, 3)]["plugin"] > 2 * K.exact_te(0.2) and b[(100, 3)]["gauss"] < K.exact_te(0.2) / 3


# --- Texte der Presets ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyses():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


def _t(a, key):
    return _c(a.row(key).value, 2)


def test_claim_the_preset_help_texts_quote_the_analysis_and_the_grid(analyses, grid):
    h = C.PRESET_HELP
    a = analyses["Standardfall (Voreinstellung)"]
    cut = _c(a.row("eg").threshold, 2)
    assert (f"der naive Test nennt r = {_t(a, 'naive')} „signifikant“, der Engle–Granger-Test findet die Beziehung in dieser Stichprobe nicht (t = {_t(a, 'eg')} bei der Schwelle {cut}), im Raster aber in "
            f"{_p(cell(grid, 'power', T=104, a=0.2)['eg'])} der Läufe") in h["Standardfall (Voreinstellung)"] and a.extra["sig"] and a.row("eg").value > a.row("eg").threshold
    a = analyses["Keine Beziehung"]
    assert f"(r = {_t(a, 'naive')}), der Engle–Granger-Test findet nichts (t = {_t(a, 'eg')} bei der Schwelle {_c(a.row('eg').threshold, 2)})" in h["Keine Beziehung"] and a.extra["sig"]
    a = analyses["Langsame Anpassung"]
    assert (f"(t = {_t(a, 'eg')} bei der Schwelle {_c(a.row('eg').threshold, 2)}); im Raster findet er sie nur in {_p(cell(grid, 'power', T=260, a=0.05)['eg'])} der Läufe" in h["Langsame Anpassung"]
            and a.settings.t == 260 and a.row("eg").value > a.row("eg").threshold)
    a = analyses["Beide passen sich an"]
    assert (f"α̂_y = {_t(a, 'alpha_y')} und α̂_x = {_t(a, 'alpha_x')} (wahr {_c(K.exact_alphas(0.2, 0.5, 1.0)[0], 2)} und {_c(K.exact_alphas(0.2, 0.5, 1.0)[1], 2)}); beide t-Tests liegen in dieser Stichprobe knapp "
            f"über 1,96 (t = {_c(a.extra['t_alpha_y'], 2)} und {_c(a.extra['t_alpha_x'], 2)})") in h["Beide passen sich an"] and abs(a.extra["t_alpha_y"]) > 1.96 and abs(a.extra["t_alpha_x"]) > 1.96
    a = analyses["Kurzfristige Kopplung"]
    assert f"ist „signifikant“ (r = {_c(a.extra['r_diff'], 2)}), der Engle–Granger-Test findet nichts (t = {_t(a, 'eg')})" in h["Kurzfristige Kopplung"] and a.extra["sig_diff"] and a.row("eg").value > a.row("eg").threshold
    a = analyses["Gekoppelte Reihen"]
    assert (f"Granger-F x → y {_c(a.extra['F_xy'], 2)} (Schwelle {_c(a.extra['crit'], 2)}), y → x {_c(a.extra['F_yx'], 2)}; gaußsche Transferentropie {_c(a.extra['te_xy'], 4)} (exakt {_c(K.exact_te(0.2), 4)} plus "
            f"{_c(K.gauss_te_bias(a.extra['n']), 4)} Überschuss); der Plug-in-Schätzer mit 3 Klassen findet die Richtung in dieser Stichprobe nicht (p = {_c(a.extra['plugin_xy_p'], 3)})") in h["Gekoppelte Reihen"]
    assert a.extra["plugin_xy_p"] > 0.05 and a.extra["F_xy"] > a.extra["crit"] > a.extra["F_yx"]
    a = analyses["Verzögerte Ursache"]
    assert f"Granger-F x → y ist {_c(a.extra['F_xy'], 2)} (Schwelle {_c(a.extra['crit'], 2)}) und y → x {_c(a.extra['F_yx'], 2)}" in h["Verzögerte Ursache"] and a.extra["F_xy"] > a.extra["crit"] > a.extra["F_yx"]


def test_claim_hand_values_of_the_text():
    x, y = np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([[2.0, 4.0, 5.0, 8.0]])
    b, c, u = K.ols_line(x, y)
    assert b[0] == pytest.approx(1.9) and c[0] == pytest.approx(0.0, abs=1e-12) and K.engle_granger(x, y)[0][0] == pytest.approx(-2.8831, abs=5e-5)
    assert K.exact_spread_variance(0.2, 1.0, 0.0) == pytest.approx(5.5556, abs=5e-5) and float(K.te_from_f(np.array(5.5556), 53)) == pytest.approx(0.5 * math.log(100 / 90), abs=5e-6)


def test_claim_mutation_numbers_in_the_readme():
    state = json.loads((ROOT / "tests" / "mutation_state.json").read_text(encoding="utf-8"))
    killed = sum(1 for v in state.values() if v == "killed")
    equivalent = sum(1 for v in state.values() if v == "equivalent")
    assert (killed, equivalent) == (138, 8)
    assert "138 durch eigene Tests gefangen" in README and "8 als gleichwertig festgehalten" in README and "151 Stellen" in README
