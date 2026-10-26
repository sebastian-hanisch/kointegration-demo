"""Auswertung: eine Stichprobe in einer der drei Welten (Settings -> Analysis mit den Zeilen der Hauptansicht), Zellen des vorgerechneten Rasters (Messreihen), Verdikte als reine
Funktionen und die Probe für den Rastertest."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import koi_algorithm as K
import koi_constants as C
from prs_scenario import VRng

GRID_FILE = Path(__file__).resolve().parent / "data" / "cointegration_grid.json"


def de(v, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma; „–“ für fehlende oder nicht definierte Werte."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    return f"{v:.{digits}f}".replace(".", ",")


def pct(v: float, digits: int = 1) -> str:
    return de(100.0 * v, digits) + " %"


@dataclass(frozen=True, kw_only=True)
class Settings:
    world: str = C.DEFAULT_WORLD
    t: int = C.DEFAULT_T
    a: float = C.DEFAULT_A
    share: float = C.DEFAULT_SHARE
    beta: float = C.DEFAULT_BETA
    gamma: float = C.DEFAULT_GAMMA
    coupling: float = C.DEFAULT_COUPLING
    driver: float = C.DEFAULT_DRIVER
    classes: int = C.DEFAULT_CLASSES
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Row:
    key: str
    name: str
    value: float | None
    threshold: float | None
    decision: str
    truth: str
    digits: int = 3


@dataclass
class Analysis:
    settings: Settings
    x: np.ndarray
    y: np.ndarray
    rows: list
    spread: np.ndarray | None            # Rest der Engle–Granger-Regression (nur Fehlerkorrektur-Welt)
    cv: dict | None                      # kritische Werte der Nullsimulation (nur Fehlerkorrektur-Welt)
    extra: dict                          # weitere Größen für Verdikte

    def row(self, key: str):
        return next((r for r in self.rows if r.key == key), None)


def make_series(s: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Eine Stichprobe (Form (1, T)) der gewählten Welt."""
    rng = VRng(s.seed)
    if s.world == "ecm":
        return K.make_ecm_world(1, s.t, s.a, s.share, s.beta, s.gamma, rng)
    if s.world == "coupled":
        return K.make_coupled(1, s.t, s.coupling, rng)
    if s.world == "lagged":
        return K.make_lagged_driver(1, s.t, s.driver, rng)
    raise ValueError(f"unbekannte Welt: {s.world}")


def _analyse_ecm(s: Settings, x: np.ndarray, y: np.ndarray) -> Analysis:
    cv = K.critical_values(s.t)
    eg_cut, df_cut = cv["eg"][1], cv["df"][1]
    t_eg, u, b = K.engle_granger(x, y)
    ty, ay, tx, ax = K.ecm_alpha(x, y, u)
    r, sig = K.naive_test(x, y)
    rd, sigd = K.difference_test(x, y)
    df_x, df_y = float(K.df_tstat(x, const=True)[0]), float(K.df_tstat(y, const=True)[0])
    related = s.a > 0.0
    true_y, true_x = K.exact_alphas(s.a, s.share, s.beta)
    var = K.exact_spread_variance(s.a, s.beta, s.gamma)
    rows = [
        Row("naive", "Naiver Test auf r (Werte)", float(r[0]), None, "signifikant" if sig[0] else "nicht signifikant", "Beziehung vorhanden" if related else "keine Beziehung"),
        Row("diff", "Test auf r der Differenzen", float(rd[0]), None, "signifikant" if sigd[0] else "nicht signifikant", f"kurzfristige Kopplung γ = {de(s.gamma, 1)}"),
        Row("df_x", "Dickey–Fuller x (Einheitswurzel?)", df_x, df_cut, "stationär" if df_x < df_cut else "wandert (Einheitswurzel nicht abgelehnt)", "wandert (Irrfahrt)"),
        Row("df_y", "Dickey–Fuller y (Einheitswurzel?)", df_y, df_cut, "stationär" if df_y < df_cut else "wandert (Einheitswurzel nicht abgelehnt)", "wandert (Irrfahrt)"),
        Row("eg", "Engle–Granger (Abstand stationär?)", float(t_eg[0]), eg_cut, "kointegriert" if t_eg[0] < eg_cut else "keine Beziehung nachweisbar", "kointegriert" if related else "nicht kointegriert"),
        Row("beta", "β̂ der Regression y auf x", float(b[0]), None, "–", f"β = {de(s.beta, 1)}"),
        Row("alpha_y", "Anpassung von y (α̂_y)", float(ay[0]), None, "y passt sich an" if abs(ty[0]) > C.Z_ADAPT else "y passt sich nicht an", f"α_y = {de(true_y + 0.0)}"),
        Row("alpha_x", "Anpassung von x (α̂_x)", float(ax[0]), None, "x passt sich an" if abs(tx[0]) > C.Z_ADAPT else "x passt sich nicht an", f"α_x = {de(true_x + 0.0)}"),
        Row("spread", "Standardabweichung des Abstands (Rest)", float(u[0].std()), None, "–", "wandert (a = 0)" if not related else f"{de(math.sqrt(var))} (exakt, stationär)", 2),
    ]
    return Analysis(s, x[0], y[0], rows, u[0], cv, {"t_eg": float(t_eg[0]), "t_alpha_y": float(ty[0]), "t_alpha_x": float(tx[0]), "r": float(r[0]), "r_diff": float(rd[0]), "sig": bool(sig[0]),
                                                   "sig_diff": bool(sigd[0]), "eg_cut": eg_cut})


def _analyse_directed(s: Settings, x: np.ndarray, y: np.ndarray) -> Analysis:
    rng = VRng(s.seed + 1000003)
    F1, te1, n = K.granger(x, y)
    F2, te2, _ = K.granger(y, x)
    crit = K.f_critical(n - 3)
    p1, p2 = K.f_pvalue(float(F1[0]), 1, n - 3), K.f_pvalue(float(F2[0]), 1, n - 3)
    k = s.classes
    plug1 = K.te_permutation(x[0], y[0], k, C.PERMUTATIONS, rng)
    plug2 = K.te_permutation(y[0], x[0], k, C.PERMUTATIONS, rng)
    coupled = s.world == "coupled"
    truth_dir = f"Kopplung c = {de(s.coupling, 1)}" if coupled else "keine Wirkung von x auf y"
    rows = [
        Row("granger_xy", "Granger-F x → y", float(F1[0]), crit, "x hilft, y vorherzusagen (p = " + de(p1, 3) + ")" if F1[0] > crit else "kein Gewinn (p = " + de(p1, 3) + ")", truth_dir, 2),
        Row("granger_yx", "Granger-F y → x", float(F2[0]), crit, "y hilft, x vorherzusagen (p = " + de(p2, 3) + ")" if F2[0] > crit else "kein Gewinn (p = " + de(p2, 3) + ")",
            "keine Wirkung von y auf x", 2),
        Row("te_xy", "Gaußsche Transferentropie x → y (nats)", float(te1[0]), None, "–", f"exakt {de(K.exact_te(s.coupling), 4)} (+ {de(K.gauss_te_bias(n), 4)} Überschuss im Mittel)" if coupled else "–", 4),
        Row("te_yx", "Gaußsche Transferentropie y → x (nats)", float(te2[0]), None, "–", f"exakt 0 (+ {de(K.gauss_te_bias(n), 4)} Überschuss im Mittel)" if coupled else "–", 4),
        Row("plugin_xy", f"Plug-in-Transferentropie x → y ({k} Klassen, nats)", plug1[0], None, f"p = {de(plug1[1], 3)} (Permutation)" + (": Richtung gefunden" if plug1[1] <= C.ALPHA else ": nicht nachweisbar"),
            f"Überschuss ohne Kopplung etwa {de(K.plugin_te_bias(n, k), 3)}", 4),
        Row("plugin_yx", f"Plug-in-Transferentropie y → x ({k} Klassen, nats)", plug2[0], None, f"p = {de(plug2[1], 3)} (Permutation)" + (": Richtung gefunden" if plug2[1] <= C.ALPHA else ": nicht nachweisbar"),
            f"Überschuss ohne Kopplung etwa {de(K.plugin_te_bias(n, k), 3)}", 4),
    ]
    return Analysis(s, x[0], y[0], rows, None, None, {"F_xy": float(F1[0]), "F_yx": float(F2[0]), "crit": crit, "n": n, "te_xy": float(te1[0]), "te_yx": float(te2[0]), "p_xy": p1, "p_yx": p2,
                                                       "plugin_xy": plug1[0], "plugin_xy_p": plug1[1], "plugin_xy_corrected": plug1[2], "plugin_yx_p": plug2[1]})


def analyse(s: Settings) -> Analysis:
    x, y = make_series(s)
    return _analyse_ecm(s, x, y) if s.world == "ecm" else _analyse_directed(s, x, y)


# --- Zellen des Rasters (Anteile und Mittel über viele Wiederholungen) -----------------------------------------------------------------------------

def _batches(make, seeds: tuple, batches: int, size: int):
    for seed in seeds:
        rng = VRng(seed)
        for _ in range(batches):
            yield make(size, rng)


def _ecm_maker(T: int, a: float, s: float = 1.0, beta: float = 1.0, gamma: float = 0.0):
    return lambda size, rng: K.make_ecm_world(size, T, a, s, beta, gamma, rng)


def power_cell(T: int, a: float, seeds: tuple = C.GRID_SEEDS, batches: int = C.BATCHES, size: int = C.BATCH) -> dict:
    """E1: Anteil „Beziehung gefunden“ des Engle–Granger-Tests (simulierter Wert des Fünf-Prozent-Niveaus), „signifikant“ des naiven Tests und „stationär“ der Dickey–Fuller-Tests je Reihe."""
    cut = K.critical_values(T)["eg"][1]
    df_cut = K.critical_values(T)["df"][1]
    acc = []
    for x, y in _batches(_ecm_maker(T, a), seeds, batches, size):
        t = K.engle_granger(x, y)[0]
        acc.append(np.column_stack([t < cut, K.naive_test(x, y)[1], K.df_tstat(x, const=True) < df_cut]))
    m = np.vstack(acc).mean(axis=0)
    return {"T": T, "a": a, "runs": int(sum(len(v) for v in acc)), "eg": float(m[0]), "naive": float(m[1]), "df_x": float(m[2]), "cut": cut}


def null_cell(T: int, seeds: tuple = C.GRID_SEEDS, batches: int = C.BATCHES, size: int = C.BATCH) -> dict:
    """E2: Quantile (1, 5, 10 %) der Engle–Granger-Statistik und das 5-%-Quantil der Dickey–Fuller-Statistik unter unabhängigen Irrfahrten; Fehlalarm mit dem tabellierten Wert −3,34."""
    eg, df = [], []
    for x, y in _batches(_ecm_maker(T, 0.0), seeds, batches, size):
        eg.append(K.engle_granger(x, y)[0])
        df.append(K.df_tstat(x, const=True))
    eg, df = np.concatenate(eg), np.concatenate(df)
    return {"T": T, "runs": len(eg), "q": [float(v) for v in np.quantile(eg, (0.01, 0.05, 0.10))], "df_q05": float(np.quantile(df, 0.05)), "alarm_table": float((eg < -3.3377).mean()),
            "alarm_live": float((eg < K.critical_values(T)["eg"][1]).mean()), "cut_live": K.critical_values(T)["eg"][1]}


def diff_cell(gamma: float, a: float, T: int = C.E3_T, seeds: tuple = C.GRID_SEEDS, batches: int = C.BATCHES, size: int = C.BATCH) -> dict:
    """E3: Anteil „signifikant“ des Tests auf r der Differenzen und „Beziehung gefunden“ des Engle–Granger-Tests; mittleres r der Differenzen."""
    cut = K.critical_values(T)["eg"][1]
    acc = []
    for x, y in _batches(_ecm_maker(T, a, gamma=gamma), seeds, batches, size):
        rd, sd = K.difference_test(x, y)
        acc.append(np.column_stack([sd, K.engle_granger(x, y)[0] < cut, rd]))
    m = np.vstack(acc).mean(axis=0)
    return {"gamma": gamma, "a": a, "T": T, "runs": int(sum(len(v) for v in acc)), "diff": float(m[0]), "eg": float(m[1]), "r_diff": float(m[2])}


def adapt_cell(T: int, a: float, share: float, seeds: tuple = C.GRID_SEEDS, batches: int = C.BATCHES, size: int = C.BATCH) -> dict:
    """E4: Anteil |t| > 1,96 der Anpassung von y und von x und die mittleren Schätzer α̂_y, α̂_x."""
    acc = []
    for x, y in _batches(_ecm_maker(T, a, share), seeds, batches, size):
        _t, u, _b = K.engle_granger(x, y)
        ty, ay, tx, ax = K.ecm_alpha(x, y, u)
        acc.append(np.column_stack([np.abs(ty) > C.Z_ADAPT, np.abs(tx) > C.Z_ADAPT, ay, ax]))
    m = np.vstack(acc).mean(axis=0)
    true_y, true_x = K.exact_alphas(a, share, 1.0)
    return {"T": T, "a": a, "share": share, "runs": int(sum(len(v) for v in acc)), "sig_y": float(m[0]), "sig_x": float(m[1]), "alpha_y": float(m[2]), "alpha_x": float(m[3]), "true_y": true_y, "true_x": true_x}


def granger_cell(c: float, T: int = C.E5_T, seeds: tuple = C.GRID_SEEDS, batches: int = C.BATCHES, size: int = C.BATCH) -> dict:
    """E5: Anteil „erkannt“ des Granger-F-Tests in beide Richtungen, mittlere gaußsche Transferentropie, exakter Wert und größte Abweichung der Identität 2 · TE = ln(1 + F/(n − 3))."""
    acc, ident = [], 0.0
    n = T - 1
    crit = K.f_critical(n - 3)
    for x, y in _batches(lambda size_, rng: K.make_coupled(size_, T, c, rng), seeds, batches, size):
        F1, te1, _ = K.granger(x, y)
        F2, te2, _ = K.granger(y, x)
        acc.append(np.column_stack([F1 > crit, F2 > crit, te1, te2]))
        ident = max(ident, float(np.abs(2 * te1 - np.log1p(F1 / (n - 3))).max()))
    m = np.vstack(acc).mean(axis=0)
    return {"c": c, "T": T, "runs": int(sum(len(v) for v in acc)), "xy": float(m[0]), "yx": float(m[1]), "te_xy": float(m[2]), "te_yx": float(m[3]), "exact": K.exact_te(c), "bias": K.gauss_te_bias(n), "identity": ident}


def bias_cell(T: int, k: int, seeds: tuple = C.GRID_SEEDS, size: int = C.E6_BATCH) -> dict:
    """E6: Verzerrung der Transferentropie ohne Kopplung: Mittel der gaußschen und der Plug-in-Schätzung (k Klassen) gegen 1/(2n) und k (k − 1)²/(2n)."""
    n = T - 1
    gauss, plug = [], []
    for x, y in _batches(lambda size_, rng: K.make_coupled(size_, T, 0.0, rng), seeds, 1, size):
        gauss.append(K.granger(x, y)[1])
        plug.append([K.te_plugin(x[i], y[i], k) for i in range(len(x))])
    g, p = np.concatenate(gauss), np.concatenate(plug)
    return {"T": T, "k": k, "runs": len(g), "gauss": float(g.mean()), "plugin": float(p.mean()), "gauss_pred": K.gauss_te_bias(n), "plugin_pred": K.plugin_te_bias(n, k)}


def lagged_cell(b: float, T: int = C.E7_T, seeds: tuple = C.GRID_SEEDS, batches: int = C.BATCHES, size: int = C.BATCH) -> dict:
    """E7: verzögerte gemeinsame Ursache der Stärke b: Anteil „x → y erkannt“ und „y → x erkannt“ des Granger-F-Tests, obwohl x nie auf y wirkt."""
    n = T - 1
    crit = K.f_critical(n - 3)
    acc = []
    for x, y in _batches(lambda size_, rng: K.make_lagged_driver(size_, T, b, rng), seeds, batches, size):
        acc.append(np.column_stack([K.granger(x, y)[0] > crit, K.granger(y, x)[0] > crit]))
    m = np.vstack(acc).mean(axis=0)
    return {"b": b, "T": T, "runs": int(sum(len(v) for v in acc)), "xy": float(m[0]), "yx": float(m[1])}


def grid_probe() -> dict:
    """Kleine feste Probe, die tests/test_grid.py nachrechnet und mit der Datei vergleicht."""
    kw = {"seeds": (101,), "batches": 1, "size": 50}
    return {"power": power_cell(104, 0.2, **kw), "null": null_cell(52, **kw), "diff": diff_cell(0.5, 0.2, **kw), "adapt": adapt_cell(104, 0.2, 0.5, **kw), "granger": granger_cell(0.2, **kw),
            "bias": bias_cell(100, 3, seeds=(101,), size=40), "lagged": lagged_cell(0.8, **kw)}


def load_grid(path: Path = GRID_FILE):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(cells: list, **kw):
    return next((c for c in cells if all(c.get(k) == v for k, v in kw.items())), None)


# --- Verdikte (reine Funktionen: Art, Text) ----------------------------------------------------------------------------------------------------

def verdict_main(a: Analysis):
    s, ex = a.settings, a.extra
    if s.world == "ecm":
        eg = a.row("eg")
        found, related = eg.value < eg.threshold, s.a > 0.0
        naive = "der naive Test auf r nennt die Reihen " + ("**signifikant** verbunden" if ex["sig"] else "nicht signifikant verbunden")
        stat = f"Engle–Granger t = {de(eg.value)} bei der Schwelle {de(eg.threshold)}"
        if related and found:
            return "success", f"Der Engle–Granger-Test findet die Beziehung ({stat}); {naive}" + (", und das hätte er auch ohne jede Beziehung getan." if ex["sig"] else ".")
        if related and not found:
            return "warning", (f"Es gibt eine Beziehung (Anpassungstempo {de(s.a)}), aber der Test findet sie nicht ({stat}): zu wenige Wochen oder zu langsame Anpassung (der Abstand hat die "
                               f"Autokorrelation {de(1 - s.a)}). {naive[0].upper() + naive[1:]}.")
        if found:
            return "warning", f"Fehlalarm: es gibt keine Beziehung, aber der Test meldet eine ({stat}); bei einem Niveau von 5 % passiert das in etwa jedem zwanzigsten Lauf."
        return "info", f"Der Engle–Granger-Test findet keine Beziehung ({stat}), richtig; {naive}."
    xy, yx = ex["F_xy"] > ex["crit"], ex["F_yx"] > ex["crit"]
    base = f"Granger-F x → y {de(ex['F_xy'])} (Schwelle {de(ex['crit'])}), y → x {de(ex['F_yx'])}"
    if s.world == "lagged":
        if xy:
            return "warning", (f"{base}: der Test „findet“ die Richtung x → y, obwohl x nie auf y wirkt. x_{{t−1}} trägt die verborgene Ursache von vor zwei Wochen in sich (x folgt ihr mit einer, y mit zwei Wochen Verzögerung) und hilft deshalb, y vorherzusagen: "
                               "Vorhersagbarkeit, keine Wirkung.")
        return "info", f"{base}: in dieser Stichprobe findet der Test keine Richtung, obwohl die verzögerte Ursache sie in den meisten Läufen erzeugt (Experiment 7)."
    if s.coupling == 0.0:
        return "info", f"{base}: ohne Kopplung ist jede „gefundene“ Richtung ein Fehlalarm ({'hier einer' if xy or yx else 'hier keiner'})."
    if xy and not yx:
        return "success", f"{base}: der Test findet die Richtung x → y (Kopplung c = {de(s.coupling, 1)}) und nicht die umgekehrte."
    if xy and yx:
        return "warning", f"{base}: beide Richtungen sind „signifikant“, in Wahrheit wirkt nur x auf y."
    return "warning", f"{base}: die Kopplung c = {de(s.coupling, 1)} ist da, aber der Test findet sie nicht (zu wenige Wochen)."


def verdict_adapt(a: Analysis):
    """Nur Fehlerkorrektur-Welt: wer passt sich an."""
    s, ex = a.settings, a.extra
    ay, ax = abs(ex["t_alpha_y"]) > C.Z_ADAPT, abs(ex["t_alpha_x"]) > C.Z_ADAPT
    who = "y passt sich an" if ay and not ax else "x passt sich an" if ax and not ay else "beide passen sich an" if ay and ax else "keiner passt sich nachweisbar an"
    return "info", (f"Fehlerkorrektur: {who} (t für α_y {de(ex['t_alpha_y'])}, für α_x {de(ex['t_alpha_x'])}); wahr: {C.SHARES[s.share]}, Tempo {de(s.a)}. "
                    "Der Partner, der sich nicht anpasst, führt; bei kurzen Reihen ist diese Richtung unscharf.")


def verdict_difference(a: Analysis):
    s, ex = a.settings, a.extra
    state = "signifikant" if ex["sig_diff"] else "nicht signifikant"
    note = ("Differenzen zeigen die Bewegung derselben Woche, nicht das Zurückziehen des Abstands: bei einer Beziehung ohne kurzfristige Kopplung sehen sie fast nichts."
            if s.gamma == 0.0 else "Die kurzfristige Kopplung macht die Differenzen „signifikant“, ganz unabhängig davon, ob es eine langfristige Beziehung gibt.")
    return "info", f"Test auf r der Differenzen: r = {de(ex['r_diff'])}, {state}. {note}"


def verdict_te(a: Analysis):
    """Nur gerichtete Welten: Transferentropie, Identität mit Granger-F und Verzerrung."""
    s, ex = a.settings, a.extra
    n = ex["n"]
    ident = 0.5 * math.log1p(ex["F_xy"] / (n - 3))
    return "info", (f"Gaußsche Transferentropie x → y {de(ex['te_xy'], 4)} nats = ½ ln(1 + F/(n − 3)) = {de(ident, 4)} (dieselbe Zahl wie der F-Test, nur in anderer Einheit); im Mittel liegt sie ohne Kopplung um "
                    f"1/(2n) = {de(K.gauss_te_bias(n), 4)} über null. Der Plug-in-Schätzer mit {s.classes} Klassen liefert {de(ex['plugin_xy'], 4)} (Permutationstest p = {de(ex['plugin_xy_p'], 3)}, "
                    f"nach Abzug des Mittels der Mischungen {de(ex['plugin_xy_corrected'], 4)}); ohne Kopplung wäre sein Überschuss etwa {de(K.plugin_te_bias(n, s.classes), 3)}.")


# --- Verdikte der Experimente (aus dem Raster) --------------------------------------------------------------------------------------------------

def verdict_e1(cells: list, T: int):
    zero, fast = find(cells, T=T, a=0.0), find(cells, T=T, a=0.2)
    return "info", (f"T = {T}: ohne Beziehung (a = 0) sagt der naive Test in {pct(zero['naive'])} „signifikant“, der Engle–Granger-Test in {pct(zero['eg'])}; bei echter Beziehung (a = 0,2) sind es "
                    f"{pct(fast['naive'])} und {pct(fast['eg'])}. Der naive Test trennt beides nicht, der Engle–Granger-Test schon.")


def verdict_e2(cells: list, T: int):
    c = find(cells, T=T)
    return "info", (f"T = {T}: die simulierten Quantile der Engle–Granger-Statistik sind {de(c['q'][0])} / {de(c['q'][1])} / {de(c['q'][2])} (1 / 5 / 10 %); mit dem tabellierten Wert −3,34 gäbe es "
                    f"{pct(c['alarm_table'])} Fehlalarm, mit dem simulierten Wert dieser Demo ({de(c['cut_live'])}) {pct(c['alarm_live'])}.")


def verdict_e3(cells: list, gamma: float, a: float):
    c = find(cells, gamma=gamma, a=a)
    return "warning", (f"Kurzfristige Kopplung {de(gamma, 1)}, Anpassungstempo {de(a, 1)}: der Test auf die Differenzen ist in {pct(c['diff'])} der Läufe „signifikant“, der Engle–Granger-Test findet die "
                       f"Beziehung in {pct(c['eg'])} (mittleres r der Differenzen {de(c['r_diff'], 3)}).")


def verdict_e4(cell: dict):
    return "info", (f"T = {cell['T']}, Tempo {de(cell['a'])}, {C.SHARES[cell['share']]}: α_y ist in {pct(cell['sig_y'])} der Läufe signifikant, α_x in {pct(cell['sig_x'])}; "
                    f"mittlere Schätzer α̂_y {de(cell['alpha_y'], 3)} (wahr {de(cell['true_y'], 3)}) und α̂_x {de(cell['alpha_x'], 3)} (wahr {de(cell['true_x'], 3)}).")


def verdict_e5(cell: dict):
    return "info", (f"Kopplung c = {de(cell['c'], 1)}: x → y erkannt in {pct(cell['xy'])}, y → x in {pct(cell['yx'])} der {cell['runs']} Läufe; mittlere Transferentropie x → y {de(cell['te_xy'], 4)} "
                    f"(exakt {de(cell['exact'], 4)} plus 1/(2n) = {de(cell['bias'], 4)}); die Identität 2 · TE = ln(1 + F/(n − 3)) stimmt auf {cell['identity']:.0e}.")


def verdict_e6(cells: list, k: int):
    mine = sorted((c for c in cells if c["k"] == k), key=lambda c: c["T"])
    lo, hi = mine[0], mine[-1]
    return "warning", (f"Ohne jede Kopplung: gaußsche Schätzung {de(lo['gauss'], 4)} (T = {lo['T']}) bis {de(hi['gauss'], 4)} (T = {hi['T']}); Plug-in mit {k} Klassen {de(lo['plugin'], 4)} bis {de(hi['plugin'], 4)} "
                       f"(Näherung k (k − 1)²/(2n): {de(lo['plugin_pred'], 4)} bis {de(hi['plugin_pred'], 4)}).")


def verdict_e7(cell: dict):
    return "warning", (f"Verzögerte gemeinsame Ursache der Stärke {de(cell['b'], 1)}: der Granger-Test „findet“ x → y in {pct(cell['xy'])} der Läufe und y → x in {pct(cell['yx'])}, obwohl x nie auf y wirkt.")
