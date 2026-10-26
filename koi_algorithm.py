"""Kern: drei Welten zweier Wochenreihen (Fehlerkorrektur mit gemeinsamer Irrfahrt, gekoppelte stationäre Reihen, verzögerte gemeinsame Ursache), Regression, Dickey–Fuller- und
Engle–Granger-Test mit kritischen Werten aus einer Nullsimulation, Fehlerkorrekturtests, Granger-F, gaußsche und Plug-in-Transferentropie mit Permutationstest und die exakten Formeln.

Alle Rechnungen sind zeilenweise vektorisiert: Arrays der Form (Wiederholungen, Wochen). Fehlerkorrektur-Welt: Δx = a (1 − s)/β · z_{t−1} + ε_x, Δy = −a s · z_{t−1} + γ ε_x + ε_y mit dem
Abstand z = y − β x; er folgt einem AR(1) mit Koeffizient 1 − a (a = 0: unabhängige Irrfahrten). Zufall und Verteilungsfunktionen kommen aus den wörtlichen Kopien von pearson-demo
(prs_*.py). Ohne scipy zur Laufzeit."""

import math
from functools import lru_cache

import numpy as np

import koi_constants as C
from prs_algorithm import betainc_reg, t_critical
from prs_scenario import VRng


# --- Welten ----------------------------------------------------------------------------------------------------------------------------------

def make_ecm_world(R: int, T: int, a: float, s: float, beta: float, gamma: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """R Wiederholungen der Fehlerkorrektur-Welt mit T Wochen: Rückgabe (x, y) der Form (R, T)."""
    ex, ey = rng.normal((R, T)), rng.normal((R, T))
    x, y = np.zeros((R, T)), np.zeros((R, T))
    x[:, 0] = ex[:, 0]
    y[:, 0] = beta * x[:, 0] + ey[:, 0]
    for t in range(1, T):
        z = y[:, t - 1] - beta * x[:, t - 1]
        x[:, t] = x[:, t - 1] + a * (1.0 - s) / beta * z + ex[:, t]
        y[:, t] = y[:, t - 1] - a * s * z + gamma * ex[:, t] + ey[:, t]
    return x, y


def make_coupled(R: int, T: int, c: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Zwei stationäre AR(1)-Reihen mit Koeffizient C.PHI; x_{t−1} wirkt mit Stärke c auf y_t. Einschwingzeit C.BURN Wochen."""
    n = T + C.BURN
    ex, ey = rng.normal((R, n)), rng.normal((R, n))
    x, y = np.zeros((R, n)), np.zeros((R, n))
    for t in range(1, n):
        x[:, t] = C.PHI * x[:, t - 1] + ex[:, t]
        y[:, t] = C.PHI * y[:, t - 1] + c * x[:, t - 1] + ey[:, t]
    return x[:, C.BURN:], y[:, C.BURN:]


def make_lagged_driver(R: int, T: int, b: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Eine verborgene AR(1)-Reihe z wirkt mit Stärke b und einer Woche Verzögerung auf x und mit zwei Wochen Verzögerung auf y; x wirkt nie auf y."""
    n = T + C.BURN
    ez, ex, ey = rng.normal((R, n)), rng.normal((R, n)), rng.normal((R, n))
    z = np.zeros((R, n))
    for t in range(1, n):
        z[:, t] = C.PHI * z[:, t - 1] + ez[:, t]
    x, y = ex.copy(), ey.copy()
    x[:, 1:] += b * z[:, :-1]
    y[:, 2:] += b * z[:, :-2]
    return x[:, C.BURN:], y[:, C.BURN:]


# --- Regression und Einheitswurzeltests ----------------------------------------------------------------------------------------------------------

def ols_line(x: np.ndarray, y: np.ndarray):
    """Zeilenweise Regression y = c + b x: Steigung, Achsenabschnitt, Rest."""
    xm, ym = x.mean(axis=1, keepdims=True), y.mean(axis=1, keepdims=True)
    b = ((x - xm) * (y - ym)).sum(axis=1, keepdims=True) / ((x - xm) ** 2).sum(axis=1, keepdims=True)
    c = ym - b * xm
    return b[:, 0], c[:, 0], y - c - b * x


def df_tstat(u: np.ndarray, const: bool = False) -> np.ndarray:
    """Dickey–Fuller-t-Statistik zeilenweise: Δu_t = [c +] γ u_{t−1} + e (ohne verzögerte Differenzen). Ohne Konstante für Reste aus einer Regression mit Konstante."""
    du, lag = u[:, 1:] - u[:, :-1], u[:, :-1]
    if const:
        lag, du = lag - lag.mean(axis=1, keepdims=True), du - du.mean(axis=1, keepdims=True)
    sxx = (lag**2).sum(axis=1, keepdims=True)
    g = (lag * du).sum(axis=1, keepdims=True) / sxx
    e = du - g * lag
    se2 = (e**2).sum(axis=1, keepdims=True) / (du.shape[1] - (2 if const else 1)) / sxx
    return (g / np.sqrt(se2))[:, 0]


def engle_granger(x: np.ndarray, y: np.ndarray):
    """Engle–Granger in zwei Schritten: Regression von y auf x mit Konstante, Dickey–Fuller an den Resten (ohne Konstante). Gibt (t, Rest, β̂) zurück."""
    b, _c, u = ols_line(x, y)
    return df_tstat(u, const=False), u, b


@lru_cache(maxsize=64)
def null_statistics(T: int, runs: int = C.NULL_RUNS, seed: int = C.NULL_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Nullverteilung für genau dieses T aus einer Simulation unabhängiger Irrfahrten (fester Seed): die Engle–Granger-Statistik und die Dickey–Fuller-Statistik (mit Konstante) der Reihe x."""
    x, y = make_ecm_world(runs, T, 0.0, 1.0, 1.0, 0.0, VRng(seed))
    return engle_granger(x, y)[0], df_tstat(x, const=True)


@lru_cache(maxsize=64)
def critical_values(T: int, runs: int = C.NULL_RUNS, seed: int = C.NULL_SEED) -> dict:
    """Kritische Werte (1 %, 5 %, 10 % Quantile) der Engle–Granger- und der Dickey–Fuller-Statistik für genau dieses T, aus der Nullsimulation `null_statistics`."""
    eg, df = null_statistics(T, runs, seed)
    q = (0.01, 0.05, 0.10)
    return {"eg": tuple(float(v) for v in np.quantile(eg, q)), "df": tuple(float(v) for v in np.quantile(df, q))}


def ecm_alpha(x: np.ndarray, y: np.ndarray, u: np.ndarray):
    """Fehlerkorrektur: Regression von Δy und von Δx auf den Rest û_{t−1} (mit Konstante). Gibt (t_y, α̂_y, t_x, α̂_x) zurück."""
    lag = u[:, :-1]
    out = []
    for v in (y, x):
        d = v[:, 1:] - v[:, :-1]
        lc, dc = lag - lag.mean(axis=1, keepdims=True), d - d.mean(axis=1, keepdims=True)
        sxx = (lc**2).sum(axis=1, keepdims=True)
        g = (lc * dc).sum(axis=1, keepdims=True) / sxx
        e = dc - g * lc
        se2 = (e**2).sum(axis=1, keepdims=True) / (d.shape[1] - 2) / sxx
        out += [(g / np.sqrt(se2))[:, 0], g[:, 0]]
    return out[0], out[1], out[2], out[3]


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac, bc = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    return (ac * bc).sum(axis=1) / np.sqrt((ac**2).sum(axis=1) * (bc**2).sum(axis=1))


@lru_cache(maxsize=256)
def _t_crit(df: int) -> float:
    return t_critical(df, C.ALPHA)


def naive_test(x: np.ndarray, y: np.ndarray):
    """t-Test auf r = 0 über die Werte (T − 2 Freiheitsgrade), als wären die Wochen unabhängig. Gibt (r, signifikant) zurück."""
    T = x.shape[1]
    r = _corr_rows(x, y)
    t = r * np.sqrt((T - 2) / (1.0 - r * r))
    return r, np.abs(t) > _t_crit(T - 2)


def difference_test(x: np.ndarray, y: np.ndarray):
    """t-Test auf r = 0 über die Differenzen (T − 1 Wochen, T − 3 Freiheitsgrade). Gibt (r, signifikant) zurück."""
    r = _corr_rows(np.diff(x, axis=1), np.diff(y, axis=1))
    n = x.shape[1] - 1
    t = r * np.sqrt((n - 2) / (1.0 - r * r))
    return r, np.abs(t) > _t_crit(n - 2)


# --- Granger und Transferentropie --------------------------------------------------------------------------------------------------------------

def granger(x: np.ndarray, y: np.ndarray):
    """Granger-F für x → y mit Verzögerung 1 (y_t auf y_{t−1} und x_{t−1}, mit Konstante) und die gaußsche Transferentropie TE = ½ ln(RSS_eingeschränkt / RSS_voll).
    Gibt (F, TE, n) zurück; n ist die Zahl der Beobachtungen der Regression (T − 1), F hat (1, n − 3) Freiheitsgrade."""
    yt, y1, x1 = y[:, 1:], y[:, :-1], x[:, :-1]
    yt, y1, x1 = (v - v.mean(axis=1, keepdims=True) for v in (yt, y1, x1))
    s11, sxx, s1x = (y1**2).sum(axis=1), (x1**2).sum(axis=1), (y1 * x1).sum(axis=1)
    sy1, syx, syy = (yt * y1).sum(axis=1), (yt * x1).sum(axis=1), (yt**2).sum(axis=1)
    rss_r = syy - sy1**2 / s11
    det = s11 * sxx - s1x**2
    b1, bx = (sy1 * sxx - syx * s1x) / det, (syx * s11 - sy1 * s1x) / det
    rss_f = syy - b1 * sy1 - bx * syx
    n = yt.shape[1]
    return (rss_r - rss_f) / (rss_f / (n - 3)), 0.5 * np.log(rss_r / rss_f), n


def f_pvalue(F: float, d1: int, d2: int) -> float:
    """Obere Fläche der F-Verteilung: p = I_{d2/(d2 + d1 F)}(d2/2, d1/2)."""
    if F <= 0.0:
        return 1.0
    return betainc_reg(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F))


@lru_cache(maxsize=256)
def f_critical(d2: int, alpha: float = C.ALPHA) -> float:
    """Kritischer Wert der F-Verteilung mit (1, d2) Freiheitsgraden durch Bisektion auf dem p-Wert."""
    lo, hi = 0.0, 1.0
    while f_pvalue(hi, 1, d2) > alpha:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f_pvalue(mid, 1, d2) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def te_from_f(F: np.ndarray, n: int) -> np.ndarray:
    """Die Identität: 2 · TE = ln(1 + F/(n − 3))."""
    return 0.5 * np.log1p(np.asarray(F) / (n - 3))


def _classes(v: np.ndarray, k: int) -> np.ndarray:
    """Klassen nach Rängen: k gleich besetzte Klassen (bei Gleichstand nach Reihenfolge)."""
    ranks = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return np.minimum(ranks * k // len(v), k - 1)


def te_plugin(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Plug-in-Transferentropie x → y in nats für eine Reihe: I(Y_{t+1}; X_t | Y_t) aus den Häufigkeiten von k Klassen je Größe (Klassen nach Rängen)."""
    n = len(y) - 1
    xb, yb, yn = _classes(np.asarray(x)[:-1], k), _classes(np.asarray(y)[:-1], k), _classes(np.asarray(y)[1:], k)
    cnt = np.bincount(yn * k * k + yb * k + xb, minlength=k**3).reshape(k, k, k).astype(float)
    py, pyx, pyny = cnt.sum(axis=(0, 2)), cnt.sum(axis=0), cnt.sum(axis=2)
    a, b, c = np.nonzero(cnt)
    return float(np.sum(cnt[a, b, c] / n * np.log(cnt[a, b, c] * py[b] / (pyx[b, c] * pyny[a, b]))))


def te_permutation(x: np.ndarray, y: np.ndarray, k: int, B: int, rng) -> tuple[float, float, float]:
    """Plug-in-Transferentropie mit Permutationstest: x wird B-mal in zufälliger Reihenfolge gemischt (zerstört die Beziehung zu y, erhält jedes Rand). Gibt (TE, p, TE minus Mittel der
    Mischungen) zurück; p = (1 + Zahl der Mischungen mit TE ≥ beobachtet) / (B + 1)."""
    x = np.asarray(x, dtype=float)
    te = te_plugin(x, y, k)
    perm = np.argsort(rng.uniform((B, len(x))), axis=1)
    null = np.array([te_plugin(x[p], y, k) for p in perm])
    return te, float((1 + np.sum(null >= te)) / (B + 1)), te - float(null.mean())


# --- Exakte Formeln --------------------------------------------------------------------------------------------------------------------------

def exact_spread_variance(a: float, beta: float, gamma: float) -> float:
    """Varianz des stationären Abstands: ((γ − β)² + 1) / (1 − (1 − a)²); unendlich bei a = 0 (keine Beziehung: der Abstand wandert)."""
    if a <= 0.0:
        return math.inf
    return ((gamma - beta) ** 2 + 1.0) / (1.0 - (1.0 - a) ** 2)


def exact_alphas(a: float, s: float, beta: float) -> tuple[float, float]:
    """Wahre Anpassungskoeffizienten (α_y, α_x) = (−a s, a (1 − s)/β)."""
    return -a * s, a * (1.0 - s) / beta


def exact_te(c: float, phi: float = C.PHI) -> float:
    """Exakte gaußsche Transferentropie x → y der gekoppelten AR(1)-Reihen: ½ ln(1 + c² (Σ_xx − Σ_xy²/Σ_yy)), Σ aus der Lyapunov-Gleichung Σ = A Σ Aᵀ + I mit A = [[φ, 0], [c, φ]]."""
    A = np.array([[phi, 0.0], [c, phi]])
    sigma = np.linalg.solve(np.eye(4) - np.kron(A, A), np.eye(2).reshape(4)).reshape(2, 2)
    return 0.5 * math.log(1.0 + c * c * (sigma[0, 0] - sigma[0, 1] ** 2 / sigma[1, 1]))


def gauss_te_bias(n: int) -> float:
    """Erwarteter Überschuss der gaußschen Transferentropie ohne Kopplung: 1/(2n) (ein Parameter)."""
    return 1.0 / (2.0 * n)


def plugin_te_bias(n: int, k: int) -> float:
    """Näherung des Überschusses der Plug-in-Transferentropie ohne Kopplung: k (k − 1)² / (2n) (Zahl der Parameter des Unabhängigkeitstests durch 2n)."""
    return k * (k - 1) ** 2 / (2.0 * n)
