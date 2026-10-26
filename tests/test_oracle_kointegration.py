"""Orakel (ORACLE-PLAYBOOK): unabhängige Wege zu denselben Zahlen.

Regression und Dickey–Fuller-Statistik gegen numpy.linalg.lstsq mit eigener Standardfehler-Formel; die simulierten kritischen Werte gegen die asymptotischen Werte der
Engle–Granger-Tabelle (aus dem Gedächtnis) und der Fehlalarm gegen die Nullrate; Granger-F gegen lstsq-Anpassungen und scipy.stats.f; die exakte Transferentropie gegen
scipy.linalg.solve_discrete_lyapunov und eine lange Simulation; die Plug-in-Transferentropie gegen eine Schleife über Entropien (scipy.stats.entropy); die naiven Tests gegen
scipy.stats.pearsonr. statsmodels fehlt im Test-venv und ist keine Abhängigkeit (siehe README, „Bewusst nicht umgesetzt“)."""

import math

import numpy as np
import pytest
from scipy import linalg, stats

import koi_algorithm as K
import koi_constants as C
from prs_scenario import VRng


def _lstsq_line(x, y):
    A = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return coef[1], coef[0], y - A @ coef


def test_regression_matches_lstsq_row_by_row():
    x, y = K.make_ecm_world(6, 80, 0.2, 1.0, 1.5, 0.1, VRng(3))
    b, c, u = K.ols_line(x, y)
    for i in range(6):
        bi, ci, ui = _lstsq_line(x[i], y[i])
        assert b[i] == pytest.approx(bi, abs=1e-9) and c[i] == pytest.approx(ci, abs=1e-8) and np.allclose(u[i], ui, atol=1e-9)


def _df_manual(u, const):
    du, lag = np.diff(u), u[:-1]
    A = np.column_stack([np.ones_like(lag), lag]) if const else lag[:, None]
    coef = np.linalg.lstsq(A, du, rcond=None)[0]
    e = du - A @ coef
    cov = (e @ e) / (len(du) - A.shape[1]) * np.linalg.inv(A.T @ A)
    return coef[-1] / math.sqrt(cov[-1, -1])


def test_dickey_fuller_matches_lstsq_with_and_without_constant():
    x, y = K.make_ecm_world(5, 120, 0.1, 0.5, 1.0, 0.0, VRng(4))
    t, u, _b = K.engle_granger(x, y)
    for i in range(5):
        assert t[i] == pytest.approx(_df_manual(u[i], False), abs=1e-9)
        assert K.df_tstat(x, const=True)[i] == pytest.approx(_df_manual(x[i], True), abs=1e-9)


def test_simulated_critical_values_agree_with_the_asymptotic_table_at_T_520():
    cv = K.critical_values(520)["eg"]
    for sim, table in zip(cv, (-3.9001, -3.3377, -3.0462)):                    # MacKinnon, aus dem Gedächtnis (Engle–Granger, zwei Reihen, mit Konstante)
        assert sim == pytest.approx(table, abs=0.08)
    assert K.critical_values(520)["df"][1] == pytest.approx(-2.8621, abs=0.06)    # Dickey–Fuller mit Konstante, asymptotisch −2,86


def test_the_null_rejection_rate_with_the_simulated_value_is_five_percent_on_fresh_data():
    for T in (52, 260):
        x, y = K.make_ecm_world(4000, T, 0.0, 1.0, 1.0, 0.0, VRng(123))
        rate = (K.engle_granger(x, y)[0] < K.critical_values(T)["eg"][1]).mean()
        assert rate == pytest.approx(0.05, abs=0.015), T                                    # der kritische Wert aus 5 000 Läufen hat selbst einen Fehler von etwa ±0,7 Prozentpunkten Fehlalarm


def test_naive_tests_match_scipy():
    x, y = K.make_ecm_world(5, 60, 0.0, 1.0, 1.0, 0.0, VRng(5))
    r, sig = K.naive_test(x, y)
    rd, sigd = K.difference_test(x, y)
    for i in range(5):
        assert r[i] == pytest.approx(stats.pearsonr(x[i], y[i])[0], abs=1e-12) and sig[i] == (stats.pearsonr(x[i], y[i])[1] < 0.05)
        dx, dy = np.diff(x[i]), np.diff(y[i])
        assert rd[i] == pytest.approx(stats.pearsonr(dx, dy)[0], abs=1e-12) and sigd[i] == (stats.pearsonr(dx, dy)[1] < 0.05)


def test_granger_f_matches_two_lstsq_fits_and_scipy_f():
    x, y = K.make_coupled(4, 90, 0.3, VRng(6))
    F, te, n = K.granger(x, y)
    for i in range(4):
        yt, y1, x1 = y[i, 1:], y[i, :-1], x[i, :-1]
        Ar, Af = np.column_stack([np.ones(n), y1]), np.column_stack([np.ones(n), y1, x1])
        rr = ((yt - Ar @ np.linalg.lstsq(Ar, yt, rcond=None)[0]) ** 2).sum()
        rf = ((yt - Af @ np.linalg.lstsq(Af, yt, rcond=None)[0]) ** 2).sum()
        assert F[i] == pytest.approx((rr - rf) / (rf / (n - 3)), rel=1e-9) and te[i] == pytest.approx(0.5 * math.log(rr / rf), abs=1e-12)
        assert K.f_pvalue(float(F[i]), 1, n - 3) == pytest.approx(stats.f.sf(F[i], 1, n - 3), abs=1e-9)
    assert K.f_critical(100) == pytest.approx(stats.f.ppf(0.95, 1, 100), abs=1e-6) and K.f_critical(50, 0.01) == pytest.approx(stats.f.ppf(0.99, 1, 50), abs=1e-6)


def test_granger_has_the_right_size_and_finds_the_right_direction():
    x, y = K.make_coupled(3000, 104, 0.2, VRng(7))
    crit = K.f_critical(102 - 0)
    F1, _t1, n = K.granger(x, y)
    F2, _t2, _ = K.granger(y, x)
    crit = K.f_critical(n - 3)
    assert (F1 > crit).mean() > 0.5 and (F2 > crit).mean() == pytest.approx(0.05, abs=0.02)


def test_exact_transfer_entropy_matches_lyapunov_and_a_long_simulation():
    for c in (0.1, 0.2, 0.4, 0.6):
        A = np.array([[C.PHI, 0.0], [c, C.PHI]])
        sigma = linalg.solve_discrete_lyapunov(A, np.eye(2))
        assert K.exact_te(c) == pytest.approx(0.5 * math.log(1 + c * c * (sigma[0, 0] - sigma[0, 1] ** 2 / sigma[1, 1])), abs=1e-12)
    x, y = K.make_coupled(40, 30000, 0.4, VRng(8))
    assert K.granger(x, y)[1].mean() == pytest.approx(K.exact_te(0.4), abs=0.002)


def test_the_mean_transfer_entropy_exceeds_the_exact_value_by_about_one_over_two_n():
    x, y = K.make_coupled(4000, 104, 0.2, VRng(9))
    F, te, n = K.granger(x, y)
    assert te.mean() == pytest.approx(K.exact_te(0.2) + K.gauss_te_bias(n), abs=0.0015)
    x0, y0 = K.make_coupled(4000, 104, 0.0, VRng(10))
    assert K.granger(x0, y0)[1].mean() == pytest.approx(K.gauss_te_bias(103), abs=0.0008)


def test_the_plugin_transfer_entropy_matches_a_loop_over_entropies():
    x, y = K.make_coupled(3, 400, 0.4, VRng(11))
    for i in range(3):
        for k in (2, 3, 5):
            n = len(y[i]) - 1
            xb, yb, yn = K._classes(x[i, :-1], k), K._classes(y[i, :-1], k), K._classes(y[i, 1:], k)

            def h(*cols):
                keys = np.zeros(n, dtype=int)
                for col in cols:
                    keys = keys * k + col
                return stats.entropy(np.bincount(keys) / n)
            assert K.te_plugin(x[i], y[i], k) == pytest.approx(h(yn, yb) + h(yb, xb) - h(yb) - h(yn, yb, xb), abs=1e-10)


def test_the_plugin_transfer_entropy_is_biased_upwards_by_about_k_times_k_minus_one_squared_over_2n():
    for n, k in ((260, 3), (1000, 3)):
        x, y = K.make_coupled(300, n + 1, 0.0, VRng(12))
        te = np.mean([K.te_plugin(x[i], y[i], k) for i in range(300)])
        assert te == pytest.approx(K.plugin_te_bias(n, k), rel=0.2), (n, k)


def test_the_permutation_test_keeps_its_level_without_coupling():
    x, y = K.make_coupled(300, 104, 0.0, VRng(13))
    rng = VRng(14)
    p = np.array([K.te_permutation(x[i], y[i], 3, 49, rng)[1] for i in range(300)])
    assert (p <= 0.05).mean() == pytest.approx(0.05, abs=0.04) and p.min() >= 1 / 50
    xc, yc = K.make_coupled(100, 104, 0.4, VRng(15))
    pc = np.array([K.te_permutation(xc[i], yc[i], 3, 49, rng)[1] for i in range(100)])
    assert (pc <= 0.05).mean() > 0.25                                                  # mit nur 3 Klassen und 104 Wochen hat der Plug-in-Test deutlich weniger Stärke als der Granger-F-Test


def test_the_adaptation_t_statistics_match_lstsq_with_n_minus_two_degrees_of_freedom():
    x, y = K.make_ecm_world(4, 60, 0.3, 0.5, 1.0, 0.0, VRng(16))
    _t, u, _b = K.engle_granger(x, y)
    ty, ay, tx, ax = K.ecm_alpha(x, y, u)
    for i in range(4):
        for v, t_val, a_val in ((y[i], ty[i], ay[i]), (x[i], tx[i], ax[i])):
            d, lag = np.diff(v), u[i, :-1]
            A = np.column_stack([np.ones_like(lag), lag])
            coef = np.linalg.lstsq(A, d, rcond=None)[0]
            e = d - A @ coef
            se = math.sqrt((e @ e) / (len(d) - 2) * np.linalg.inv(A.T @ A)[1, 1])
            assert a_val == pytest.approx(coef[1], abs=1e-10) and t_val == pytest.approx(coef[1] / se, abs=1e-9)
