"""Einheiten des Kerns an Mini-Instanzen von Hand (DEMO-PLAYBOOK §1b): Regression, Dickey–Fuller, Fehlerkorrektur, Granger, Transferentropie, exakte Formeln, Welten."""

import math

import numpy as np
import pytest

import koi_algorithm as K
import koi_constants as C
from prs_scenario import VRng


# --- Regression und Dickey–Fuller ------------------------------------------------------------------------------------------------------------

def test_regression_by_hand(four_weeks):
    x, y = four_weeks
    b, c, u = K.ols_line(x, y)
    assert b[0] == pytest.approx(1.9, abs=1e-12) and c[0] == pytest.approx(0.0, abs=1e-12) and np.allclose(u[0], [0.1, 0.2, -0.7, 0.4], atol=1e-12)


def test_dickey_fuller_at_the_residual_by_hand(four_weeks):
    x, y = four_weeks
    t, u, b = K.engle_granger(x, y)
    assert b[0] == pytest.approx(1.9) and t[0] == pytest.approx(-2.88311006, abs=1e-7)
    lag, du = u[0, :-1], np.diff(u[0])
    g = (lag * du).sum() / (lag**2).sum()
    assert g == pytest.approx(-0.94 / 0.54, abs=1e-12)


def test_dickey_fuller_with_constant_removes_the_means():
    u = np.array([[1.0, 2.0, 4.0, 3.0, 5.0]])
    with_const = K.df_tstat(u, const=True)[0]
    shifted = K.df_tstat(u + 100.0, const=True)[0]
    assert with_const == pytest.approx(shifted, abs=1e-9) and K.df_tstat(u + 100.0, const=False)[0] != pytest.approx(with_const, abs=1e-3)


def test_a_stationary_series_has_a_very_negative_statistic_and_a_random_walk_does_not():
    rng = VRng(3)
    e = rng.normal((200, 200))
    walk = np.cumsum(e, axis=1)
    ar = np.zeros_like(e)
    for t in range(1, 200):
        ar[:, t] = 0.3 * ar[:, t - 1] + e[:, t]
    assert np.median(K.df_tstat(ar, const=True)) < -7.0 and -3.0 < np.median(K.df_tstat(walk, const=True)) < -0.5


# --- Welten ----------------------------------------------------------------------------------------------------------------------------------

def test_the_ecm_world_has_independent_walks_at_a_zero_and_a_stationary_spread_otherwise():
    x, y = K.make_ecm_world(2000, 300, 0.0, 1.0, 1.0, 0.0, VRng(1))
    assert x.shape == (2000, 300) and np.median(K.engle_granger(x, y)[0]) > -3.2
    x, y = K.make_ecm_world(2000, 300, 0.5, 1.0, 1.0, 0.0, VRng(1))
    assert np.median(K.engle_granger(x, y)[0]) < -5.0


def test_the_spread_variance_matches_the_formula_and_a_simulation():
    assert K.exact_spread_variance(0.2, 1.0, 0.0) == pytest.approx(2 / 0.36, abs=1e-12) and K.exact_spread_variance(0.2, 1.0, 0.0) == pytest.approx(5.5556, abs=1e-4)
    assert K.exact_spread_variance(0.5, 2.0, 0.3) == pytest.approx((1.7**2 + 1) / 0.75, abs=1e-12) and math.isinf(K.exact_spread_variance(0.0, 1.0, 0.0))
    for a, beta, gamma, s in ((0.2, 1.0, 0.0, 1.0), (0.4, 2.0, 0.3, 0.5), (0.3, 0.5, 0.0, 0.0)):
        x, y = K.make_ecm_world(3000, 400, a, s, beta, gamma, VRng(2))
        z = (y - beta * x)[:, 100:]
        assert z.var() == pytest.approx(K.exact_spread_variance(a, beta, gamma), rel=0.05), (a, beta, gamma, s)


def test_the_coupled_world_has_the_documented_cross_correlation():
    x, y = K.make_coupled(2000, 200, 0.4, VRng(4))
    lag_corr = np.mean([np.corrcoef(x[i, :-1], y[i, 1:])[0, 1] for i in range(300)])
    back = np.mean([np.corrcoef(y[i, :-1], x[i, 1:])[0, 1] for i in range(300)])
    assert lag_corr > 0.4 and back < lag_corr - 0.2 and x.shape == (2000, 200)           # die Rückrichtung ist nur über die Autokorrelation von x nicht null


def test_the_lagged_driver_world_has_x_leading_y_by_one_week_without_a_link():
    x, y = K.make_lagged_driver(2000, 200, 1.0, VRng(4))
    a = np.mean([np.corrcoef(x[i, :-1], y[i, 1:])[0, 1] for i in range(300)])
    b = np.mean([np.corrcoef(y[i, :-1], x[i, 1:])[0, 1] for i in range(300)])
    assert a > 0.15 and a > b + 0.1
    x0, y0 = K.make_lagged_driver(50, 100, 0.0, VRng(4))
    assert abs(np.corrcoef(x0.ravel(), y0.ravel())[0, 1]) < 0.05


def test_worlds_are_reproducible():
    a1 = K.make_ecm_world(5, 30, 0.2, 0.5, 1.5, 0.1, VRng(6))
    a2 = K.make_ecm_world(5, 30, 0.2, 0.5, 1.5, 0.1, VRng(6))
    assert np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1]) and not np.array_equal(a1[0], K.make_ecm_world(5, 30, 0.2, 0.5, 1.5, 0.1, VRng(7))[0])


# --- Kritische Werte und Fehlerkorrektur -------------------------------------------------------------------------------------------------------

def test_critical_values_come_from_the_null_simulation_and_are_cached():
    cv = K.critical_values(104)
    assert cv is K.critical_values(104) and cv["eg"][0] < cv["eg"][1] < cv["eg"][2] < -2.5 and cv["df"][0] < cv["df"][1] < cv["df"][2]
    assert -3.7 < cv["eg"][1] < -3.2 and -3.1 < cv["df"][1] < -2.7


def test_exact_adaptation_coefficients():
    assert K.exact_alphas(0.2, 1.0, 1.0) == (-0.2, 0.0) and K.exact_alphas(0.2, 0.0, 1.0) == (0.0, 0.2)
    ay, ax = K.exact_alphas(0.4, 0.5, 2.0)
    assert ay == pytest.approx(-0.2) and ax == pytest.approx(0.1) and ay - 2.0 * ax == pytest.approx(-0.4)           # α_y − β α_x = −a


def test_ecm_estimates_are_close_to_the_truth_for_long_series():
    x, y = K.make_ecm_world(400, 520, 0.3, 0.5, 1.0, 0.0, VRng(8))
    _t, u, _b = K.engle_granger(x, y)
    ty, ay, tx, ax = K.ecm_alpha(x, y, u)
    assert ay.mean() == pytest.approx(-0.15, abs=0.03) and ax.mean() == pytest.approx(0.15, abs=0.03) and (np.abs(ty) > 1.96).mean() > 0.95 and (np.abs(tx) > 1.96).mean() > 0.95


# --- Naiver Test und Differenzen -------------------------------------------------------------------------------------------------------------

def test_naive_and_difference_tests_on_hand_data():
    x = np.array([[1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 6.0, 8.0]])
    y = 2.0 * x + 1.0 + np.array([[0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]])
    r, sig = K.naive_test(x, y)
    assert r[0] > 0.999 and sig[0]
    wiggle = np.array([[1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 3.0, 6.0]])
    r, sig = K.difference_test(x, wiggle)
    assert r[0] == pytest.approx(np.corrcoef(np.diff(x[0]), np.diff(wiggle[0]))[0, 1], abs=1e-12) and abs(r[0]) < 0.7 and not sig[0]                # sieben Differenzen mit kleinem r: bei 5 Freiheitsgraden nicht signifikant


# --- Granger und Transferentropie -----------------------------------------------------------------------------------------------------------

def test_granger_f_and_the_transfer_entropy_identity_on_random_data():
    x, y = K.make_coupled(50, 80, 0.3, VRng(2))
    F, te, n = K.granger(x, y)
    assert n == 79 and np.allclose(2 * te, np.log1p(F / (n - 3)), atol=1e-12) and np.allclose(te, K.te_from_f(F, n), atol=1e-12)


def test_the_identity_by_hand_for_two_residual_sums():
    assert float(K.te_from_f(np.array(10 / (90 / 50)), 53)) == pytest.approx(0.5 * math.log(100 / 90), abs=1e-12) and float(K.te_from_f(np.array(5.5556), 53)) == pytest.approx(0.0526803, abs=1e-6)


def test_f_pvalue_and_critical_value():
    assert K.f_pvalue(0.0, 1, 100) == 1.0 and K.f_pvalue(-1.0, 1, 100) == 1.0
    assert K.f_critical(100) == pytest.approx(3.9361, abs=5e-4) and K.f_pvalue(K.f_critical(100), 1, 100) == pytest.approx(0.05, abs=1e-9)
    assert K.f_pvalue(3.9361, 1, 100) < 0.05 + 1e-4 and K.f_pvalue(100.0, 1, 100) < 1e-10


def test_exact_transfer_entropy_by_hand_and_for_the_default_case():
    assert K.exact_te(0.0) == pytest.approx(0.0, abs=1e-12)
    sx = 1 / (1 - 0.25)
    # Σ aus der Lyapunov-Gleichung: Σxx = 1/(1 − φ²); Σxy und Σyy lösen sich aus den Gleichungen Σxy = φ² Σxy + φ c Σxx, Σyy = φ² Σyy + 2 φ c Σxy + c² Σxx + 1
    c, phi = 0.2, 0.5
    sxy = phi * c * sx / (1 - phi**2)
    syy = (2 * phi * c * sxy + c * c * sx + 1) / (1 - phi**2)
    assert K.exact_te(c) == pytest.approx(0.5 * math.log(1 + c * c * (sx - sxy**2 / syy)), abs=1e-12)
    assert K.exact_te(0.2) == pytest.approx(0.0256, abs=2e-4) and K.exact_te(0.4) == pytest.approx(0.0921, abs=3e-4)


def test_bias_formulas():
    assert K.gauss_te_bias(100) == 0.005 and K.plugin_te_bias(100, 3) == pytest.approx(0.06, abs=1e-12) and K.plugin_te_bias(260, 5) == pytest.approx(5 * 16 / 520)


def test_the_plugin_transfer_entropy_of_a_copy_is_ln_2_and_of_nothing_is_zero(de_bruijn_pair):
    x, y = de_bruijn_pair
    assert K.te_plugin(x, y, 2) == pytest.approx(math.log(2), abs=1e-12)
    rng = VRng(1)
    assert K.te_plugin(rng.normal((4000,)), rng.normal((4000,)), 2) < 0.01


def test_the_permutation_test_returns_a_p_value_between_one_over_b_plus_one_and_one(de_bruijn_pair):
    x, y = de_bruijn_pair
    te, p, corrected = K.te_permutation(x, y, 2, 99, VRng(3))
    assert te == pytest.approx(math.log(2), abs=1e-12) and 0.01 <= p <= 1.0 and corrected < te
    xs, ys = VRng(4).normal((300,)), VRng(5).normal((300,))
    p_ind = K.te_permutation(xs, ys, 3, 99, VRng(6))[1]
    assert 0.01 <= p_ind <= 1.0


def test_unknown_inputs_are_not_silently_accepted():
    with pytest.raises(ValueError):
        K.make_ecm_world(2, 30, 0.2, 1.0, 1.0, 0.0, VRng(1))[0].reshape(7, 7)
    assert K.exact_te(C.COUPLING_MAX) > K.exact_te(C.DEFAULT_COUPLING) > 0
