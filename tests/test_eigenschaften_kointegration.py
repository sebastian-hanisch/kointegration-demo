"""Eigenschaften: Invarianzen, Asymmetrie des Engle–Granger-Tests, Schranken und Monotonie, Reproduzierbarkeit."""

import numpy as np
import pytest

import koi_algorithm as K
from prs_scenario import VRng


def _ecm(seed=3, a=0.2):
    return K.make_ecm_world(20, 120, a, 1.0, 1.0, 0.0, VRng(seed))


def test_the_engle_granger_statistic_is_invariant_under_shifting_and_scaling_both_series():
    x, y = _ecm()
    t, _u, b = K.engle_granger(x, y)
    t2, _u2, b2 = K.engle_granger(3.0 * x + 7.0, 0.25 * y - 2.0)
    assert np.allclose(t, t2, atol=1e-9) and np.allclose(b2, b * 0.25 / 3.0, atol=1e-9)


def test_the_regression_direction_matters_so_x_and_y_are_not_interchangeable():
    x, y = _ecm()
    assert not np.allclose(K.engle_granger(x, y)[0], K.engle_granger(y, x)[0], atol=1e-3)


def test_dickey_fuller_is_invariant_under_scaling_and_shifting_a_series():
    x, _y = _ecm()
    assert np.allclose(K.df_tstat(x, const=True), K.df_tstat(5.0 * x - 3.0, const=True), atol=1e-9)


def test_granger_f_and_transfer_entropy_are_invariant_under_scaling_and_shifting():
    x, y = K.make_coupled(10, 100, 0.3, VRng(2))
    F, te, _n = K.granger(x, y)
    F2, te2, _ = K.granger(2.0 * x + 1.0, -3.0 * y + 5.0)
    assert np.allclose(F, F2, atol=1e-8) and np.allclose(te, te2, atol=1e-12)


def test_the_plugin_transfer_entropy_is_invariant_under_monotone_transforms_and_not_negative():
    x, y = K.make_coupled(30, 150, 0.3, VRng(3))
    for i in range(30):
        base = K.te_plugin(x[i], y[i], 3)
        assert base == pytest.approx(K.te_plugin(np.exp(x[i]), y[i] ** 3, 3), abs=1e-12) and base >= -1e-12


def test_the_permutation_p_value_is_larger_for_independent_than_for_coupled_series():
    xc, yc = K.make_coupled(20, 260, 0.6, VRng(4))
    xi, yi = K.make_coupled(20, 260, 0.0, VRng(5))
    rng = VRng(6)
    pc = np.mean([K.te_permutation(xc[i], yc[i], 3, 49, rng)[1] for i in range(20)])
    pi = np.mean([K.te_permutation(xi[i], yi[i], 3, 49, rng)[1] for i in range(20)])
    assert pc < 0.15 < pi


def test_the_exact_transfer_entropy_grows_with_the_coupling():
    values = [K.exact_te(c) for c in (0.0, 0.1, 0.2, 0.4, 0.6)]
    assert values == sorted(values) and values[0] == 0.0 and values[-1] < 0.5


def test_the_exact_spread_variance_falls_with_a_and_the_estimated_adjustment_grows_with_a():
    var = [K.exact_spread_variance(a, 1.0, 0.0) for a in (0.05, 0.1, 0.2, 0.5)]
    assert var == sorted(var, reverse=True)
    est = []
    for a in (0.05, 0.2, 0.5):
        x, y = K.make_ecm_world(300, 200, a, 1.0, 1.0, 0.0, VRng(7))
        _t, u, _b = K.engle_granger(x, y)
        est.append(-K.ecm_alpha(x, y, u)[1].mean())
    assert est == sorted(est)


def test_a_larger_adjustment_makes_the_engle_granger_statistic_more_negative():
    medians = []
    for a in (0.0, 0.1, 0.3, 0.5):
        x, y = K.make_ecm_world(500, 104, a, 1.0, 1.0, 0.0, VRng(8))
        medians.append(np.median(K.engle_granger(x, y)[0]))
    assert medians == sorted(medians, reverse=True)


def test_the_same_seed_gives_the_same_statistics():
    x1, y1 = _ecm(5)
    x2, y2 = _ecm(5)
    assert np.array_equal(K.engle_granger(x1, y1)[0], K.engle_granger(x2, y2)[0]) and not np.array_equal(K.engle_granger(x1, y1)[0], K.engle_granger(*_ecm(6))[0])
