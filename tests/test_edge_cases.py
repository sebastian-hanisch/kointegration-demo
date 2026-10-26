"""Randfälle (vor dem ersten Mutationstest): kürzeste Reihen, a = 0, alle Anpassungsanteile, wenige Wochen für viele Klassen, Grenzen der Verteilungsfunktionen."""

import math

import numpy as np
import pytest

import koi_algorithm as K
import koi_constants as C
from prs_scenario import VRng


@pytest.mark.parametrize("world", list(C.WORLDS))
def test_every_world_runs_at_the_shortest_and_longest_series(world):
    for T in (min(C.T_OPTIONS), max(C.T_OPTIONS)):
        if world == "ecm":
            x, y = K.make_ecm_world(3, T, C.A_MAX, 0.5, C.BETA_OPTIONS[-1], C.GAMMA_MAX, VRng(2))
        elif world == "coupled":
            x, y = K.make_coupled(3, T, C.COUPLING_MAX, VRng(2))
        else:
            x, y = K.make_lagged_driver(3, T, C.DRIVER_MAX, VRng(2))
        assert x.shape == (3, T) and np.isfinite(x).all() and np.isfinite(y).all()
        t, u, b = K.engle_granger(x, y)
        F, te, n = K.granger(x, y)
        assert np.isfinite(t).all() and np.isfinite(F).all() and n == T - 1 and (te >= -1e-12).all()


def test_the_zero_adjustment_gives_independent_walks_for_every_share_and_beta():
    for s in C.SHARES:
        for beta in C.BETA_OPTIONS:
            x, y = K.make_ecm_world(2, 40, 0.0, s, beta, 0.0, VRng(3))
            x2, y2 = K.make_ecm_world(2, 40, 0.0, 1.0, 1.0, 0.0, VRng(3))
            assert np.allclose(np.diff(x, axis=1), np.diff(x2, axis=1))                  # ohne Anpassung hängt Δx nicht an s und β


def test_plugin_transfer_entropy_with_few_weeks_and_many_classes_stays_finite_and_nonnegative():
    x, y = K.make_coupled(20, 26, 0.3, VRng(4))
    for k in C.CLASS_OPTIONS:
        for i in range(20):
            te = K.te_plugin(x[i], y[i], k)
            assert math.isfinite(te) and te >= -1e-12


def test_classes_are_balanced_by_rank_and_stable_for_ties():
    v = np.arange(10.0)
    assert list(K._classes(v, 2)) == [0] * 5 + [1] * 5 and list(K._classes(v[::-1], 2)) == [1] * 5 + [0] * 5
    assert list(np.bincount(K._classes(np.zeros(9), 3))) == [3, 3, 3]                    # Gleichstände: nach Reihenfolge, trotzdem gleich besetzt
    assert K._classes(np.array([5.0]), 3).tolist() == [0]


def test_p_value_and_critical_value_boundaries():
    assert K.f_pvalue(1e-12, 1, 10) == pytest.approx(1.0, abs=1e-3) and K.f_pvalue(1e6, 1, 10) < 1e-9
    assert K.f_critical(10, 0.05) > K.f_critical(1000, 0.05) > 3.8 and K.f_critical(10, 0.01) > K.f_critical(10, 0.05)


def test_exact_formulas_at_the_boundaries():
    assert math.isinf(K.exact_spread_variance(0.0, 1.0, 0.0)) and K.exact_spread_variance(1.0, 1.0, 1.0) == pytest.approx(1.0, abs=1e-12)     # a = 1: z_t = Neuerung
    assert K.exact_alphas(0.0, 0.5, 2.0) == (-0.0, 0.0) and K.exact_alphas(0.5, 0.5, 1.0) == (-0.25, 0.25)
    assert K.exact_te(0.0) == 0.0 and K.exact_te(-0.3) == pytest.approx(K.exact_te(0.3), abs=1e-12)


def test_critical_values_exist_for_every_series_length_in_the_slider():
    for T in C.T_OPTIONS:
        cv = K.critical_values(T)
        assert all(math.isfinite(v) for v in cv["eg"] + cv["df"]) and cv["eg"][0] < cv["eg"][1] < cv["eg"][2] < 0


def test_the_difference_test_uses_one_week_fewer_than_the_level_test():
    x, y = K.make_ecm_world(500, 26, 0.0, 1.0, 1.0, 0.0, VRng(5))
    _r, sig_n = K.naive_test(x, y)
    _rd, sig_d = K.difference_test(x, y)
    assert sig_n.mean() > 0.3 and sig_d.mean() == pytest.approx(0.05, abs=0.03)


def _pair_with_corr(n, r, seed=7):
    """Zwei Reihen mit der Stichprobenkorrelation genau r (Gram–Schmidt aus Zufallszahlen)."""
    rng = VRng(seed)
    a, e = rng.normal((n,)), rng.normal((n,))
    a, e = a - a.mean(), e - e.mean()
    e = e - (e @ a) / (a @ a) * a
    a, e = a / np.linalg.norm(a), e / np.linalg.norm(e)
    return a, r * a + math.sqrt(1 - r * r) * e


def test_the_naive_test_has_its_threshold_at_t_minus_two_degrees_of_freedom():
    """T = 10: das 5-%-Niveau liegt bei |r| = 2,306/√(2,306² + 8) = 0,632. Knapp darüber ist der Test „signifikant“, knapp darunter nicht (jede andere Freiheitsgradzahl verschiebt die Grenze über dieses Fenster)."""
    for r, expected in ((0.640, True), (0.625, False), (-0.640, True), (-0.625, False)):
        a, b = _pair_with_corr(10, r)
        rr, sig = K.naive_test((3.0 * a + 5.0)[None, :], (2.0 * b - 1.0)[None, :])
        assert rr[0] == pytest.approx(r, abs=1e-12) and bool(sig[0]) is expected, r


def test_the_difference_test_works_on_t_minus_one_differences_with_the_same_threshold():
    for r, expected in ((0.640, True), (0.625, False)):
        dx, dy = _pair_with_corr(10, r, seed=8)
        x, y = np.concatenate([[1.0], 1.0 + np.cumsum(dx)])[None, :], np.concatenate([[2.0], 2.0 + np.cumsum(dy)])[None, :]
        rr, sig = K.difference_test(x, y)
        assert rr[0] == pytest.approx(r, abs=1e-9) and bool(sig[0]) is expected, r


def test_the_permutation_p_value_counts_ties_and_adds_one_to_numerator_and_denominator():
    x, y = VRng(21).normal((12,)), VRng(22).normal((12,))
    B = 99
    te, p, corrected = K.te_permutation(x, y, 2, B, VRng(23))
    perm = np.argsort(VRng(23).uniform((B, len(x))), axis=1)
    null = np.array([K.te_plugin(x[q], y, 2) for q in perm])
    assert te == K.te_plugin(x, y, 2) and p == (1 + np.sum(null >= te)) / (B + 1) and corrected == pytest.approx(te - null.mean(), abs=1e-15)
    assert np.sum(null == te) > 0 and p != (1 + np.sum(null > te)) / (B + 1)                      # mit Gleichständen: „≥“ und „>“ unterscheiden sich
    assert K.te_permutation(x, y, 2, 9, VRng(1))[1] * 10 == pytest.approx(round(K.te_permutation(x, y, 2, 9, VRng(1))[1] * 10))


def test_the_permutation_p_value_of_a_perfect_copy_is_small_but_not_below_one_over_b_plus_one(de_bruijn_pair):
    x, y = de_bruijn_pair
    p = K.te_permutation(x, y, 2, 99, VRng(3))[1]
    assert 0.01 <= p <= 0.05                                                  # nur wenige Mischungen (gleiche Zustandsfolge) erreichen ln 2 wieder
