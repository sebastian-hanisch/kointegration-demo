"""Abbildungen: jede wird gebaut, hat die erwarteten Spuren und Achsen."""

import numpy as np
import pytest

import koi_constants as C
import koi_evaluation as ev
import koi_visualization as V


@pytest.fixture(scope="module")
def grid():
    return ev.load_grid()


def test_every_chart_builds_and_locks_its_axes(grid):
    a = ev.analyse(ev.Settings(t=104, seed=3))
    b = ev.analyse(ev.Settings(world="coupled", seed=3))
    figs = [V.build_series(a), V.build_series(b), V.build_null(a), V.build_e1(grid["power"]), V.build_e2(grid["null"]), V.build_e3(grid["diff"]), V.build_e4(grid["adapt"], 104, 0.2),
            V.build_e5(grid["granger"]), V.build_e6(grid["bias"]), V.build_e7(grid["lagged"])]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and len(fig.data) >= 1


def test_the_series_chart_has_a_second_panel_with_the_spread_only_in_the_ecm_world():
    a = ev.analyse(ev.Settings(a=0.2, seed=3))
    fig = V.build_series(a)
    assert [t.name for t in fig.data] == ["x", "y", "Abstand"] and len(fig.data[0].x) == 104 and len(fig.layout.shapes) == 3                  # Nulllinie und das exakte ±1,96-Band
    flat = V.build_series(ev.analyse(ev.Settings(a=0.0, seed=3)))
    assert len(flat.layout.shapes) == 1                                                                                                   # ohne Beziehung wandert der Abstand: kein Band
    assert [t.name for t in V.build_series(ev.analyse(ev.Settings(world="lagged", seed=3))).data] == ["x", "y"]


def test_the_null_chart_marks_three_critical_values_and_the_observed_statistic():
    a = ev.analyse(ev.Settings(t=52, seed=3))
    fig = V.build_null(a)
    lines = [s.x0 for s in fig.layout.shapes]
    assert lines[:3] == list(a.cv["eg"]) and lines[3] == pytest.approx(a.extra["t_eg"]) and len(fig.data[0].x) == C.NULL_RUNS


def test_e1_has_four_lines_and_the_five_percent_reference(grid):
    fig = V.build_e1(grid["power"])
    assert [t.name for t in fig.data] == ["Engle–Granger, T = 104", "naiver Test, T = 104", "Engle–Granger, T = 260", "naiver Test, T = 260"] and list(fig.data[0].x) == list(C.E1_AS)
    assert fig.layout.shapes[0].y0 == 5 and fig.data[0].y[0] == pytest.approx(100 * [c for c in grid["power"] if c["T"] == 104 and c["a"] == 0.0][0]["eg"])


def test_e2_shows_the_quantiles_against_the_table_value(grid):
    fig = V.build_e2(grid["null"])
    assert [t.name for t in fig.data] == ["simuliertes 5-%-Quantil", "simuliertes 1-%-Quantil"] and list(fig.data[0].x) == sorted(C.E2_TS) and fig.layout.shapes[0].y0 == pytest.approx(-3.3377)


def test_e3_to_e7_values(grid):
    e3 = V.build_e3(grid["diff"])
    assert [t.name for t in e3.data] == ["Test auf die Differenzen", "Engle–Granger"] and len(e3.data[0].x) == 4
    e4 = V.build_e4(grid["adapt"], 104, 0.2)
    assert list(e4.data[0].x) == ["nur y passt sich an", "beide je zur Hälfte", "nur x passt sich an"] and e4.data[0].y[0] == pytest.approx(100.0, abs=0.5)
    e5 = V.build_e5(grid["granger"])
    assert [t.name for t in e5.data] == ["x → y erkannt", "y → x erkannt"] and e5.layout.shapes[0].y0 == 5
    e6 = V.build_e6(grid["bias"])
    assert [t.name for t in e6.data] == ["gaußsch", "gaußsch: Näherung", "Plug-in, 3 Klassen", "Plug-in, 3 Klassen: Näherung", "Plug-in, 5 Klassen", "Plug-in, 5 Klassen: Näherung"] and e6.layout.yaxis.type == "log"
    assert list(e6.data[0].x) == sorted(C.E6_TS) and np.all(np.array(e6.data[2].y) > np.array(e6.data[0].y))
    e7 = V.build_e7(grid["lagged"])
    assert [t.name for t in e7.data] == ["x → y „erkannt“", "y → x „erkannt“"] and e7.layout.shapes[0].y0 == 5
