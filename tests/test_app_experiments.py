"""App: jedes Preset, der Permalink, jedes Experiment-Auswahlfeld einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import koi_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    p = C.PRESETS[name]
    assert at.session_state["world_select"] == p["world"] and at.session_state["t_select"] == p["t"] and at.session_state["a_slider"] == p["a"] and at.session_state["share_select"] == p["share"]
    assert at.session_state["gamma_slider"] == p["gamma"] and at.session_state["coupling_slider"] == p["coupling"] and at.session_state["classes_select"] == p["classes"]
    assert len(at.get("plotly_chart")) == (2 if p["world"] == "ecm" else 1) + 7             # Hauptansicht: Reihen (und Nullverteilung); E1 bis E7: je eine Abbildung


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["world"] = "coupled"
    at.query_params["a"] = "9"
    at.query_params["c"] = "-3"
    at.query_params["t"] = "260"
    at.query_params["k"] = "4"                        # ungültig: wird ignoriert
    at.query_params["share"] = "0.5"
    at.run()
    assert not at.exception
    assert at.session_state["world_select"] == "coupled" and at.session_state["a_slider"] == C.A_MAX and at.session_state["coupling_slider"] == C.COUPLING_MIN
    assert at.session_state["t_select"] == 260 and at.session_state["classes_select"] == C.DEFAULT_CLASSES and at.session_state["share_select"] == 0.5


def test_permalink_ignores_unknown_choices():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["world"] = "stern"
    at.query_params["t"] = "77"
    at.run()
    assert not at.exception and at.session_state["world_select"] == C.DEFAULT_WORLD and at.session_state["t_select"] == C.DEFAULT_T


def test_metrics_depend_on_the_world():
    at = _app()
    assert [m.label for m in at.metric] == ["Naiver Test auf r", "Engle–Granger t", "α̂_y (y passt sich an)", "α̂_x (x passt sich an)"] and all(m.value for m in at.metric)
    at.sidebar.selectbox(key="world_select").set_value("coupled").run()
    assert not at.exception and [m.label for m in at.metric] == ["Granger-F x → y", "Granger-F y → x", "Transferentropie x → y (nats)", "Plug-in x → y (nats)"]


def test_the_headline_verdict_depends_on_the_world_and_the_coupling():
    at = _app()
    assert "Es gibt eine Beziehung (Anpassungstempo 0,20), aber der Test findet sie nicht" in _texts(at)
    at.sidebar.button(key="preset_Verzögerte Ursache").click().run()
    assert not at.exception and "obwohl x nie auf y wirkt" in _texts(at)
    at.sidebar.button(key="preset_Gekoppelte Reihen").click().run()
    assert not at.exception and "findet die Richtung x → y (Kopplung c = 0,2) und nicht die umgekehrte" in _texts(at)


def test_the_table_has_the_rows_of_the_world_and_the_truth_column():
    at = _app()
    rows = at.dataframe[0].value
    assert list(rows.columns) == ["Prüfung", "Wert", "Schwelle", "Entscheidung", "Wahrheit"] and len(rows) == 9 and "kointegriert" in list(rows["Wahrheit"])
    at.sidebar.selectbox(key="world_select").set_value("lagged").run()
    assert len(at.dataframe[0].value) == 6


def test_every_sidebar_input_runs_at_its_corners():
    at = _app()
    for key, value in (("t_select", 26), ("t_select", 520), ("share_select", 0.0), ("share_select", 0.5), ("beta_select", 0.5), ("beta_select", 2.0), ("classes_select", 8), ("classes_select", 3), ("t_select", 104)):
        at.sidebar.select_slider(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])
    for key, value in (("a_slider", 0.0), ("a_slider", 0.5), ("gamma_slider", 0.5), ("coupling_slider", 0.6), ("driver_slider", 1.2), ("driver_slider", 0.0)):
        at.sidebar.slider(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])
    for w in ("coupled", "lagged", "ecm"):
        at.sidebar.selectbox(key="world_select").set_value(w).run()
        assert not at.exception, (w, [str(e) for e in at.exception])


def test_pdf_button_then_download_in_every_world():
    for w in C.WORLDS:
        at = _app()
        at.sidebar.selectbox(key="world_select").set_value(w).run()
        at.button(key="pdf_make").click().run()
        assert not at.exception and len(at.get("download_button")) == 1, w


def test_experiment_selectors_run():
    at = _app()
    for key, value in (("e1_t", 260), ("e2_t", 26), ("e2_t", 520), ("e3_gamma", 0.5), ("e3_a", 0.0), ("e4_t", 260), ("e4_a", 0.5), ("e4_share", 0.0), ("e4_share", 0.5), ("e5_c", 0.0), ("e5_c", 0.4),
                       ("e6_k", 5), ("e7_b", 0.0), ("e7_b", 1.2)):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_the_experiment_sections_show_their_verdicts():
    text = _texts(_app())
    for needle in ("T = 104: ohne Beziehung (a = 0) sagt der naive Test in", "die simulierten Quantile der Engle–Granger-Statistik sind", "Kurzfristige Kopplung 0,0, Anpassungstempo 0,2: der Test auf die Differenzen",
                   "T = 104, Tempo 0,20, nur y passt sich an: α_y ist in", "Kopplung c = 0,2: x → y erkannt in", "Ohne jede Kopplung: gaußsche Schätzung", "Verzögerte gemeinsame Ursache der Stärke 0,8"):
        assert needle in text, needle
