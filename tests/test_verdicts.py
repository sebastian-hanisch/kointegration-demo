"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np

import koi_constants as C
import koi_evaluation as ev


def _ecm(a=0.2, t_eg=-4.0, cut=-3.35, sig=True, share=1.0, ty=-5.0, tx=0.3, gamma=0.0, r_diff=0.05, sig_diff=False, t=104):
    s = ev.Settings(world="ecm", t=t, a=a, share=share, gamma=gamma)
    rows = [ev.Row("eg", "Engle–Granger", t_eg, cut, "x", "y")]
    extra = {"t_eg": t_eg, "t_alpha_y": ty, "t_alpha_x": tx, "r": 0.9, "r_diff": r_diff, "sig": sig, "sig_diff": sig_diff, "eg_cut": cut}
    return ev.Analysis(s, np.zeros(3), np.zeros(3), rows, np.zeros(3), {"eg": (-4.0, cut, -3.0), "df": (-3.4, -2.9, -2.6)}, extra)


def test_main_verdict_ecm_four_cases():
    kind, text = ev.verdict_main(_ecm(a=0.2, t_eg=-4.0))
    assert kind == "success" and "findet die Beziehung (Engle–Granger t = -4,00 bei der Schwelle -3,35)" in text and "**signifikant** verbunden, und das hätte er auch ohne jede Beziehung getan." in text
    kind, text = ev.verdict_main(_ecm(a=0.2, t_eg=-4.0, sig=False))
    assert kind == "success" and text.endswith("nicht signifikant verbunden.") and "hätte er auch" not in text
    kind, text = ev.verdict_main(_ecm(a=0.2, t_eg=-3.0))
    assert kind == "warning" and "aber der Test findet sie nicht (Engle–Granger t = -3,00 bei der Schwelle -3,35)" in text and "Autokorrelation 0,80" in text and text.endswith("**signifikant** verbunden.")
    kind, text = ev.verdict_main(_ecm(a=0.0, t_eg=-3.5))
    assert kind == "warning" and text.startswith("Fehlalarm: es gibt keine Beziehung, aber der Test meldet eine (Engle–Granger t = -3,50")
    kind, text = ev.verdict_main(_ecm(a=0.0, t_eg=-1.0, sig=False))
    assert kind == "info" and "findet keine Beziehung" in text and "richtig; der naive Test auf r nennt die Reihen nicht signifikant verbunden." in text


def test_main_verdict_threshold_is_strict_at_the_critical_value():
    assert ev.verdict_main(_ecm(a=0.2, t_eg=-3.35, cut=-3.35))[0] == "warning"          # Statistik gleich Schwelle: nicht „gefunden“
    assert ev.verdict_main(_ecm(a=0.2, t_eg=-3.3501, cut=-3.35))[0] == "success"


def test_adaptation_verdict_names_who_adapts():
    texts = {k: ev.verdict_adapt(_ecm(share=s, ty=ty, tx=tx))[1] for k, (s, ty, tx) in {"y": (1.0, -5.0, 0.3), "x": (0.0, -0.4, 4.0), "both": (0.5, -3.0, 2.5), "none": (1.0, -1.0, 1.0)}.items()}
    assert "y passt sich an (t für α_y -5,00, für α_x 0,30); wahr: nur y passt sich an, Tempo 0,20." in texts["y"]
    assert "x passt sich an (t für α_y -0,40, für α_x 4,00); wahr: nur x passt sich an" in texts["x"] and "beide passen sich an" in texts["both"] and "keiner passt sich nachweisbar an" in texts["none"]
    assert "Der Partner, der sich nicht anpasst, führt" in texts["y"]
    assert "keiner" in ev.verdict_adapt(_ecm(ty=-1.96, tx=1.96))[1] and "y passt sich an" in ev.verdict_adapt(_ecm(ty=-1.9601, tx=1.0))[1]       # |t| > 1,96 strikt


def test_difference_verdict_depends_on_the_short_term_coupling():
    kind, text = ev.verdict_difference(_ecm(gamma=0.0, r_diff=0.08, sig_diff=False))
    assert "r = 0,08, nicht signifikant." in text and "bei einer Beziehung ohne kurzfristige Kopplung sehen sie fast nichts." in text
    text = ev.verdict_difference(_ecm(gamma=0.5, r_diff=0.46, sig_diff=True))[1]
    assert "r = 0,46, signifikant." in text and "ganz unabhängig davon, ob es eine langfristige Beziehung gibt." in text


def _directed(world="coupled", coupling=0.2, fxy=11.9, fyx=2.4, crit=3.94, classes=3):
    s = ev.Settings(world=world, coupling=coupling, classes=classes)
    extra = {"F_xy": fxy, "F_yx": fyx, "crit": crit, "n": 103, "te_xy": 0.0562, "te_yx": 0.0117, "p_xy": 0.001, "p_yx": 0.13, "plugin_xy": 0.0636, "plugin_xy_p": 0.45, "plugin_xy_corrected": -0.0017,
             "plugin_yx_p": 0.8}
    return ev.Analysis(s, np.zeros(3), np.zeros(3), [], None, None, extra)


def test_main_verdict_for_directed_worlds():
    kind, text = ev.verdict_main(_directed())
    assert kind == "success" and "Granger-F x → y 11,90 (Schwelle 3,94), y → x 2,40" in text and "findet die Richtung x → y (Kopplung c = 0,2) und nicht die umgekehrte." in text
    kind, text = ev.verdict_main(_directed(fyx=5.0))
    assert kind == "warning" and "beide Richtungen sind „signifikant“, in Wahrheit wirkt nur x auf y." in text
    kind, text = ev.verdict_main(_directed(fxy=2.0))
    assert kind == "warning" and "die Kopplung c = 0,2 ist da, aber der Test findet sie nicht" in text
    assert ev.verdict_main(_directed(fxy=3.94))[0] == "warning" and ev.verdict_main(_directed(fxy=3.9401))[0] == "success"      # F strikt über der Schwelle
    kind, text = ev.verdict_main(_directed(coupling=0.0, fxy=1.0, fyx=1.0))
    assert kind == "info" and "ist jede „gefundene“ Richtung ein Fehlalarm (hier keiner)." in text and "(hier einer)" in ev.verdict_main(_directed(coupling=0.0, fxy=9.0))[1]


def test_main_verdict_for_the_lagged_driver():
    kind, text = ev.verdict_main(_directed(world="lagged", fxy=25.0, fyx=1.5))
    assert kind == "warning" and "„findet“ die Richtung x → y, obwohl x nie auf y wirkt" in text and "von vor zwei Wochen" in text and "Vorhersagbarkeit, keine Wirkung." in text
    kind, text = ev.verdict_main(_directed(world="lagged", fxy=1.0, fyx=1.0))
    assert kind == "info" and "findet der Test keine Richtung" in text and "Experiment 7" in text


def test_transfer_entropy_verdict_quotes_the_identity_and_the_biases():
    kind, text = ev.verdict_te(_directed(classes=5))
    assert kind == "info" and "Transferentropie x → y 0,0562 nats = ½ ln(1 + F/(n − 3)) = 0,0562" in text and "um 1/(2n) = 0,0049 über null" in text
    assert "Plug-in-Schätzer mit 5 Klassen liefert 0,0636 (Permutationstest p = 0,450, nach Abzug des Mittels der Mischungen -0,0017)" in text and "Überschuss etwa 0,388" in text


def _power(T, a, eg, naive):
    return {"T": T, "a": a, "eg": eg, "naive": naive}


def test_experiment_verdicts_quote_their_cells():
    cells = [_power(104, 0.0, 0.053, 0.761), _power(104, 0.2, 0.623, 0.995)]
    kind, text = ev.verdict_e1(cells, 104)
    assert kind == "info" and "ohne Beziehung (a = 0) sagt der naive Test in 76,1 % „signifikant“, der Engle–Granger-Test in 5,3 %; bei echter Beziehung (a = 0,2) sind es 99,5 % und 62,3 %" in text
    null = [{"T": 52, "q": [-4.195, -3.464, -3.163], "alarm_table": 0.067, "alarm_live": 0.046, "cut_live": -3.503}]
    assert "T = 52: die simulierten Quantile der Engle–Granger-Statistik sind -4,20 / -3,46 / -3,16 (1 / 5 / 10 %); mit dem tabellierten Wert −3,34 gäbe es 6,7 % Fehlalarm, mit dem simulierten Wert dieser Demo (-3,50) 4,6 %." in ev.verdict_e2(null, 52)[1]
    diff = [{"gamma": 0.5, "a": 0.0, "diff": 0.998, "eg": 0.053, "r_diff": 0.444}]
    kind, text = ev.verdict_e3(diff, 0.5, 0.0)
    assert kind == "warning" and "Kurzfristige Kopplung 0,5, Anpassungstempo 0,0: der Test auf die Differenzen ist in 99,8 % der Läufe „signifikant“, der Engle–Granger-Test findet die Beziehung in 5,3 % (mittleres r der Differenzen 0,444)." in text


def test_experiment_verdicts_for_adaptation_granger_bias_and_the_lagged_driver():
    cell = {"T": 104, "a": 0.2, "share": 1.0, "sig_y": 1.0, "sig_x": 0.069, "alpha_y": -0.231, "alpha_x": -0.004, "true_y": -0.2, "true_x": 0.0}
    assert ("T = 104, Tempo 0,20, nur y passt sich an: α_y ist in 100,0 % der Läufe signifikant, α_x in 6,9 %; mittlere Schätzer α̂_y -0,231 (wahr -0,200) und α̂_x -0,004 (wahr 0,000)." in ev.verdict_e4(cell)[1])
    g = {"c": 0.2, "xy": 0.622, "yx": 0.053, "runs": 4000, "te_xy": 0.0307, "exact": 0.0256, "bias": 0.0049, "identity": 1.2e-16}
    text = ev.verdict_e5(g)[1]
    assert "Kopplung c = 0,2: x → y erkannt in 62,2 %, y → x in 5,3 % der 4000 Läufe; mittlere Transferentropie x → y 0,0307 (exakt 0,0256 plus 1/(2n) = 0,0049); die Identität 2 · TE = ln(1 + F/(n − 3)) stimmt auf 1e-16." in text
    bias = [{"T": 50, "k": 3, "gauss": 0.0116, "plugin": 0.1552, "plugin_pred": 0.1224}, {"T": 1000, "k": 3, "gauss": 0.0005, "plugin": 0.006, "plugin_pred": 0.006}, {"T": 100, "k": 5, "gauss": 0.0, "plugin": 0.0, "plugin_pred": 0.0}]
    kind, text = ev.verdict_e6(bias, 3)
    assert kind == "warning" and "gaußsche Schätzung 0,0116 (T = 50) bis 0,0005 (T = 1000); Plug-in mit 3 Klassen 0,1552 bis 0,0060 (Näherung k (k − 1)²/(2n): 0,1224 bis 0,0060)." in text
    kind, text = ev.verdict_e7({"b": 0.8, "xy": 0.996, "yx": 0.084})
    assert kind == "warning" and "Stärke 0,8: der Granger-Test „findet“ x → y in 99,6 % der Läufe und y → x in 8,4 %, obwohl x nie auf y wirkt." in text
    assert set(C.SHARES) == {1.0, 0.5, 0.0}
