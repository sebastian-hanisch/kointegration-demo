"""Konstanten: Regler-Grenzen, Welten, Raster der Messreihen, Farben und Presets.

Zufallserzeuger, Student- und Betaverteilung kommen als wörtliche Kopien aus pearson-demo (prs_rng.py, prs_constants.py, prs_scenario.py, prs_algorithm.py).
Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): T = 520 Wochen; die Nullsimulation der kritischen Werte (5 000 Paare unabhängiger Irrfahrten, fester Seed) kostet bei T = 520 rund 0,5 s, alles
Übrige Millisekunden, die Plug-in-Transferentropie mit 199 Mischungen unter einer Sekunde. Die Wiederholungsexperimente stehen vorgerechnet im Raster (data/cointegration_grid.json)."""

from prs_constants import SEED_MAX  # noqa: F401  (Wiederverwendung der Kopie)

WORLDS = {
    "ecm": "Fehlerkorrektur (x wandert, y wird zurückgezogen)",
    "coupled": "Gekoppelte Reihen (x wirkt auf y)",
    "lagged": "Verzögerte gemeinsame Ursache (keine Wirkung von x auf y)",
}
DEFAULT_WORLD = "ecm"
T_OPTIONS = (26, 52, 104, 260, 520)
DEFAULT_T = 104
A_MIN, A_MAX, DEFAULT_A = 0.0, 0.5, 0.2              # Anpassungstempo: der Abstand folgt einem AR(1) mit Koeffizient 1 − a
SHARES = {1.0: "nur y passt sich an", 0.5: "beide je zur Hälfte", 0.0: "nur x passt sich an"}
DEFAULT_SHARE = 1.0
BETA_OPTIONS = (0.5, 1.0, 1.5, 2.0)                  # Faktor des Abstands z = y − β x
DEFAULT_BETA = 1.0
GAMMA_MIN, GAMMA_MAX, DEFAULT_GAMMA = 0.0, 0.5, 0.0  # kurzfristige Kopplung der Neuerungen
COUPLING_MIN, COUPLING_MAX, DEFAULT_COUPLING = 0.0, 0.6, 0.2    # gekoppelte Reihen: Wirkung von x_{t−1} auf y_t
DRIVER_MIN, DRIVER_MAX, DEFAULT_DRIVER = 0.0, 1.2, 0.8          # verzögerte Ursache: Stärke
CLASS_OPTIONS = (3, 5, 8)                            # Klassen des Plug-in-Schätzers der Transferentropie
DEFAULT_CLASSES = 3
DEFAULT_SEED = 35

PHI = 0.5                                            # AR(1)-Koeffizient der gekoppelten Reihen und der verborgenen Ursache
ALPHA = 0.05                                         # Niveau aller Tests
BURN = 50                                            # Einschwingzeit der stationären Welten
NULL_RUNS, NULL_SEED = 5000, 777                     # Nullsimulation der kritischen Werte (unabhängige Irrfahrten)
PERMUTATIONS = 199                                   # Mischungen des Permutationstests der Transferentropie
Z_ADAPT = 1.96                                       # |t| über diesem Wert: „passt sich an“

# Raster der Messreihen (tools/make_grid.py): 4 Seeds · BATCHES Pakete · BATCH Wiederholungen
GRID_SEEDS = (101, 102, 103, 104)
BATCHES, BATCH = 4, 250
E1_TS, E1_AS = (104, 260), (0.0, 0.05, 0.1, 0.2, 0.5)
E2_TS = (26, 52, 104, 260, 520)
E3_GAMMAS, E3_AS, E3_T = (0.0, 0.5), (0.0, 0.2), 104
E4_TS, E4_AS, E4_SHARES = (104, 260), (0.2, 0.5), (1.0, 0.5, 0.0)
E5_COUPLINGS, E5_T = (0.0, 0.1, 0.2, 0.4), 104
E6_TS, E6_KS, E6_BATCH = (50, 100, 260, 1000), (3, 5), 250
E7_T = 104

COLOR_X = "#4c78a8"
COLOR_Y = "#e45756"
COLOR_Z = "#54a24b"
COLOR_LIMIT = "#f58518"
COLOR_GREY = "#9d9d9d"
COLOR_DARK = "#222222"

_BASE = {"world": "ecm", "t": 104, "a": 0.2, "share": 1.0, "beta": 1.0, "gamma": 0.0, "coupling": 0.2, "driver": 0.8, "classes": 3, "seed": DEFAULT_SEED}
PRESETS = {
    "Standardfall (Voreinstellung)": dict(_BASE),
    "Keine Beziehung": {**_BASE, "a": 0.0},
    "Langsame Anpassung": {**_BASE, "a": 0.05, "t": 260},
    "Beide passen sich an": {**_BASE, "share": 0.5},
    "Kurzfristige Kopplung": {**_BASE, "a": 0.0, "gamma": 0.5},
    "Gekoppelte Reihen": {**_BASE, "world": "coupled"},
    "Verzögerte Ursache": {**_BASE, "world": "lagged"},
}
# Die Hilfetexte nennen exakte Werte und Zahlen der Auswertung; tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Zwei Irrfahrten mit echter Beziehung (Anpassungstempo 0,2, y passt sich an): der naive Test nennt r = 0,87 „signifikant“, der Engle–Granger-Test findet die Beziehung in dieser Stichprobe nicht (t = -3,17 bei der Schwelle -3,35), im Raster aber in 62,3 % der Läufe.",
    "Keine Beziehung": "Unabhängige Irrfahrten: der naive Test nennt sie trotzdem „signifikant“ (r = 0,74), der Engle–Granger-Test findet nichts (t = -1,58 bei der Schwelle -3,35).",
    "Langsame Anpassung": "Tempo 0,05 bei 260 Wochen: der Abstand hat die Autokorrelation 0,95, der Engle–Granger-Test übersieht die Beziehung (t = -3,05 bei der Schwelle -3,35); im Raster findet er sie nur in 21,2 % der Läufe.",
    "Beide passen sich an": "Beide Partner ziehen den Abstand je zur Hälfte zurück: α̂_y = -0,10 und α̂_x = 0,08 (wahr -0,10 und 0,10); beide t-Tests liegen in dieser Stichprobe knapp über 1,96 (t = -2,21 und 2,09).",
    "Kurzfristige Kopplung": "Keine langfristige Beziehung, aber kurzfristige Kopplung 0,5: der Test auf die Differenzen ist „signifikant“ (r = 0,46), der Engle–Granger-Test findet nichts (t = -1,58).",
    "Gekoppelte Reihen": "Kopplung 0,2 von x auf y: Granger-F x → y 11,90 (Schwelle 3,94), y → x 2,37; gaußsche Transferentropie 0,0562 (exakt 0,0256 plus 0,0049 Überschuss); der Plug-in-Schätzer mit 3 Klassen findet die Richtung in dieser Stichprobe nicht (p = 0,450).",
    "Verzögerte Ursache": "x wirkt nie auf y, doch Granger-F x → y ist 25,02 (Schwelle 3,94) und y → x 1,50: eine Richtung ohne Wirkung.",
}
