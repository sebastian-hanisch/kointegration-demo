# Gleichlauf, der hält: Kointegration und gerichtete Abhängigkeit – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kointegration-demo.streamlit.app/)** -->

Elftes und letztes Stück der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von [scheinkorrelation-demo](https://github.com/sebastian-hanisch/scheinkorrelation-demo) (Kante `fix`: unabhängige Irrfahrten sind „signifikant“ korreliert, aber manche Paare wandern **zusammen**, und das ist echt), neben [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo), [rangkorrelation-demo](https://github.com/sebastian-hanisch/rangkorrelation-demo), [abhaengigkeitsmasse-demo](https://github.com/sebastian-hanisch/abhaengigkeitsmasse-demo), [korrelationsmatrix-demo](https://github.com/sebastian-hanisch/korrelationsmatrix-demo), [korrelationsnetz-demo](https://github.com/sebastian-hanisch/korrelationsnetz-demo), [aggregationsebene-demo](https://github.com/sebastian-hanisch/aggregationsebene-demo), [uebereinstimmung-demo](https://github.com/sebastian-hanisch/uebereinstimmung-demo), [randabhaengigkeit-demo](https://github.com/sebastian-hanisch/randabhaengigkeit-demo) und [korrelierte-merkmale-demo](https://github.com/sebastian-hanisch/korrelierte-merkmale-demo). Es ist die **erste Demo des Portfolios über Kointegration und gerichtete Abhängigkeit**: zwei Wochenreihen **x und y** in drei Welten mit bekannter Wahrheit (Fehlerkorrektur, gekoppelte stationäre Reihen, verzögerte gemeinsame Ursache). Die Fragen: **Wie unterscheidet man echten von scheinbarem Gleichlauf? Wer passt sich wem an? Was verliert das Differenzieren? Und was sagen Granger-Test und Transferentropie über die Richtung, und was nicht?**

## Warum dieses Problem

Der naive Test auf r kann bei wandernden Reihen nicht zwischen „keine Beziehung“ und „echte Beziehung“ unterscheiden. Befunde (104 Wochen, 4 000 Läufe je Zelle): bei **unabhängigen** Irrfahrten sagt er in **76,1 %** der Läufe „signifikant“, bei echter Beziehung (Anpassungstempo 0,2) in **99,5 %**; der **Engle–Granger-Test** trennt beides (5,3 % gegen **62,3 %**), braucht aber Zeit: bei langsamer Anpassung (0,05) findet er die Beziehung bei 104 Wochen nur in 7,1 %, bei 260 in 21,2 % der Läufe. **Differenzieren**, das Gegenmittel aus Stück 4, macht die echte Beziehung unsichtbar, wenn nur ein Partner sich anpasst (5,2 % „signifikant“ bei Anpassungstempo 0,2) und liest kurzfristige Kopplung als „signifikant“ (99,8 %) ganz ohne langfristige Beziehung. Die **Fehlerkorrektur** zeigt, wer sich anpasst (α_y in 100,0 %, α_x in 6,9 % der Läufe signifikant, wenn nur y sich anpasst; im umgekehrten Fall α_x in 99,3 %, aber α_y in 17,7 %). Bei der **gerichteten Abhängigkeit** findet der Granger-Test die Richtung (62,2 % gegen 5,3 % bei Kopplung 0,2), und die gaußsche Transferentropie ist **dieselbe Zahl** (2 · TE = ln(1 + F/(n − 3))); aber eine **verzögerte gemeinsame Ursache** lässt ihn x → y in **99,6 %** der Läufe „finden“, obwohl x nie auf y wirkt, und der Plug-in-Schätzer der Transferentropie liegt bei 100 Wochen ohne jede Kopplung im Mittel bei 0,0711 nats, mit einer echten Kopplung von 0,2 nur bei 0,0863; zum Vergleich: die **exakte** gaußsche Transferentropie dieser Kopplung ist 0,0256, eine andere Größe ohne Klasseneinteilung.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-11-kointegration.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit gleichem Kern; die gebauten Zahlen stimmen mit der Vormessung überein, mit diesen Ausnahmen:

- **Der PLAN-Stand „noch keine Vormessung“** ist überholt; gemessen sind Fehlalarm und Teststärke des Engle–Granger-Tests, die Fehlerkorrektur, das Differenzieren, die Granger- und Transferentropie-Zahlen und die Verzerrung der Plug-in-Schätzung.
- **Kritische Werte:** die Demo benutzt **simulierte** Schwellen für genau das gewählte T (5 000 Paare unabhängiger Irrfahrten, fester Seed), keine Tabellen. Die Quantile bei 520 Wochen treffen den tabellierten Wert aus dem Gedächtnis; bei kurzen Reihen liegen sie deutlich darunter (Experiment 2). Die Schwelle hat selbst einen Fehler: der Fehlalarm der simulierten Schwellen liegt in frischen Läufen zwischen etwa 4,2 % und 5,8 % statt bei 5 %.
- **Die Identität 2 · TE = ln(1 + F/(n − 3))** ist exakt (bis 1e-16 in jeder Stichprobe); die Vormessung hatte sie wegen eines falschen Teilers (101 statt 100) nur auf 4·10⁻³ bestätigt.
- **Gleichwertige Mutanten:** acht (siehe Verifikation). **Umfang:** die Demo enthält beide Teile (Kointegration und gerichtete Abhängigkeit) und sieben Experimente; `statsmodels` ist **keine** Abhängigkeit (nicht im Test-venv, nicht in `requirements-dev.txt`): die Orakel sind `numpy.linalg.lstsq`, `scipy` und eigene Simulationen.

## Modell

Fehlerkorrektur-Welt (`koi_algorithm.py`, Zufall aus SplitMix64 in wörtlichen Kopien von `pearson-demo`): x ist eine Irrfahrt, y wird von dem Abstand z = y − β x zurückgezogen: Δx = a (1 − s)/β · z_{t−1} + ε_x, Δy = −a s · z_{t−1} + γ ε_x + ε_y mit unabhängigen Standardnormalneuerungen. **Der Abstand folgt einem AR(1) mit Koeffizient 1 − a**, unabhängig davon, wer sich anpasst (s = 1: nur y, 0,5: beide, 0: nur x); seine Varianz ist ((γ − β)² + 1)/(1 − (1 − a)²). a = 0 heißt: nicht kointegriert; mit γ = 0 sind es **unabhängige Irrfahrten**. **Gekoppelte Reihen:** zwei AR(1)-Reihen mit Koeffizient 0,5, y_t = 0,5 y_{t−1} + c x_{t−1} + Neuerung. **Verzögerte gemeinsame Ursache:** eine verborgene AR(1)-Reihe z (0,5) wirkt mit Stärke b und einer Woche Verzögerung auf x und mit zwei Wochen Verzögerung auf y; x wirkt nie auf y.

## Methodik

- **Engle–Granger-Test:** Regression von y auf x mit Konstante, **Dickey–Fuller**-t-Statistik am Rest (ohne verzögerte Differenzen, ohne Konstante); kritische Werte = Quantile der Nullverteilung aus der Simulation (`critical_values`). **Dickey–Fuller je Reihe** (mit Konstante) prüft, ob x und y wandern. **Fehlerkorrektur:** t-Tests des Koeffizienten von û_{t−1} in Δy und in Δx; „passt sich an“ heißt |t| > 1,96. **Naiver Test:** t-Test auf r der Werte (T − 2 Freiheitsgrade); **Test auf die Differenzen:** t-Test auf r der T − 1 Differenzen.
- **Granger-F** für x → y (Verzögerung 1, mit Konstante, n = T − 1 Beobachtungen, F mit (1, n − 3) Freiheitsgraden) und **gaußsche Transferentropie** TE = ½ ln(RSS_eingeschränkt/RSS_voll). **Plug-in-Transferentropie** aus den Häufigkeiten von k gleich besetzten Klassen je Reihe (nach Rängen) mit **Permutationstest** (x mischen, 199 Mischungen) und Abzug des Mittels der Mischungen.
- **Exakte Formeln:** Varianz des Abstands, wahre Anpassungskoeffizienten (−a s, a (1 − s)/β), exakte gaußsche Transferentropie ½ ln(1 + c² (Σ_xx − Σ_xy²/Σ_yy)) aus der Lyapunov-Gleichung, Überschuss der gaußschen Transferentropie ohne Kopplung 1/(2n) und der Plug-in-Schätzung k (k − 1)²/(2n) (Näherung).
- **Experimente:** (1) Teststärke gegen das Anpassungstempo, (2) die Nullverteilung, (3) Differenzieren, (4) wer passt sich an, (5) Granger und Transferentropie, (6) Verzerrung der Transferentropie, (7) verzögerte gemeinsame Ursache.
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/cointegration_grid.json`, 2 s auf 10 Prozessen): 4 Seeds · 4 Pakete · 250 Wiederholungen je Zelle (4 000 Läufe), die Verzerrungszellen (Plug-in) mit 1 000 Läufen; feste Seeds.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, die Tabellen Zeile für Zeile in der Rundung der App, die Hilfetexte der Presets über dieselben Funktionen wie in der App). Mittelwerte und Anteile über alle Läufe einer Zelle.

**Teststärke gegen das Anpassungstempo** (Experiment 1; y passt sich an, β = 1, γ = 0; Anteil der Läufe, in denen der Test eine Beziehung meldet; Engle–Granger mit simuliertem 5-%-Wert; Dickey–Fuller: Anteil, in dem x für stationär erklärt wird, also ein Fehlalarm, denn x bleibt eine Irrfahrt):

| Wochen T | Anpassungstempo a | Autokorrelation des Abstands | Engle–Granger | naiver Test | Dickey–Fuller: x stationär |
|---|---|---|---|---|---|
| 104 | 0,00 | 1,00 | 5,3 % | 76,1 % | 4,8 % |
| 104 | 0,05 | 0,95 | 7,1 % | 83,8 % | 4,8 % |
| 104 | 0,10 | 0,90 | 15,7 % | 94,3 % | 4,8 % |
| 104 | 0,20 | 0,80 | 62,3 % | 99,5 % | 4,8 % |
| 104 | 0,50 | 0,50 | 100,0 % | 100,0 % | 4,8 % |
| 260 | 0,00 | 1,00 | 5,5 % | 84,0 % | 4,9 % |
| 260 | 0,05 | 0,95 | 21,2 % | 98,6 % | 4,9 % |
| 260 | 0,10 | 0,90 | 81,1 % | 100,0 % | 4,9 % |
| 260 | 0,20 | 0,80 | 100,0 % | 100,0 % | 4,9 % |
| 260 | 0,50 | 0,50 | 100,0 % | 100,0 % | 4,9 % |

**Nullverteilung** (Experiment 2; unabhängige Irrfahrten; simulierte Quantile der Engle–Granger-Statistik 1 % / 5 % / 10 %; Fehlalarm mit dem tabellierten Wert −3,34 (aus dem Gedächtnis); simulierte Schwelle dieser Demo und ihr Fehlalarm in frischen Läufen; 5-%-Quantil der Dickey–Fuller-Statistik je Reihe):

| Wochen T | Quantile 1 / 5 / 10 % | Fehlalarm mit −3,34 | simulierter Wert (5 %) | Fehlalarm damit | Dickey–Fuller 5 % |
|---|---|---|---|---|---|
| 26 | -4,232 / -3,582 / -3,217 | 8,0 % | -3,625 | 4,7 % | -3,017 |
| 52 | -4,195 / -3,464 / -3,163 | 6,7 % | -3,503 | 4,7 % | -2,882 |
| 104 | -3,995 / -3,374 / -3,083 | 5,4 % | -3,348 | 5,3 % | -2,885 |
| 260 | -3,993 / -3,396 / -3,085 | 5,7 % | -3,353 | 5,5 % | -2,880 |
| 520 | -3,887 / -3,315 / -3,018 | 4,9 % | -3,365 | 4,6 % | -2,880 |

**Differenzieren** (Experiment 3; T = 104; y passt sich an; Anteil „signifikant“ des Tests auf r der Differenzen und „Beziehung gefunden“ des Engle–Granger-Tests):

| kurzfristige Kopplung γ | Anpassungstempo a | Differenzen „signifikant“ | Engle–Granger | mittleres r der Differenzen |
|---|---|---|---|---|
| 0,0 | 0,0 | 4,9 % | 5,3 % | -0,002 |
| 0,0 | 0,2 | 5,2 % | 62,3 % | -0,010 |
| 0,5 | 0,0 | 99,8 % | 5,3 % | 0,444 |
| 0,5 | 0,2 | 99,4 % | 72,1 % | 0,417 |

**Wer passt sich an?** (Experiment 4; β = 1; α̂ mit wahrem Wert in Klammern):

| T, a | wer passt sich an | α_y signifikant | α_x signifikant | α̂_y (wahr) | α̂_x (wahr) |
|---|---|---|---|---|---|
| 104, 0,2 | nur y passt sich an | 100,0 % | 6,9 % | -0,231 (-0,200) | -0,004 (0,000) |
| 104, 0,2 | beide je zur Hälfte | 88,3 % | 48,4 % | -0,154 (-0,100) | 0,100 (0,100) |
| 104, 0,2 | nur x passt sich an | 17,7 % | 99,3 % | -0,052 (0,000) | 0,224 (0,200) |
| 104, 0,5 | nur y passt sich an | 100,0 % | 5,4 % | -0,522 (-0,500) | -0,005 (0,000) |
| 104, 0,5 | beide je zur Hälfte | 99,8 % | 95,1 % | -0,296 (-0,250) | 0,244 (0,250) |
| 104, 0,5 | nur x passt sich an | 11,5 % | 100,0 % | -0,049 (0,000) | 0,519 (0,500) |
| 260, 0,2 | nur y passt sich an | 100,0 % | 5,8 % | -0,212 (-0,200) | -0,001 (0,000) |
| 260, 0,2 | beide je zur Hälfte | 99,8 % | 93,3 % | -0,121 (-0,100) | 0,100 (0,100) |
| 260, 0,2 | nur x passt sich an | 10,8 % | 100,0 % | -0,021 (0,000) | 0,210 (0,200) |
| 260, 0,5 | nur y passt sich an | 100,0 % | 5,2 % | -0,508 (-0,500) | -0,002 (0,000) |
| 260, 0,5 | beide je zur Hälfte | 100,0 % | 100,0 % | -0,268 (-0,250) | 0,247 (0,250) |
| 260, 0,5 | nur x passt sich an | 8,4 % | 100,0 % | -0,019 (0,000) | 0,507 (0,500) |

**Granger und Transferentropie** (Experiment 5; gekoppelte AR(1)-Reihen, T = 104; Anteil „erkannt“ des F-Tests in beide Richtungen; mittlere gaußsche Transferentropie in nats, exakter Wert in Klammern):

| Kopplung c | x → y erkannt | y → x erkannt | TE x → y (exakt) | TE y → x |
|---|---|---|---|---|
| 0,0 | 5,4 % | 5,2 % | 0,0053 (0,0000) | 0,0052 |
| 0,1 | 21,6 % | 5,1 % | 0,0118 (0,0066) | 0,0052 |
| 0,2 | 62,2 % | 5,3 % | 0,0307 (0,0256) | 0,0052 |
| 0,4 | 99,0 % | 5,8 % | 0,0971 (0,0921) | 0,0052 |

**Verzerrung der Transferentropie ohne Kopplung** (Experiment 6; mittlere Schätzung in nats, Näherung in Klammern):

| Wochen T | gaußsch (1/(2n)) | Plug-in, 3 Klassen (k (k − 1)²/(2n)) | Plug-in, 5 Klassen |
|---|---|---|---|
| 50 | 0,0116 (0,0102) | 0,1552 (0,1224) | 0,6790 (0,8163) |
| 100 | 0,0057 (0,0051) | 0,0711 (0,0606) | 0,4447 (0,4040) |
| 260 | 0,0020 (0,0019) | 0,0241 (0,0232) | 0,1817 (0,1544) |
| 1000 | 0,0005 (0,0005) | 0,0060 (0,0060) | 0,0426 (0,0400) |

**Verzögerte gemeinsame Ursache** (Experiment 7; T = 104; x wirkt nie auf y; Anteil „erkannt“ des Granger-F-Tests):

| Stärke der Ursache | x → y „erkannt“ | y → x „erkannt“ |
|---|---|---|
| 0,0 | 5,1 % | 5,1 % |
| 0,4 | 40,0 % | 6,1 % |
| 0,8 | 99,6 % | 8,4 % |
| 1,2 | 100,0 % | 7,4 % |

Befunde in Worten:

- **Der naive Test meldet auch ohne Beziehung meist „signifikant“ und trennt „keine“ und „echte“ Beziehung deshalb nicht** (ohne Beziehung 76,1 % bei 104 Wochen und 84,0 % bei 260, mit Beziehung 99,5 % bei 104 Wochen). Der Engle–Granger-Test trennt sie (5,3 % gegen 62,3 % bei 104, 100,0 % bei 260 Wochen), **braucht aber Wochen und eine schnelle Anpassung**: bei Tempo 0,05 sind es 7,1 % und 21,2 %, bei 0,1 und 260 Wochen 81,1 %. Dickey–Fuller je Reihe erklärt x in 4,8 % der Läufe fälschlich für stationär (Fehlalarm um 5 %).
- **Die Nullverteilung ist keine t-Verteilung:** die simulierten 5-%-Quantile liegen bei etwa −3,6 (26 Wochen) bis −3,35 (520 Wochen); Ausschläge der Tabelle wie −3,396 bei 260 und −3,315 bei 520 Wochen sind Rauschen von etwa ± 0,04. Mit dem tabellierten Wert −3,34 gäbe es bei 26 Wochen 8,0 % Fehlalarm, bei 520 Wochen 4,9 % (± 0,4 Prozentpunkte Rauschen): deshalb simulierte Schwellen je T.
- **Differenzieren wirft die langfristige Beziehung weg, wenn nur ein Partner sich anpasst:** bei echter Kointegration ohne kurzfristige Kopplung sind die Differenzen in 5,2 % der Läufe „signifikant“ (Engle–Granger: 62,3 %; bei geteilter Anpassung sehen die Differenzen sie teilweise); bei kurzfristiger Kopplung 0,5 sind sie es in 99,8 % der Läufe auch **ohne** jede langfristige Beziehung (mittleres r der Differenzen 0,444). Differenzen beantworten „bewegen sie sich in derselben Woche zusammen“, Engle–Granger „zieht der Abstand sie zurück“; mit beidem (γ = 0,5, a = 0,2) sind es 99,4 % und 72,1 %.
- **Die Fehlerkorrektur zeigt, wer sich anpasst:** passt sich nur y an, ist α_y in 100,0 % und α_x in 6,9 % der Läufe signifikant; passt sich nur x an, α_x in 99,3 % und α_y in 17,7 % (**bei 104 Wochen ist die Richtung unscharf**); bei geteilter Anpassung 88,3 % und 48,4 %. α̂_y liegt bei kurzen Reihen weiter von null weg als die Wahrheit (-0,231 bei wahr −0,200; bei geteilter Anpassung -0,154 bei wahr −0,100), zum Teil, weil der Rest aus denselben Daten geschätzt ist.
- **Granger-F findet die Richtung** (x → y in 62,2 %, y → x in 5,3 % bei Kopplung 0,2; ohne Kopplung 5,4 %, also das Niveau). **Die gaußsche Transferentropie ist dieselbe Zahl:** 2 · TE = ln(1 + F/(n − 3)); ihr Mittel liegt um 1/(2n) = 0,0049 über dem exakten Wert (gemessen 0,0307, exakt 0,0256; ohne Kopplung 0,0053).
- **Der Plug-in-Schätzer ist ohne Kopplung stark verzerrt:** bei 100 Wochen und 3 Klassen im Mittel 0,0711 nats (Näherung 0,0606), die gaußsche Schätzung 0,0057; bei 50 Wochen und 5 Klassen 0,6790. Bei kleinem n braucht die Plug-in-Schätzung eine Korrektur, etwa einen Permutationstest (der hier wenig Macht hat: 11,5 % bei Kopplung 0,2 gegen 62 % beim Granger-F-Test).
- **Granger ist Vorhersagbarkeit, keine Wirkung:** bei einer verzögerten gemeinsamen Ursache der Stärke 0,8 „findet“ der Test x → y in 99,6 % der Läufe und y → x in 8,4 %, obwohl x nie auf y wirkt (ohne Ursache 5,1 %).

## Ehrliche Grenzen

- **Zwei Reihen, ein Abstand.** Mehr als zwei Reihen können mehrere Beziehungen haben; der Johansen-Test ist nicht gebaut.
- **Ein fester Beziehungsvektor, keine Brüche.** Ändert sich β oder bricht die Beziehung, findet der Test sie schlechter oder gar nicht.
- **Verzögerung 1 und keine Modellwahl.** Echte Reihen brauchen mehrere Verzögerungen und eine Auswahl nach Informationskriterien; Dickey–Fuller und Granger laufen hier ohne verzögerte Differenzen beziehungsweise mit genau einer Verzögerung.
- **Normalverteilte Neuerungen.** Schwere Ränder und Ausreißer verschieben die Schwellen; die simulierten Schwellen gelten für genau diese Welt.
- **Granger und Transferentropie sind Vorhersagbarkeit.** Eine verborgene oder verzögerte Ursache erzeugt eine Richtung ohne Wirkung (Experiment 7); Ursachen gehören der Linie *Kausale Diagramme und Adjustierung*.
- **Plug-in mit Klassen.** Stark verzerrt bei kleinem n; bedingte und längere Transferentropie, andere Schätzer (Nachbarn, Kerne) und Frequenzbereichs-Granger sind nicht gebaut. Die Näherung k (k − 1)²/(2n) ist bei k = 3 und bei k = 5 ab 100 Wochen zu klein (um bis zu etwa ein Fünftel des gemessenen Werts) und nur bei k = 5 und 50 Wochen zu groß (viele leere Zellen).
- **Kritische Werte aus 5 000 Läufen.** Sie haben selbst einen Fehler von etwa ±0,7 Prozentpunkten Fehlalarm.
- **Quellen:** keine Literaturangaben; die Formeln sind durch unabhängige Rechenwege (unten) und die Messungen gestützt.

## Verifikation

- **Orakel** (`tests/test_oracle_kointegration.py`): Regression gegen **`numpy.linalg.lstsq`**; die Dickey–Fuller- und die Fehlerkorrektur-t-Statistik gegen **Kleinste-Quadrate-Rechnungen mit eigenem Standardfehler**; die simulierten kritischen Werte bei 520 Wochen gegen die **asymptotischen Werte der Engle–Granger-Tabelle** (aus dem Gedächtnis, Toleranz 0,08) und der Fehlalarm in frischen Läufen gegen 5 %; die naiven Tests gegen **`scipy.stats.pearsonr`**; Granger-F gegen zwei **`lstsq`**-Anpassungen und **`scipy.stats.f`**; die exakte Transferentropie gegen **`scipy.linalg.solve_discrete_lyapunov`** und gegen eine **Simulation mit 40 · 30 000 Wochen**; das Mittel der Stichprobenschätzung gegen exakt + 1/(2n); die Plug-in-Transferentropie gegen eine **Schleife über Entropien** (`scipy.stats.entropy`) und gegen das Handbeispiel ln 2; der Permutationstest hält sein Niveau.
- **Kern** (`tests/test_algorithm.py`, `tests/test_edge_cases.py`): Handrechnungen (Mini-Instanzen in `conftest.py`: vier Wochen mit Steigung 1,9 und Dickey–Fuller-t = −2,8831, Abstandsvarianz 5,5556, zwei Residuensummen 100 und 90 gegen F = 5,556 und 2 · TE = ln(100/90), de-Bruijn-Folge mit Transferentropie ln 2), **inklusive Grenzen** (das 5-%-Niveau der naiven Tests bei T = 10: r = 0,640 signifikant, r = 0,625 nicht; die F-Schwelle strikt; Gleichstände im Permutationstest), kürzeste und längste Reihen aller Welten.
- **Eigenschaften** (`tests/test_eigenschaften_kointegration.py`): Engle–Granger-t invariant gegen Verschieben und Strecken beider Reihen, Asymmetrie der Regression, Granger-F und Transferentropie invariant, Plug-in-TE invariant gegen monotone Umformung, Monotonie der exakten Formeln und der Teststärke gegen das Tempo.
- **Raster und Aussagen** (`tests/test_grid.py`, `tests/test_claims.py`): Probe des Erzeugers gegen die gespeicherte, jede Zahl dieser README gegen das Raster oder die Live-Funktion, die Tabellen in der Rundung der App.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an seinen Grenzen), alle Presets, Adresszeile, jedes Auswahlfeld der Experimente, PDF in allen Welten, Layout, Footer, Laufzeit-Importe.
- **Mutationsprüfung:** `tools/mutate_demo.py` des Website-Repos mutiert den Kern `koi_algorithm` (die wörtlichen Kopien `prs_*` sind ausgenommen, sie werden in `pearson-demo` geprüft; Auswertung, Verdikte und Abbildungen prüfen die übrigen Tests): 151 Stellen; 138 durch eigene Tests gefangen, 5 durch wortgleiche Zeilen in Nachbar-Demos gesichert und 8 als gleichwertig festgehalten: die Klassen nach Rängen werden nie durch das Minimum begrenzt (zwei Mutanten), die Gleichheit mit der Schwelle ist bei stetigen Statistiken nicht erreichbar (vier Vergleiche in den Tests und der Bisektion), der p-Wert bei F = 0 ist ohnehin 1, und die letzte Woche der verborgenen Reihe geht nie in x oder y ein. Der erste Lauf fand 28 Überlebende; neue Tests schlossen die Lücken: die Schwelle des naiven Tests und des Tests auf die Differenzen (T − 2 Freiheitsgrade, Paare mit r = 0,640 und 0,625 um das 5-%-Niveau bei T = 10), die Freiheitsgrade der Fehlerkorrektur-t-Statistik und die Konstanten und Gleichstände des Permutationstests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `koi_algorithm.py` | Kern: Welten, Regression, Dickey–Fuller, Engle–Granger, kritische Werte, Fehlerkorrektur, Granger, Transferentropie, exakte Formeln |
| `koi_constants.py` | Regler-Grenzen, Welten, Raster der Messreihen, Farben, Presets |
| `koi_evaluation.py` | Einzelstichprobe mit ihren Zeilen, Zellen des Rasters, Verdikte |
| `koi_layout.py` | Fester Kopf (Stylesheet) |
| `koi_pdf_export.py` | PDF-Export der Hauptansicht |
| `koi_presets.py` | Permalink, Presets (Schnellstart mit Hervorhebung), Zufalls-Seed-Button |
| `koi_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `app.py` | Streamlit-App |
| `data/cointegration_grid.json` | vorgerechnetes Raster (erzeugt von `tools/make_grid.py`) |
| `prs_*.py` | wörtliche Kopien aus `pearson-demo` (SplitMix64, Szenario, Student- und Betaverteilung, Konstanten) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy für die Orakel |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel, Kern, Eigenschaften, Raster, Aussagen, Verdikte, Oberfläche, Layout, Footer, Laufzeit-Importe, Kopie-Test, Mutationszustand |
| `tools/make_grid.py` | rechnet das Raster vor |

## Bewusst nicht umgesetzt

- **Johansen-Test, mehr als zwei Reihen, mehrere Verzögerungen, Informationskriterien** (Modellwahl), **strukturelle Brüche**, **nichtlineare Kointegration**.
- **Bedingte und längere Transferentropie**, andere Schätzer (Nachbarn, Kerne), **Frequenzbereichs-Granger**.
- **Ursachen aus Daten**: Linie *Kausale Diagramme und Adjustierung*.
- **`statsmodels`** (`adfuller`, `coint`) als zweiter Rechenweg: nicht installiert, nicht in den Abhängigkeiten; die Orakel sind `lstsq`, `scipy` und eigene Simulationen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Raster entsteht neu mit `python tools/make_grid.py` (2 s auf zehn Prozessen); `tests/test_grid.py` rechnet eine feste Probe nach und schlägt an, wenn Erzeuger oder Statistik das Raster überholt haben.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
