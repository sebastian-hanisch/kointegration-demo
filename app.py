"""Gleichlauf, der hält: Kointegration und gerichtete Abhängigkeit - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes Stück der Korrelations-Reihe. Zwei Wochenreihen in drei Welten mit bekannter Wahrheit: Fehlerkorrektur (Irrfahrten mit gemeinsamem Abstand), gekoppelte stationäre Reihen und eine verzögerte
gemeinsame Ursache. Engle-Granger-Test mit simulierten kritischen Werten, Fehlerkorrektur (wer passt sich an), Test auf die Differenzen, Granger-F und Transferentropie (gaußsch, Plug-in).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import koi_constants as C
import koi_evaluation as ev
from koi_layout import dunkles_design, kopf_css
from koi_pdf_export import build_pdf
from koi_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from koi_visualization import (
    build_e1,
    build_e2,
    build_e3,
    build_e4,
    build_e5,
    build_e6,
    build_e7,
    build_null,
    build_series,
)

st.set_page_config(page_title="Kointegration und gerichtete Abhängigkeit – Sebastian Hanisch", layout="wide")

de = ev.de
pct = ev.pct


def n_text(v: float) -> str:
    return f"{v:g}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in koi_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔗 Gleichlauf, der hält: Kointegration und gerichtete Abhängigkeit")
st.markdown(
    """
**Elftes Stück der Korrelations-Reihe** und **die erste Demo des Portfolios über Kointegration und gerichtete Abhängigkeit**. Zwei Wochenreihen **x und y** wandern, und in Stück 4 war jede „Korrelation“ zwischen
unabhängigen Wanderern ein Fehlalarm. Aber manche Paare wandern **zusammen**, weil ein Abstand sie zurückzieht: das ist echt. Die Fragen: **Wie unterscheidet man echten von scheinbarem Gleichlauf? Wer passt sich
wem an? Und was sagen Granger-Test und Transferentropie über die Richtung, und was nicht?**

Weiter unten: sieben Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Ursachen gehören der Linie *Kausale Diagramme und Adjustierung*: Granger und
Transferentropie messen **Vorhersagbarkeit**, keine Wirkung.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        r"""
| Begriff | Was er hier bedeutet |
|---|---|
| **Irrfahrt, Einheitswurzel** | Eine Reihe, die sich von selbst nirgends hinzieht: jede Woche kommt eine Neuerung dazu. Der **Dickey–Fuller-Test** fragt, ob sie doch zurückgezogen wird (stationär). |
| **Kointegration, Abstand** | Zwei Irrfahrten x und y sind kointegriert, wenn der **Abstand** z = y − β x stationär ist: sie wandern zusammen. Das **Anpassungstempo** a sagt, wie schnell der Abstand zurückgezogen wird (a = 0: unabhängig). |
| **Engle–Granger-Test** | Regression von y auf x, dann Dickey–Fuller-Test am Rest. Die **kritischen Werte** kommen aus einer Simulation unabhängiger Irrfahrten mit genau dieser Wochenzahl: die t-Verteilung gilt hier nicht. |
| **Fehlerkorrektur** | Wer zieht den Abstand zurück? Regression von Δy und Δx auf den Rest der Vorwoche: y, x oder beide. Der Partner, der sich nicht anpasst, führt. |
| **Test auf die Differenzen** | Das Gegenmittel aus Stück 4: er sieht die Bewegung **derselben Woche** und verliert die langfristige Beziehung. |
| **Granger-F** | Hilft die Vergangenheit von x, y vorherzusagen (über die von y hinaus)? Vorhersagbarkeit, keine Wirkung. |
| **Transferentropie** | Dieselbe Frage als Informationsmaß in nats. Im Gauß-Fall gilt 2 · TE = ln(1 + F/(n − 3)); der **Plug-in-Schätzer** mit Klassen ist bei kleinem n stark verzerrt und braucht einen **Permutationstest**. |
| **Wahrheit** | Zu jeder Prüfung steht, was in dieser Welt wirklich gilt (Anpassungstempo, Kopplung), daneben die exakte Formel, wo es eine gibt. |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt**")
    world = st.selectbox("Welt", list(C.WORLDS), format_func=C.WORLDS.get, key="world_select", help="Fehlerkorrektur: zwei Irrfahrten mit gemeinsamem Abstand. Gekoppelt: x wirkt auf y. Verzögerte Ursache: eine verborgene Reihe, x wirkt nie auf y.")
    t = st.select_slider("Wochen T", options=C.T_OPTIONS, key="t_select", help="Länge der Reihen. Die kritischen Werte werden für genau dieses T simuliert.")
    st.markdown("**Fehlerkorrektur**")
    a_ = st.slider("Anpassungstempo a", *bounds("a_slider"), step=0.05, key="a_slider", format="%.2f", help="0 = unabhängige Irrfahrten; je größer, desto schneller wird der Abstand zurückgezogen (Autokorrelation 1 − a).")
    share = st.select_slider("Wer passt sich an", options=list(C.SHARES), format_func=C.SHARES.get, key="share_select", help="y, beide je zur Hälfte oder x zieht den Abstand zurück.")
    beta = st.select_slider("Faktor β des Abstands", options=C.BETA_OPTIONS, key="beta_select", help="Abstand z = y − β x; die Regression schätzt β.")
    gamma = st.slider("Kurzfristige Kopplung γ", *bounds("gamma_slider"), step=0.1, key="gamma_slider", format="%.1f", help="Anteil der x-Neuerung, der in derselben Woche auch y bewegt.")
    st.markdown("**Gerichtete Abhängigkeit**")
    coupling = st.slider("Kopplung c (x → y)", *bounds("coupling_slider"), step=0.1, key="coupling_slider", format="%.1f", help="Wirkung von x der Vorwoche auf y; nur in der Welt „Gekoppelte Reihen“.")
    driver = st.slider("Stärke der verzögerten Ursache", *bounds("driver_slider"), step=0.2, key="driver_slider", format="%.1f", help="Wirkung der verborgenen Reihe auf x (eine Woche Verzögerung) und y (zwei Wochen); nur in der Welt „Verzögerte Ursache“.")
    classes = st.select_slider("Klassen des Plug-in-Schätzers", options=C.CLASS_OPTIONS, key="classes_select", help="Zahl der gleich besetzten Klassen je Reihe für die Plug-in-Transferentropie.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Daten ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"world_select": world, "t_select": int(t), "a_slider": float(a_), "share_select": float(share), "beta_select": float(beta), "gamma_slider": float(gamma),
                   "coupling_slider": float(coupling), "driver_slider": float(driver), "classes_select": int(classes), "seed_input": int(seed)})

settings = ev.Settings(world=world, t=int(t), a=float(a_), share=float(share), beta=float(beta), gamma=float(gamma), coupling=float(coupling), driver=float(driver), classes=int(classes), seed=int(seed))
with st.spinner("Rechne die kritischen Werte (Nullsimulation) und die Prüfungen …"):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
cols = st.columns(2)
if world == "ecm":
    eg = a.row("eg")
    cols[0].metric("Naiver Test auf r", de(a.row("naive").value), help=f"{a.row('naive').decision}; er ist bei Irrfahrten fast immer „signifikant“.")
    cols[1].metric("Engle–Granger t", de(eg.value), help=f"Schwelle {de(eg.threshold)}: {eg.decision}. Wahrheit: {eg.truth}.")
    cols = st.columns(2)
    cols[0].metric("α̂_y (y passt sich an)", de(a.row("alpha_y").value), help=f"{a.row('alpha_y').decision}; wahr {a.row('alpha_y').truth}.")
    cols[1].metric("α̂_x (x passt sich an)", de(a.row("alpha_x").value), help=f"{a.row('alpha_x').decision}; wahr {a.row('alpha_x').truth}.")
else:
    cols[0].metric("Granger-F x → y", de(a.row("granger_xy").value), help=f"Schwelle {de(a.row('granger_xy').threshold)}: {a.row('granger_xy').decision}.")
    cols[1].metric("Granger-F y → x", de(a.row("granger_yx").value), help=f"Schwelle {de(a.row('granger_yx').threshold)}: {a.row('granger_yx').decision}.")
    cols = st.columns(2)
    cols[0].metric("Transferentropie x → y (nats)", de(a.row("te_xy").value, 4), help=a.row("te_xy").truth)
    cols[1].metric("Plug-in x → y (nats)", de(a.row("plugin_xy").value, 4), help=a.row("plugin_xy").decision)
st.plotly_chart(build_series(a), width="stretch", key="main_series")
if world == "ecm":
    st.plotly_chart(build_null(a), width="stretch", key="main_null")
    st.caption("Nullverteilung der Engle–Granger-Statistik bei unabhängigen Irrfahrten mit genau diesen T Wochen (grau), ihre 1-, 5- und 10-%-Quantile (gestrichelt) und die beobachtete Statistik (rot).")
rows = [{"Prüfung": r.name, "Wert": de(r.value, r.digits), "Schwelle": de(r.threshold, 2), "Entscheidung": r.decision, "Wahrheit": r.truth} for r in a.rows]
st.dataframe(rows, hide_index=True, width="stretch")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)
extra = (ev.verdict_adapt, ev.verdict_difference) if world == "ecm" else (ev.verdict_te,)
for verdict in extra:
    kind, text = verdict(a)
    getattr(st, kind)(text)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="kointegration-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Wann ist der Gleichlauf echt, und wer führt wen?")
st.markdown("**Nicht der naive Test, und nicht die Differenzen:** der Engle–Granger-Test unterscheidet echten von scheinbarem Gleichlauf, braucht aber Wochen und eigene Schwellen; die Fehlerkorrektur zeigt, wer sich "
            "anpasst. Granger und Transferentropie zeigen Vorhersagbarkeit in eine Richtung, keine Wirkung (Experimente unten).")

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/cointegration_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`. Die Experimente zeigen es.")
runs_text = f"{len(C.GRID_SEEDS) * C.BATCHES * C.BATCH:,}".replace(",", " ")

# --- E1 Teststärke --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wann sieht man die Beziehung?")
if grid is not None:
    st.caption(f"Fehlerkorrektur mit y passt sich an, β = 1, γ = 0; {runs_text} Läufe je Zelle. Engle–Granger mit dem simulierten Wert des 5-%-Niveaus; der naive Test ist der t-Test auf r der Werte.")
    st.plotly_chart(build_e1(grid["power"]), width="stretch", key="e1_chart")
    st.dataframe([{"Wochen T": c["T"], "Anpassungstempo a": de(c["a"]), "Engle–Granger": pct(c["eg"]), "naiver Test": pct(c["naive"]), "Dickey–Fuller: x stationär": pct(c["df_x"])}
                  for c in sorted(grid["power"], key=lambda c: (c["T"], c["a"]))], hide_index=True, width="stretch")
    e1_t = st.selectbox("Wochen T für das Verdikt", list(C.E1_TS), key="e1_t")
    kind, text = ev.verdict_e1(grid["power"], e1_t)
    getattr(st, kind)(text)
    st.caption("Der naive Test sagt bei keiner Beziehung und bei echter Beziehung „signifikant“. Der Engle–Granger-Test trennt beides, braucht aber Zeit: bei langsamer Anpassung bleibt er lange blind.")

st.markdown("---")

# --- E2 Nullverteilung ----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Die Nullverteilung ist keine t-Verteilung")
if grid is not None:
    st.caption(f"Unabhängige Irrfahrten, {runs_text} Läufe je Länge. Der tabellierte Wert −3,34 ist der asymptotische Wert des 5-%-Niveaus für zwei Reihen mit Konstante (aus dem Gedächtnis, hier nur als Gegenprobe).")
    st.plotly_chart(build_e2(grid["null"]), width="stretch", key="e2_chart")
    st.dataframe([{"Wochen T": c["T"], "Quantil 1 %": de(c["q"][0]), "Quantil 5 %": de(c["q"][1]), "Quantil 10 %": de(c["q"][2]), "Dickey–Fuller je Reihe, 5 %": de(c["df_q05"]),
                   "Fehlalarm mit −3,34": pct(c["alarm_table"]), "Fehlalarm mit simuliertem Wert": pct(c["alarm_live"])} for c in sorted(grid["null"], key=lambda c: c["T"])], hide_index=True, width="stretch")
    e2_t = st.selectbox("Wochen T für das Verdikt", list(C.E2_TS), index=2, key="e2_t")
    kind, text = ev.verdict_e2(grid["null"], e2_t)
    getattr(st, kind)(text)
    st.caption("Bei kurzen Reihen ist die Schwelle strenger als die Tabelle: wer den tabellierten Wert nimmt, hat zu viele Fehlalarme. Deshalb simuliert die Demo die Schwelle für genau das gewählte T.")

st.markdown("---")

# --- E3 Differenzieren ----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Differenzieren verliert die Beziehung")
if grid is not None:
    st.caption(f"T = {C.E3_T}, Anpassungstempo 0 oder 0,2, kurzfristige Kopplung γ 0 oder 0,5; {runs_text} Läufe je Zelle.")
    st.plotly_chart(build_e3(grid["diff"]), width="stretch", key="e3_chart")
    e3c1, e3c2 = st.columns(2)
    e3_gamma = e3c1.selectbox("Kurzfristige Kopplung γ", list(C.E3_GAMMAS), key="e3_gamma", format_func=n_text)
    e3_a = e3c2.selectbox("Anpassungstempo a", list(C.E3_AS), index=1, key="e3_a", format_func=n_text)
    kind, text = ev.verdict_e3(grid["diff"], e3_gamma, e3_a)
    getattr(st, kind)(text)
    st.caption("Differenzen beantworten „bewegen sie sich in derselben Woche zusammen?“, Engle–Granger „zieht der Abstand sie zurück?“. Das Mittel gegen Scheinkorrelation aus Stück 4 wirft die langfristige Beziehung weg.")

st.markdown("---")

# --- E4 Wer passt sich an -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Wer passt sich an?")
if grid is not None:
    e4c1, e4c2, e4c3 = st.columns(3)
    e4_t = e4c1.selectbox("Wochen T", list(C.E4_TS), key="e4_t")
    e4_a = e4c2.selectbox("Anpassungstempo a", list(C.E4_AS), key="e4_a", format_func=n_text)
    e4_share = e4c3.selectbox("Wer passt sich an (Verdikt)", list(C.SHARES), format_func=C.SHARES.get, key="e4_share")
    st.caption(f"Test von û der Vorwoche in Δy und in Δx, β = 1; {runs_text} Läufe je Zelle; „signifikant“ heißt |t| > 1,96.")
    st.plotly_chart(build_e4(grid["adapt"], e4_t, e4_a), width="stretch", key="e4_chart")
    kind, text = ev.verdict_e4(ev.find(grid["adapt"], T=e4_t, a=e4_a, share=e4_share))
    getattr(st, kind)(text)
    st.caption("Der führende Partner bleibt bei rund 5 % „signifikant“; die Schätzer der Anpassung liegen bei kurzen Reihen weiter von null weg als die Wahrheit, weil der Rest aus denselben Daten geschätzt ist.")

st.markdown("---")

# --- E5 Granger und Transferentropie --------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Granger und Transferentropie")
if grid is not None:
    st.caption(f"Zwei stationäre AR(1)-Reihen (0,5), x wirkt mit Kopplung c auf y, T = {C.E5_T}; {runs_text} Läufe je Zelle.")
    st.plotly_chart(build_e5(grid["granger"]), width="stretch", key="e5_chart")
    e5_c = st.selectbox("Kopplung c", list(C.E5_COUPLINGS), index=2, key="e5_c", format_func=n_text)
    kind, text = ev.verdict_e5(ev.find(grid["granger"], c=e5_c))
    getattr(st, kind)(text)
    st.caption("Der F-Test findet die Richtung und hält in der Gegenrichtung sein Niveau. Die gaußsche Transferentropie ist dieselbe Zahl in anderer Einheit; im Mittel liegt sie um 1/(2n) über dem exakten Wert.")

st.markdown("---")

# --- E6 Verzerrung der Transferentropie -------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Die Verzerrung der Transferentropie")
if grid is not None:
    st.caption(f"Ohne jede Kopplung (c = 0), {C.E6_BATCH * len(C.GRID_SEEDS):,} Läufe je Zelle; Näherungen: 1/(2n) und k (k − 1)²/(2n).".replace(",", " "))
    st.plotly_chart(build_e6(grid["bias"]), width="stretch", key="e6_chart")
    e6_k = st.selectbox("Klassen k für das Verdikt", list(C.E6_KS), key="e6_k")
    kind, text = ev.verdict_e6(grid["bias"], e6_k)
    getattr(st, kind)(text)
    st.caption("Der Plug-in-Schätzer meldet bei kleinem n Information, wo keine ist: größer als die exakte Transferentropie einer echten, mittleren Kopplung. Jede Plug-in-Schätzung braucht deshalb einen Permutationstest.")

st.markdown("---")

# --- E7 Verzögerte gemeinsame Ursache ---------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 7: Eine Richtung ohne Wirkung")
if grid is not None:
    st.caption(f"Eine verborgene AR(1)-Reihe wirkt auf x mit einer und auf y mit zwei Wochen Verzögerung; x wirkt nie auf y; T = {C.E7_T}; {runs_text} Läufe je Zelle.")
    st.plotly_chart(build_e7(grid["lagged"]), width="stretch", key="e7_chart")
    e7_b = st.selectbox("Stärke der Ursache", [c["b"] for c in sorted(grid["lagged"], key=lambda c: c["b"])], index=2, key="e7_b", format_func=n_text)
    kind, text = ev.verdict_e7(ev.find(grid["lagged"], b=e7_b))
    getattr(st, kind)(text)
    st.caption("x der Vorwoche enthält die verborgene Ursache von vor zwei Wochen und hilft deshalb, y vorherzusagen: Granger findet eine Richtung, wo es keine Wirkung gibt.")

st.markdown("---")

# --- Grenzen ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    r"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Zwei Reihen, ein Abstand** | mehr als zwei Reihen können mehrere Beziehungen haben; der Johansen-Test ist nicht gebaut | – |
| **Ein fester Beziehungsvektor, keine Brüche** | ändert sich β oder bricht die Beziehung, findet der Test sie schlechter oder gar nicht | – |
| **Verzögerung 1, keine Modellwahl** | echte Reihen brauchen mehrere Verzögerungen und eine Auswahl nach Informationskriterien | – |
| **Normalverteilte Neuerungen** | schwere Ränder und Ausreißer verschieben die Schwellen; die Schwellen gelten für genau diese Welt | – |
| **Granger und Transferentropie sind Vorhersagbarkeit** | eine verborgene oder verzögerte Ursache erzeugt eine Richtung ohne Wirkung (Experiment 7) | Linie *Kausale Diagramme und Adjustierung* |
| **Plug-in mit Klassen** | stark verzerrt bei kleinem n; bedingte und längere Transferentropie sind nicht gebaut | – |
| **Kritische Werte aus 5 000 Läufen** | sie haben selbst einen Fehler von etwa ±0,7 Prozentpunkten Fehlalarm | – |
"""
)
st.caption("Alle Daten sind simuliert; die Welten sind eine Modellwahl, kein Befund über echte Preise oder Bestände.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Zwei Reihen $x_t, y_t$ mit dem **Abstand** $z_t = y_t - \beta x_t$. In der Fehlerkorrektur-Welt gilt mit dem Anpassungstempo $a$, dem Anteil $s$, mit dem sich $y$ anpasst, und der
kurzfristigen Kopplung $\gamma$

$$ \Delta x_t = \frac{a(1-s)}{\beta}\, z_{t-1} + \varepsilon^x_t, \qquad \Delta y_t = -a s\, z_{t-1} + \gamma\,\varepsilon^x_t + \varepsilon^y_t $$

| Symbol | Bedeutung |
|---|---|
| $a$ | Anpassungstempo; $a = 0$ sind unabhängige Irrfahrten |
| $\hat u_t$ | Rest der Regression von $y$ auf $x$ |
| $\gamma_{DF}$ | Koeffizient im Dickey–Fuller-Test |

Der Abstand folgt einem AR(1) mit Koeffizient $1-a$ und hat die Varianz

$$ \operatorname{Var}(z) = \frac{(\gamma-\beta)^2 + 1}{1-(1-a)^2} $$

**Engle–Granger-Test:** Regression $y_t = c + b\,x_t + u_t$, danach die t-Statistik von $\gamma_{DF}$ in

$$ \Delta \hat u_t = \gamma_{DF}\,\hat u_{t-1} + e_t $$

mit **simulierten** kritischen Werten (Quantile der Nullverteilung unabhängiger Irrfahrten). **Fehlerkorrektur:** t-Tests der Koeffizienten von $\hat u_{t-1}$ in $\Delta y_t$ und $\Delta x_t$.

**Granger-F** (Verzögerung 1, $n$ Beobachtungen) und gaußsche **Transferentropie**:

$$ F = \frac{\mathrm{RSS}_r - \mathrm{RSS}_f}{\mathrm{RSS}_f/(n-3)}, \qquad \mathrm{TE}_{x\to y} = \tfrac12 \ln\frac{\mathrm{RSS}_r}{\mathrm{RSS}_f} = \tfrac12 \ln\!\Big(1 + \frac{F}{n-3}\Big) $$

**Exakte Transferentropie** der gekoppelten AR(1)-Reihen ($\Sigma$ aus der Lyapunov-Gleichung $\Sigma = A\Sigma A^\top + I$, $A = \begin{pmatrix}\varphi & 0\\ c & \varphi\end{pmatrix}$):

$$ \mathrm{TE}_{x\to y} = \tfrac12 \ln\!\Big(1 + c^2\big(\Sigma_{xx} - \Sigma_{xy}^2/\Sigma_{yy}\big)\Big) $$

**Plug-in-Transferentropie** mit $k$ Klassen je Reihe aus den Häufigkeiten $\hat p$:

$$ \widehat{\mathrm{TE}} = \sum \hat p(y',y,x)\,\ln\frac{\hat p(y' \mid y,x)}{\hat p(y' \mid y)}, \qquad \mathbb{E}\,\widehat{\mathrm{TE}} \approx \frac{k(k-1)^2}{2n}\ \text{ohne Kopplung} $$

**Quellen.** In dieser Fassung sind keine Literaturangaben ausgewiesen; die Formeln sind durch unabhängige Rechenwege (Tests gegen numpy, scipy, Schleifenfassungen und Simulationen) und die Messungen der Experimente gestützt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
