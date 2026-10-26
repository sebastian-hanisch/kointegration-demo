"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import koi_algorithm as K
import koi_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _fmt(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def build_series(a, height: int = 420):
    """Die Reihen der Stichprobe; in der Fehlerkorrektur-Welt darunter der Abstand (Rest der Engle–Granger-Regression) mit seinem exakten ±1,96-Streuungsband."""
    s = a.settings
    weeks = list(range(1, len(a.x) + 1))
    rows = 2 if a.spread is not None else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Die beiden Reihen", "Abstand y − β̂ x (Rest der Regression)") if rows == 2 else ("Die beiden Reihen",))
    fig.add_trace(go.Scatter(x=weeks, y=a.x.tolist(), mode="lines", name="x", line=dict(color=C.COLOR_X)), row=1, col=1)
    fig.add_trace(go.Scatter(x=weeks, y=a.y.tolist(), mode="lines", name="y", line=dict(color=C.COLOR_Y)), row=1, col=1)
    if rows == 2:
        fig.add_trace(go.Scatter(x=weeks, y=a.spread.tolist(), mode="lines", name="Abstand", line=dict(color=C.COLOR_Z)), row=2, col=1)
        fig.add_hline(y=0.0, line=dict(color=C.COLOR_DARK, dash="dot", width=1), row=2, col=1)
        var = K.exact_spread_variance(s.a, s.beta, s.gamma)
        if math.isfinite(var):
            band = 1.96 * math.sqrt(var)
            for sign in (-1, 1):
                fig.add_hline(y=sign * band, line=dict(color=C.COLOR_LIMIT, dash="dash", width=1), row=2, col=1)
    fig.update_xaxes(title_text="Woche", row=rows, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="top", y=-0.18), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_annotations(font_size=12)
    return lock_axes(fig)


def build_null(a, height: int = 320):
    """Nullverteilung der Engle–Granger-Statistik für genau dieses T (Simulation unabhängiger Irrfahrten) mit den kritischen Werten und dem beobachteten Wert."""
    s = a.settings
    eg, _df = K.null_statistics(s.t)
    cv = a.cv["eg"]
    fig = go.Figure(go.Histogram(x=eg.tolist(), nbinsx=60, marker_color=C.COLOR_GREY, opacity=0.7, name="unabhängige Irrfahrten"))
    for q, label in zip(cv, ("1 %", "5 %", "10 %")):
        fig.add_vline(x=q, line=dict(color=C.COLOR_LIMIT, dash="dash", width=2 if label == "5 %" else 1), annotation_text=label, annotation_position="top")
    fig.add_vline(x=a.extra["t_eg"], line=dict(color=C.COLOR_Y, width=3), annotation_text="beobachtet", annotation_position="bottom")
    fig.update_xaxes(title_text="Engle–Granger-t-Statistik")
    fig.update_yaxes(title_text="Läufe")
    return _base(fig, height)


def _bars(cells, keys, names, colors, x_key, ytitle, height=340, fmt=lambda v: f"{100 * v:.1f} %", scale=100.0):
    fig = go.Figure()
    for key, name, color in zip(keys, names, colors):
        vals = [scale * c[key] for c in cells]
        fig.add_trace(go.Bar(x=[x_key(c) for c in cells], y=vals, name=name, marker_color=color, text=[fmt(c[key]) for c in cells], textposition="outside"))
    fig.update_yaxes(title_text=ytitle, range=[0, 118])
    fig.update_layout(barmode="group")
    return _base(fig, height)


def build_e1(cells: list, height: int = 360):
    """E1: Anteil „Beziehung gefunden“ (Engle–Granger) und „signifikant“ (naiver Test) gegen das Anpassungstempo, für T = 104 und 260."""
    fig = go.Figure()
    for T, dash in ((104, "solid"), (260, "dot")):
        mine = sorted((c for c in cells if c["T"] == T), key=lambda c: c["a"])
        xs = [c["a"] for c in mine]
        fig.add_trace(go.Scatter(x=xs, y=[100 * c["eg"] for c in mine], mode="lines+markers", name=f"Engle–Granger, T = {T}", line=dict(color=C.COLOR_Z, dash=dash)))
        fig.add_trace(go.Scatter(x=xs, y=[100 * c["naive"] for c in mine], mode="lines+markers", name=f"naiver Test, T = {T}", line=dict(color=C.COLOR_Y, dash=dash)))
    fig.add_hline(y=5, line=dict(color=C.COLOR_DARK, dash="dot", width=1), annotation_text="5 %", annotation_position="bottom right")
    fig.update_xaxes(title_text="Anpassungstempo a (0 = unabhängige Irrfahrten)")
    fig.update_yaxes(title_text="„Beziehung“ gemeldet (%)", range=[0, 105])
    return _base(fig, height)


def build_e2(cells: list, height: int = 340):
    """E2: simulierte 5-%-Quantile der Engle–Granger-Statistik gegen T mit dem asymptotischen Wert der Tabelle (−3,34) als Linie."""
    mine = sorted(cells, key=lambda c: c["T"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[c["T"] for c in mine], y=[c["q"][1] for c in mine], mode="lines+markers", name="simuliertes 5-%-Quantil", line=dict(color=C.COLOR_Z)))
    fig.add_trace(go.Scatter(x=[c["T"] for c in mine], y=[c["q"][0] for c in mine], mode="lines+markers", name="simuliertes 1-%-Quantil", line=dict(color=C.COLOR_Z, dash="dot")))
    fig.add_hline(y=-3.3377, line=dict(color=C.COLOR_LIMIT, dash="dash"), annotation_text="Tabellenwert 5 % (asymptotisch)", annotation_position="top right")
    fig.update_xaxes(title_text="Wochen T", type="log", tickvals=[c["T"] for c in mine], ticktext=[str(c["T"]) for c in mine])
    fig.update_yaxes(title_text="Schwelle der t-Statistik")
    return _base(fig, height)


def build_e3(cells: list, height: int = 340):
    """E3: Anteil „signifikant“ des Tests auf die Differenzen und „Beziehung gefunden“ des Engle–Granger-Tests je Kombination aus kurzfristiger Kopplung und Anpassungstempo."""
    order = sorted(cells, key=lambda c: (c["gamma"], c["a"]))
    return _bars(order, ("diff", "eg"), ("Test auf die Differenzen", "Engle–Granger"), (C.COLOR_Y, C.COLOR_Z), lambda c: f"γ = {_fmt(c['gamma'], 1)}, a = {_fmt(c['a'], 1)}", "Anteil der Läufe (%)", height)


def build_e4(cells: list, T: int, a: float, height: int = 340):
    """E4: Anteil signifikanter Anpassung von y und von x je Anpassungsanteil (für ein T und ein Tempo)."""
    mine = [c for c in sorted(cells, key=lambda c: -c["share"]) if c["T"] == T and c["a"] == a]
    return _bars(mine, ("sig_y", "sig_x"), ("α_y signifikant (y passt sich an)", "α_x signifikant (x passt sich an)"), (C.COLOR_Y, C.COLOR_X), lambda c: C.SHARES[c["share"]], "Anteil der Läufe (%)", height)


def build_e5(cells: list, height: int = 340):
    """E5: Anteil „erkannt“ des Granger-F-Tests in beide Richtungen je Kopplung."""
    mine = sorted(cells, key=lambda c: c["c"])
    fig = _bars(mine, ("xy", "yx"), ("x → y erkannt", "y → x erkannt"), (C.COLOR_Y, C.COLOR_X), lambda c: f"c = {_fmt(c['c'], 1)}", "Anteil der Läufe (%)", height)
    fig.add_hline(y=5, line=dict(color=C.COLOR_DARK, dash="dot", width=1), annotation_text="5 %", annotation_position="bottom right")
    return fig


def build_e6(cells: list, height: int = 360):
    """E6: mittlere Transferentropie ohne Kopplung gegen die Wochenzahl: gaußsch und Plug-in mit 3 und 5 Klassen (durchgezogen) gegen ihre Näherungen (gestrichelt), logarithmische Achse."""
    fig = go.Figure()
    palette = {"g": C.COLOR_Z, 3: C.COLOR_Y, 5: C.COLOR_X}
    for key, label in (("g", "gaußsch"), (3, "Plug-in, 3 Klassen"), (5, "Plug-in, 5 Klassen")):
        mine = sorted((c for c in cells if key == "g" or c["k"] == key), key=lambda c: c["T"])
        if key == "g":
            mine = [c for c in mine if c["k"] == 3]
        xs = [c["T"] for c in mine]
        fig.add_trace(go.Scatter(x=xs, y=[c["gauss" if key == "g" else "plugin"] for c in mine], mode="lines+markers", name=label, line=dict(color=palette[key])))
        fig.add_trace(go.Scatter(x=xs, y=[c["gauss_pred" if key == "g" else "plugin_pred"] for c in mine], mode="lines", name=f"{label}: Näherung", line=dict(color=palette[key], dash="dash")))
    fig.update_xaxes(title_text="Wochen T", type="log")
    fig.update_yaxes(title_text="mittlere Transferentropie ohne Kopplung (nats)", type="log")
    return _base(fig, height)


def build_e7(cells: list, height: int = 340):
    """E7: Anteil „erkannt“ des Granger-F-Tests je Stärke der verzögerten gemeinsamen Ursache, obwohl x nie auf y wirkt."""
    mine = sorted(cells, key=lambda c: c["b"])
    fig = _bars(mine, ("xy", "yx"), ("x → y „erkannt“", "y → x „erkannt“"), (C.COLOR_Y, C.COLOR_X), lambda c: f"Stärke {_fmt(c['b'], 1)}", "Anteil der Läufe (%)", height)
    fig.add_hline(y=5, line=dict(color=C.COLOR_DARK, dash="dot", width=1), annotation_text="5 %", annotation_position="bottom right")
    return fig
