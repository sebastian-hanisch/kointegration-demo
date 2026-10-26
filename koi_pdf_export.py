"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Tabelle der Prüfungen mit Wert, Schwelle, Entscheidung und Wahrheit, Verdikte der Hauptansicht. Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

from fpdf import FPDF

import koi_constants as C
import koi_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, griechische und mathematische Zeichen)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "α": "alpha", "β": "beta", "γ": "gamma", "²": "^2",
             "½": "1/2", "…": "...", "ˆ": "", "̂": "", "Σ": "Summe", "φ": "phi"}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Gleichlauf, der hält: Kointegration und gerichtete Abhängigkeit"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    detail = (f"Anpassungstempo a: {s.a:g}   Anpassung: {C.SHARES[s.share]}   Faktor beta: {s.beta:g}   kurzfristige Kopplung gamma: {s.gamma:g}" if s.world == "ecm"
              else f"Kopplung c: {s.coupling:g}   Klassen: {s.classes}" if s.world == "coupled" else f"Stärke der Ursache: {s.driver:g}   Klassen: {s.classes}")
    _para(pdf, 5.5, f"Welt: {C.WORLDS[s.world]}   Wochen T: {s.t}   {detail}   Zufalls-Seed: {s.seed}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Pruefungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in a.rows:
        value = "-" if r.value is None else f"{r.value:.{r.digits}f}"
        cut = "" if r.threshold is None else f" (Schwelle {r.threshold:.2f})"
        _para(pdf, 5.5, f"{r.name}: {value}{cut}; {r.decision}; Wahrheit: {r.truth}")
    verdicts = [("Ergebnis", ev.verdict_main)] + ([("Anpassung", ev.verdict_adapt), ("Differenzen", ev.verdict_difference)] if s.world == "ecm" else [("Transferentropie", ev.verdict_te)])
    for title, verdict in verdicts:
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _text(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        _para(pdf, 5.5, verdict(a)[1])
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Simulierte Daten mit bekannter Welt. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
