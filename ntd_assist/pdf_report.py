import datetime
import unicodedata
from fpdf import FPDF


def _latinize(s: str) -> str:
    """Make a string safe for FPDF's latin-1 codepage. Decomposes accents,
    replaces μ→u, then falls back to '?' for anything still unmappable."""
    if s is None:
        return ""
    s = str(s).replace("μ", "u")
    s = unicodedata.normalize("NFKD", s)
    return s.encode("latin-1", "replace").decode("latin-1")


def create_pdf(res, sample_type, stain, magnification, context):
    """Generate PDF diagnostic report"""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "NTD-Assist Diagnostic Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Sample Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, _latinize(f"Sample Type: {sample_type}"), ln=True)
    pdf.cell(0, 6, _latinize(f"Stain: {stain}"), ln=True)
    pdf.cell(0, 6, _latinize(f"Magnification: {magnification}"), ln=True)
    pdf.cell(0, 6, _latinize(f"Patient Context: {context if context else 'Not provided'}"), ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    if res.detected_disease not in ["Negative for Parasites", "Unclear"]:
        pdf.set_fill_color(255, 230, 230)
    else:
        pdf.set_fill_color(230, 255, 230)
    pdf.cell(0, 10, _latinize(f"DIAGNOSIS: {res.detected_disease}"), ln=True, fill=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, _latinize(f"Species: {res.species}"), ln=True)
    pdf.cell(0, 7, _latinize(f"Severity: {res.severity}"), ln=True)
    pdf.cell(0, 7, _latinize(f"Confidence: {res.confidence}"), ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Morphological Evidence", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, _latinize(res.morphology_proof))
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Detailed Findings", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, _latinize(res.findings))
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Recommendation", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, _latinize(res.recommendation))
    pdf.ln(5)

    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 5, "DISCLAIMER: This AI-assisted analysis is for educational and screening purposes only. "
                         "All findings must be confirmed by a qualified medical professional. "
                         "Do not use as sole basis for clinical decisions.")

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", "replace")
    return bytes(out)
