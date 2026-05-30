"""
Regenerate docs/portfolio/StoryForge_RAG_Portfolio.pdf from the markdown source.

Usage (from repo root):
    pip install fpdf2
    py scripts/generate_portfolio_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "portfolio" / "StoryForge_RAG_Portfolio.md"
PDF_PATH = ROOT / "docs" / "portfolio" / "StoryForge_RAG_Portfolio.pdf"


def _ascii_safe(text: str) -> str:
    replacements = {
        "\u2192": "->",
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _strip_md(line: str) -> str:
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    return _ascii_safe(line.strip())


def main() -> None:
    try:
        from fpdf import FPDF
    except ImportError as e:
        raise SystemExit("Install fpdf2: pip install fpdf2") from e

    if not MD_PATH.exists():
        raise SystemExit(f"Missing {MD_PATH}")

    pdf = FPDF()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    width = pdf.epw

    in_code = False
    for raw in MD_PATH.read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            pdf.set_font("Courier", size=9)
            pdf.multi_cell(width, 4, _ascii_safe(raw.rstrip()))
            pdf.set_font("Helvetica", size=11)
            continue

        line = _strip_md(raw)
        if not line or line == "---":
            pdf.ln(3)
            continue
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(width, 8, line[2:])
            pdf.ln(2)
            pdf.set_font("Helvetica", size=11)
            continue
        if line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(width, 7, line[3:])
            pdf.ln(1)
            pdf.set_font("Helvetica", size=11)
            continue
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(width, 6, line[4:])
            pdf.set_font("Helvetica", size=11)
            continue
        if line.startswith("|"):
            if re.match(r"^\|[\s\-:|]+\|$", line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            line = "  |  ".join(cells)
        if line.startswith("- "):
            line = "- " + line[2:]
        pdf.multi_cell(width, 5, _ascii_safe(line))

    pdf.output(str(PDF_PATH))
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
