"""Build a two-page PDF of a pinned Urdu Wikipedia article.

The article text is stored beside this script's output and is under CC BY-SA 4.0.
The PDF is an ingestion source. It is not one of the CC BY research papers.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "urdu" / "artificial-intelligence-ur.txt"
FONT = ROOT / "src" / "mira" / "assets" / "fonts" / "NotoNaskhArabic-Regular.ttf"
PDF = ROOT / "data" / "papers" / "pdf" / "urdu-ai.pdf"
DUMP = ROOT / "data" / "papers" / "text" / "urdu-ai.txt"


def body_paragraphs() -> list[str]:
    raw = SOURCE.read_text(encoding="utf-8")
    body = raw.split("\n\n", 1)[1]
    chunks = [part.strip() for part in body.split("\n\n") if part.strip()]
    return chunks


def main() -> None:
    paragraphs = body_paragraphs()
    # The opening and history stay on page 1. Later sections start page 2.
    split_at = 0
    for index, paragraph in enumerate(paragraphs):
        if paragraph.startswith("== اقسام"):
            split_at = index
            break
    pages = ["\n\n".join(paragraphs[:split_at]), "\n\n".join(paragraphs[split_at:])]
    PDF.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    css = (
        f"@font-face {{ font-family: Naskh; src: url('{FONT.as_posix()}'); }} "
        "body { font-family: Naskh; font-size: 15px; line-height: 1.5; }"
    )
    for text in pages:
        page = document.new_page()
        html = "<body dir='rtl'>" + text.replace("\n", "<br>") + "</body>"
        page.insert_htmlbox(pymupdf.Rect(50, 48, 545, 790), html, css=css)
    document.save(PDF)
    document.close()
    header = SOURCE.read_text(encoding="utf-8").split("\n\n", 1)[0]
    dump = [header, ""]
    for number, text in enumerate(pages, start=1):
        dump.append(f"\n\n===== PAGE {number} =====\n")
        dump.append(text)
    DUMP.write_text("".join(dump), encoding="utf-8")
    print(PDF, "pages", len(pages))


if __name__ == "__main__":
    main()
