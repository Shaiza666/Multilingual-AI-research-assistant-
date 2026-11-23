"""Write a short original Urdu and English sample note. It is not part of the benchmark."""

from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "src" / "mira" / "assets" / "fonts" / "NotoNaskhArabic-Regular.ttf"
OUT = ROOT / "data" / "samples" / "sample-note.pdf"

ENGLISH = (
    "Sample note for the MIRA demonstration. This note is not part of the scored benchmark. "
    "It was written by Shaiza Asghar and is released under CC BY 4.0. "
    "The note records that the Indus is a major river of South Asia and that many reference books "
    "list its length as about 3180 kilometres."
)
URDU = (
    "یہ نوٹ میرا کے مظاہرے کے لیے لکھا گیا ہے اور اسکور کیے گئے بینچ مارک کا حصہ نہیں۔ "
    "اسے شائزہ اصغر نے لکھا ہے اور یہ CC BY 4.0 کے تحت جاری ہے۔ "
    "اس نوٹ کے مطابق دریائے سندھ جنوبی ایشیا کا ایک بڑا دریا ہے اور کئی حوالہ جاتی کتابیں "
    "اس کی لمبائی تقریباً 3180 کلومیٹر بتاتی ہیں۔"
)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    english = document.new_page()
    english.insert_textbox(pymupdf.Rect(72, 72, 540, 400), ENGLISH, fontsize=12, fontname="times-roman")
    urdu = document.new_page()
    css = f"@font-face {{ font-family: Naskh; src: url('{FONT.as_posix()}'); }} body {{ font-family: Naskh; font-size: 18px; }}"
    html = f"<body dir='rtl'>{URDU}</body>"
    urdu.insert_htmlbox(pymupdf.Rect(72, 72, 540, 500), html, css=css)
    document.save(OUT)
    document.close()
    print(OUT)


if __name__ == "__main__":
    main()
