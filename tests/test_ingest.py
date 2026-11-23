from pathlib import Path

import pymupdf

from mira.ingest import ingest_pdf


def test_pdf_pages_are_one_based_and_headers_drop(tmp_path: Path):
    path = tmp_path / "2103.00001.pdf"
    document = pymupdf.open()
    for index in range(4):
        page = document.new_page()
        page.insert_text((72, 72), "Running Header Example")
        page.insert_text(
            (72, 140),
            f"The transformer uses attention on leaf {index} with several additional words in the sentence.",
        )
    document.save(path)
    document.close()
    chunks = ingest_pdf(path, chunk_words=40, overlap_words=8, min_words=4)
    assert chunks
    assert all(chunk.doc_id == "2103.00001" for chunk in chunks)
    assert all("Running Header Example" not in chunk.text for chunk in chunks)
    assert any("attention" in chunk.text for chunk in chunks)
    assert chunks[0].page >= 1


def test_urdu_pdf_round_trip(tmp_path: Path):
    font = Path("src/mira/assets/fonts/NotoNaskhArabic-Regular.ttf")
    path = tmp_path / "urdu-note.pdf"
    document = pymupdf.open()
    page = document.new_page()
    css = f"@font-face {{ font-family: Naskh; src: url('{font.as_posix()}'); }} body {{ font-family: Naskh; font-size: 16px; }}"
    html = "<body dir='rtl'>دریائے سندھ جنوبی ایشیا کا ایک بڑا دریا ہے اور اس کی لمبائی مشہور ہے۔</body>"
    page.insert_htmlbox(pymupdf.Rect(72, 72, 540, 400), html, css=css)
    document.save(path)
    document.close()
    chunks = ingest_pdf(path, chunk_words=40, overlap_words=5, min_words=3)
    joined = " ".join(chunk.text for chunk in chunks)
    assert "سندھ" in joined
    assert chunks[0].language == "ur"
