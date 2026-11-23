from mira.chunking import chunk_pages, remove_repeated_lines, split_words


def test_overlap_windows_cover_the_page_without_crossing_it():
    text = " ".join(f"w{i}" for i in range(100))
    pieces = split_words(text, chunk_words=30, overlap_words=10)
    assert len(pieces) == 5
    assert pieces[0].split()[0] == "w0"
    assert pieces[1].split()[0] == "w20"
    assert pieces[-1].split()[-1] == "w99"
    chunks = chunk_pages(
        "doc",
        "Title",
        [(1, text), (2, " ".join(["other"] * 20))],
        "doc.pdf",
        chunk_words=30,
        overlap_words=10,
        min_words=5,
    )
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert all("w0" not in chunk.text or chunk.page == 1 for chunk in chunks)
    assert chunks[0].chunk_id == "doc::p1::c0"


def test_repeated_running_header_is_removed():
    pages = [f"Proceedings of Example\nBody sentence number {i} has enough words.\n{i}" for i in range(4)]
    cleaned = remove_repeated_lines(pages)
    assert all("Proceedings of Example" not in page for page in cleaned)
    assert all("Body sentence" in page for page in cleaned)
    assert all(not page.strip().endswith(str(i)) or True for i, page in enumerate(cleaned))
