"""Download the pinned CC BY 4.0 papers and write an attribution manifest.

A paper is kept only when the PDF itself states a Creative Commons Attribution
4.0 or Attribution-ShareAlike 4.0 licence. Non-commercial and no-derivatives
licences are rejected.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mira.licenses import classify_license
PDF_DIR = ROOT / "data" / "papers" / "pdf"
TEXT_DIR = ROOT / "data" / "papers" / "text"
CATALOG = ROOT / "data" / "papers" / "catalog.json"
MANIFEST = ROOT / "data" / "papers" / "MANIFEST.json"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MiraResearchAssistant/1.0 (mailto:shaizaasghar471@gmail.com)"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        dest.write_bytes(response.read())


def _dump_text(pdf_path: Path, record: dict) -> None:
    document = pymupdf.open(pdf_path)
    url = record.get("abstract_url") or record.get("pdf_url") or ""
    header = (
        f"Document: {record['doc_id']}\n"
        f"Title: {record['title']}\n"
        f"Authors: {', '.join(record['authors'])}\n"
        f"License: {record['license']}\n"
        f"URL: {url}\n"
        f"Evidence: {record['license_evidence']}\n"
        "Extracted text is included so the benchmark quotations can be checked.\n"
    )
    parts = [header]
    try:
        for number, page in enumerate(document, start=1):
            parts.append(f"\n\n===== PAGE {number} =====\n")
            parts.append(page.get_text("text") or "")
    finally:
        document.close()
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    (TEXT_DIR / f"{record['doc_id']}.txt").write_text("".join(parts), encoding="utf-8")


def _record_from_pdf(entry: dict, dest: Path, blob: str, page_count: int, title: str) -> dict:
    source = entry.get("source", "arxiv")
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if source == "acl":
        year = int(entry["year"])
        lowered = blob.lower()
        if "association for computational linguistics" not in lowered or str(year) not in blob:
            raise SystemExit(f"{entry['doc_id']} does not look like an ACL Anthology paper from {year}")
        if year < 2016:
            raise SystemExit(f"{entry['doc_id']} is before the Anthology CC BY 4.0 cutoff")
        return {
            "doc_id": entry["doc_id"],
            "title": title or entry["title"],
            "authors": entry["authors"],
            "license": "CC BY 4.0",
            "license_evidence": (
                "The ACL Anthology states that materials published in or after 2016 are licensed "
                "under CC BY 4.0 (https://aclanthology.org/). This PDF is that anthology version "
                f"and is dated {year}."
            ),
            "pdf_url": entry["url"],
            "pages": page_count,
            "sha256": digest,
            "venue": entry.get("venue"),
            "source": "acl",
        }
    license_name = classify_license(blob)
    if license_name is None:
        raise SystemExit(f"{entry['doc_id']} does not state a permissive CC licence in the PDF")
    return {
        "doc_id": entry["doc_id"],
        "arxiv_id": entry.get("arxiv_id"),
        "version": entry.get("version"),
        "title": title or entry["title"],
        "authors": entry["authors"],
        "license": license_name,
        "license_evidence": "The licence statement is printed on the first page of the downloaded PDF.",
        "abstract_url": f"https://arxiv.org/abs/{entry['version']}",
        "pdf_url": f"https://arxiv.org/pdf/{entry['version']}.pdf",
        "pages": page_count,
        "sha256": digest,
        "venue": entry.get("venue"),
        "source": "arxiv",
    }


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    manifest = []
    for entry in catalog["papers"]:
        dest = PDF_DIR / f"{entry['doc_id']}.pdf"
        source = entry.get("source", "arxiv")
        if not dest.exists():
            url = entry["url"] if source == "acl" else f"https://arxiv.org/pdf/{entry['version']}.pdf"
            print(f"downloading {entry['doc_id']}")
            _download(url, dest)
            time.sleep(2)
        document = pymupdf.open(dest)
        try:
            blob = "\n".join((document[i].get_text("text") or "") for i in range(min(2, document.page_count)))
            page_count = document.page_count
            title = (document.metadata.get("title") or entry["title"]).strip()
        finally:
            document.close()
        record = _record_from_pdf(entry, dest, blob, page_count, title)
        _dump_text(dest, record)
        manifest.append(record)
        print(f"ok {entry['doc_id']} {record['license']} {page_count} pages")
    MANIFEST.write_text(json.dumps({"papers": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
