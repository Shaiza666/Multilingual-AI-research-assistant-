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
    header = (
        f"Source: arXiv:{record['arxiv_id']} version {record['version']}\n"
        f"Title: {record['title']}\n"
        f"Authors: {', '.join(record['authors'])}\n"
        f"License: {record['license']}\n"
        f"URL: {record['abstract_url']}\n"
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


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    manifest = []
    for entry in catalog["papers"]:
        dest = PDF_DIR / f"{entry['doc_id']}.pdf"
        if not dest.exists():
            print(f"downloading {entry['version']}")
            _download(f"https://arxiv.org/pdf/{entry['version']}.pdf", dest)
            time.sleep(3)
        document = pymupdf.open(dest)
        try:
            blob = "\n".join((document[i].get_text("text") or "") for i in range(min(2, document.page_count)))
            page_count = document.page_count
            title = (document.metadata.get("title") or entry["title"]).strip()
        finally:
            document.close()
        license_name = classify_license(blob)
        if license_name is None:
            raise SystemExit(f"{entry['doc_id']} does not state a permissive CC licence in the PDF")
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        record = {
            "doc_id": entry["doc_id"],
            "arxiv_id": entry["arxiv_id"],
            "version": entry["version"],
            "title": title,
            "authors": entry["authors"],
            "license": license_name,
            "license_evidence": "The licence statement is printed on the first page of the downloaded PDF.",
            "abstract_url": f"https://arxiv.org/abs/{entry['version']}",
            "pdf_url": f"https://arxiv.org/pdf/{entry['version']}.pdf",
            "pages": page_count,
            "sha256": digest,
            "venue": entry.get("venue"),
        }
        _dump_text(dest, record)
        manifest.append(record)
        print(f"ok {entry['doc_id']} {license_name} {page_count} pages")
    MANIFEST.write_text(json.dumps({"papers": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
