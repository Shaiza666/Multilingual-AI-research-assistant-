"""Download the CPU models used by the default configuration."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import snapshot_download

from mira.config import load_config


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "mira-download-models"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temporary.replace(dest)


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    profiles = cfg.llm.profiles or {
        "default": {
            "gguf_repo": cfg.llm.gguf_repo,
            "gguf_filename": cfg.llm.gguf_filename,
            "gguf_path": cfg.llm.gguf_path,
        }
    }
    for name, profile in profiles.items():
        repo = str(profile["gguf_repo"])
        filename = str(profile["gguf_filename"])
        relative = str(profile.get("gguf_path") or f"models/{filename}")
        dest = ROOT / relative
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        print(name, url, "->", dest)
        _download(url, dest)
        print(name, dest)

    print("embedding", cfg.embedding.model_name)
    snapshot_download(cfg.embedding.model_name)
    print("reranker", cfg.rerank.model_name)
    snapshot_download(cfg.rerank.model_name)
    print("translation", cfg.translation.ur_en_model)
    snapshot_download(cfg.translation.ur_en_model)
    print("translation", cfg.translation.en_ur_model)
    snapshot_download(cfg.translation.en_ur_model)


if __name__ == "__main__":
    main()
