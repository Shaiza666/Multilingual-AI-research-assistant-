"""Download the CPU models used by the default configuration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import hf_hub_download, snapshot_download

from mira.config import load_config


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    print("embedding", cfg.embedding.model_name)
    snapshot_download(cfg.embedding.model_name)
    print("reranker", cfg.rerank.model_name)
    snapshot_download(cfg.rerank.model_name)
    print("translation", cfg.translation.ur_en_model)
    snapshot_download(cfg.translation.ur_en_model)
    print("translation", cfg.translation.en_ur_model)
    snapshot_download(cfg.translation.en_ur_model)
    dest = ROOT / "models"
    dest.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=cfg.llm.gguf_repo,
        filename=cfg.llm.gguf_filename,
        local_dir=str(dest),
    )
    print("gguf", path)


if __name__ == "__main__":
    main()
