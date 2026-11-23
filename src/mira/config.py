from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mira.errors import MiraError


@dataclass
class ChunkConfig:
    chunk_words: int = 160
    overlap_words: int = 40
    min_words: int = 8


@dataclass
class EmbeddingConfig:
    model_name: str = "intfloat/multilingual-e5-small"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = 16


@dataclass
class RerankConfig:
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    pool: int = 20
    batch_size: int = 8


@dataclass
class TranslationConfig:
    ur_en_model: str = "Helsinki-NLP/opus-mt-ur-en"
    en_ur_model: str = "Helsinki-NLP/opus-mt-en-ur"
    max_new_tokens: int = 128
    enabled: bool = True


@dataclass
class LLMConfig:
    backend: str = "local-gguf"
    gguf_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    gguf_repo: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    gguf_filename: str = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    transformers_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    n_ctx: int = 3072
    n_threads: int = 4
    max_new_tokens: int = 160
    temperature: float = 0.0
    repeat_penalty: float = 1.1
    seed: int = 0
    passage_chars: int = 1100


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 40
    rrf_k: int = 60


@dataclass
class GroundingConfig:
    lexical_threshold: float = 0.35
    embedding_threshold: float = 0.82
    min_gate_accuracy: float = 0.6


@dataclass
class EvalConfig:
    primary_k: int = 5
    ks: list[int] = field(default_factory=lambda: [5, 10])


@dataclass
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    index_dir: str = "data/indexes/benchmark"
    results_dir: str = "results"


def _section(data: dict, name: str) -> dict:
    value = data.get(name) or {}
    if not isinstance(value, dict):
        raise MiraError(f"Config section '{name}' must be a mapping")
    return value


def load_config(path: str | Path | None = None) -> Config:
    raw: dict = {}
    if path is not None:
        file_path = Path(path)
        if not file_path.exists():
            raise MiraError(f"Config file not found: {file_path}")
        loaded = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise MiraError("Config file must be a YAML mapping")
        raw = loaded
    chunk = _section(raw, "chunk")
    embedding = _section(raw, "embedding")
    rerank = _section(raw, "rerank")
    translation = _section(raw, "translation")
    llm = _section(raw, "llm")
    retrieval = _section(raw, "retrieval")
    grounding = _section(raw, "grounding")
    evaluation = _section(raw, "evaluation")
    cfg = Config(
        chunk=ChunkConfig(**{k: chunk[k] for k in chunk if k in ChunkConfig.__dataclass_fields__}),
        embedding=EmbeddingConfig(**{k: embedding[k] for k in embedding if k in EmbeddingConfig.__dataclass_fields__}),
        rerank=RerankConfig(**{k: rerank[k] for k in rerank if k in RerankConfig.__dataclass_fields__}),
        translation=TranslationConfig(
            **{k: translation[k] for k in translation if k in TranslationConfig.__dataclass_fields__}
        ),
        llm=LLMConfig(**{k: llm[k] for k in llm if k in LLMConfig.__dataclass_fields__}),
        retrieval=RetrievalConfig(**{k: retrieval[k] for k in retrieval if k in RetrievalConfig.__dataclass_fields__}),
        grounding=GroundingConfig(**{k: grounding[k] for k in grounding if k in GroundingConfig.__dataclass_fields__}),
        evaluation=EvalConfig(**{k: evaluation[k] for k in evaluation if k in EvalConfig.__dataclass_fields__}),
        index_dir=str(raw.get("index_dir") or "data/indexes/benchmark"),
        results_dir=str(raw.get("results_dir") or "results"),
    )
    _apply_env(cfg)
    return cfg


def _apply_env(cfg: Config) -> None:
    backend = os.environ.get("MIRA_LLM_BACKEND")
    if backend:
        cfg.llm.backend = backend.strip()
    gguf = os.environ.get("MIRA_GGUF_PATH")
    if gguf:
        cfg.llm.gguf_path = gguf
    if os.environ.get("OPENAI_BASE_URL"):
        cfg.llm.openai_base_url = os.environ["OPENAI_BASE_URL"]
    if os.environ.get("OPENAI_API_KEY"):
        cfg.llm.openai_api_key = os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_MODEL"):
        cfg.llm.openai_model = os.environ["OPENAI_MODEL"]
    threads = os.environ.get("MIRA_THREADS")
    if threads:
        cfg.llm.n_threads = int(threads)


def setup_logging(verbose: bool = False) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    for name in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock", "faiss", "faiss.loader"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("llama_cpp").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
    warnings.filterwarnings("ignore", message=".*sacremoses.*")
