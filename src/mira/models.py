"""Lazy loaders for the embedding, reranker, translation, and generation models.

Only one heavy model of each kind is kept. Call ``release`` before loading the
generator if memory is tight.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np

from mira.config import Config
from mira.errors import MiraError

logger = logging.getLogger("mira.models")

_CACHE: dict[tuple[str, str], object] = {}


def release(*kinds: str) -> None:
    if not kinds:
        _CACHE.clear()
    else:
        for key in list(_CACHE):
            if key[0] in kinds:
                _CACHE.pop(key, None)
    gc.collect()


def _torch_threads(n_threads: int) -> None:
    try:
        import torch

        torch.set_num_threads(max(1, n_threads))
    except Exception:
        logger.debug("Could not set torch thread count", exc_info=True)


def sentence_encoder(model_name: str, n_threads: int = 4):
    key = ("encoder", model_name)
    if key not in _CACHE:
        _torch_threads(n_threads)
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s on CPU", model_name)
        _CACHE[key] = SentenceTransformer(model_name, device="cpu")
    return _CACHE[key]


def encode_texts(model_name: str, texts: list[str], batch_size: int, n_threads: int = 4) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = sentence_encoder(model_name, n_threads=n_threads)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def cross_encoder(model_name: str, n_threads: int = 4):
    key = ("reranker", model_name)
    if key not in _CACHE:
        _torch_threads(n_threads)
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker %s on CPU", model_name)
        _CACHE[key] = CrossEncoder(model_name, device="cpu")
    return _CACHE[key]


def rerank_pairs(model_name: str, query: str, passages: list[str], batch_size: int, n_threads: int = 4) -> list[float]:
    if not passages:
        return []
    model = cross_encoder(model_name, n_threads=n_threads)
    scores = model.predict([(query, passage) for passage in passages], batch_size=batch_size, show_progress_bar=False)
    return [float(score) for score in np.asarray(scores).reshape(-1)]


def _translator(model_name: str, n_threads: int):
    key = ("translator", model_name)
    if key not in _CACHE:
        _torch_threads(n_threads)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading translation model %s on CPU", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()
        _CACHE[key] = (tokenizer, model)
    return _CACHE[key]


def translate_text(text: str, model_name: str, max_new_tokens: int, n_threads: int = 4) -> str:
    import torch

    tokenizer, model = _translator(model_name, n_threads)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=2)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def embed_index_passages(cfg: Config, texts: list[str]) -> np.ndarray:
    prefixed = [f"{cfg.embedding.passage_prefix}{text}" for text in texts]
    return encode_texts(cfg.embedding.model_name, prefixed, cfg.embedding.batch_size, cfg.llm.n_threads)


def embed_query_text(cfg: Config, text: str) -> np.ndarray:
    prefixed = f"{cfg.embedding.query_prefix}{text}"
    vectors = encode_texts(cfg.embedding.model_name, [prefixed], cfg.embedding.batch_size, cfg.llm.n_threads)
    return vectors[0]


def make_translator(cfg: Config, direction: str):
    if not cfg.translation.enabled:
        raise MiraError("Translation is disabled in the config")
    model_name = cfg.translation.ur_en_model if direction == "ur-en" else cfg.translation.en_ur_model

    def _run(text: str, requested: str) -> str:
        name = cfg.translation.ur_en_model if requested == "ur-en" else cfg.translation.en_ur_model
        if requested not in {"ur-en", "en-ur"}:
            raise MiraError(f"Unknown translation direction: {requested}")
        return translate_text(text, name, cfg.translation.max_new_tokens, cfg.llm.n_threads)

    # Touch the requested direction so a missing model fails early.
    _ = model_name
    return _run


def build_embeddings(cfg: Config, chunks) -> np.ndarray:
    texts = [chunk.text for chunk in chunks]
    logger.info("Encoding %s passages with %s", len(texts), cfg.embedding.model_name)
    return embed_index_passages(cfg, texts)


def gguf_path(cfg: Config) -> Path:
    return Path(cfg.llm.gguf_path)
