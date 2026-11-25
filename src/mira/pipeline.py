"""Question answering: retrieve, optionally abstain, generate, ground, and translate."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from mira.config import Config, apply_profile
from mira.generate import GenerationCache, build_messages, fit_passages, generate
from mira.grounding import GroundingResult, best_extractive_sentence, ground_answer, parse_citation_ids
from mira.indexing import IndexStore
from mira.retrieve import search
from mira.textnorm import detect_language

ABSTAIN_MESSAGE = {
    "en": "I could not find an answer in the indexed documents.",
    "ur": "مجھے انڈیکس شدہ دستاویزات میں اس سوال کا جواب نہیں ملا۔",
}


@dataclass
class Runtime:
    embed_query: object | None = None
    rerank: object | None = None
    translate: object | None = None
    generate: object | None = None
    embed_claim: object | None = None


@dataclass
class AskResult:
    question: str
    language: str
    method: str
    answer: str
    raw_answer: str
    english_answer: str
    abstained: bool
    abstain_reason: str | None
    retrieved: list[dict]
    citation_ids: list[str]
    judgements: list[dict]
    dropped: list[dict]
    translated_query: str | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cache_hit: bool = False
    answer_mode: str = "abstain"
    model_path: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "language": self.language,
            "method": self.method,
            "answer": self.answer,
            "raw_answer": self.raw_answer,
            "english_answer": self.english_answer,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "retrieved": self.retrieved,
            "citation_ids": self.citation_ids,
            "judgements": self.judgements,
            "dropped": self.dropped,
            "translated_query": self.translated_query,
            "timings_ms": self.timings_ms,
            "warnings": self.warnings,
            "cache_hit": self.cache_hit,
            "answer_mode": self.answer_mode,
            "model_path": self.model_path,
        }


def _default_runtime(cfg: Config, method: str, need_generate: bool) -> Runtime:
    from mira.models import embed_query_text, make_translator, rerank_pairs

    runtime = Runtime()
    if method != "bm25":
        runtime.embed_query = lambda text: embed_query_text(cfg, text)
        runtime.embed_claim = runtime.embed_query
    if method in {"hybrid_rerank", "hybrid_translate_rerank"}:
        runtime.rerank = lambda query, passages: rerank_pairs(
            cfg.rerank.model_name, query, passages, cfg.rerank.batch_size, cfg.llm.n_threads
        )
    if cfg.translation.enabled:
        runtime.translate = make_translator(cfg, "ur-en")
    if need_generate:
        runtime.generate = None  # filled by generate() using cfg
    return runtime


def _gate(method: str, top_score: float | None, thresholds: dict | None) -> bool:
    if not thresholds:
        return False
    info = thresholds.get(method)
    if not info or not info.get("enabled"):
        return False
    if top_score is None:
        return True
    return float(top_score) < float(info["threshold"])


def _judgement_dict(item) -> dict:
    return {
        "sentence": item.sentence,
        "supported": item.supported,
        "best_chunk_id": item.best_chunk_id,
        "lexical": item.lexical,
        "embedding": item.embedding,
        "model_citations": item.model_citations,
    }


def _finish_abstain(
    question: str,
    language: str,
    method: str,
    reason: str,
    retrieved: list[dict],
    translated_query: str | None,
    timings: dict[str, float],
    warnings: list[str],
    raw_answer: str = "",
    answer_mode: str = "abstain",
) -> AskResult:
    return AskResult(
        question=question,
        language=language,
        method=method,
        answer=ABSTAIN_MESSAGE.get(language, ABSTAIN_MESSAGE["en"]),
        raw_answer=raw_answer,
        english_answer="",
        abstained=True,
        abstain_reason=reason,
        retrieved=retrieved,
        citation_ids=[],
        judgements=[],
        dropped=[],
        translated_query=translated_query,
        timings_ms=timings,
        warnings=warnings,
        answer_mode=answer_mode,
    )


def ask(
    question: str,
    index: IndexStore,
    cfg: Config,
    method: str = "hybrid_rerank",
    top_k: int | None = None,
    generate_answer: bool = True,
    thresholds: dict | None = None,
    runtime: Runtime | None = None,
    cache: GenerationCache | None = None,
    language: str | None = None,
) -> AskResult:
    language = language or detect_language(question)
    if language == "unknown":
        language = "en"
    top_k = top_k or cfg.retrieval.top_k
    warnings: list[str] = []
    timings: dict[str, float] = {}
    if runtime is None:
        runtime = _default_runtime(cfg, method, generate_answer)

    started = time.perf_counter()
    retrieved, translated = search(
        index,
        question,
        method=method,
        top_k=top_k,
        candidate_k=cfg.retrieval.candidate_k,
        rrf_k=cfg.retrieval.rrf_k,
        rerank_pool=cfg.rerank.pool,
        embed_query=runtime.embed_query,
        rerank=runtime.rerank,
        translate=runtime.translate,
        query_language=language,
    )
    timings["retrieve"] = (time.perf_counter() - started) * 1000
    retrieved_dicts = [item.to_dict() for item in retrieved]
    top_score = retrieved[0].score if retrieved else None
    if not retrieved:
        return _finish_abstain(question, language, method, "no passages retrieved", [], translated, timings, warnings)
    if _gate(method, top_score, thresholds):
        return _finish_abstain(
            question,
            language,
            method,
            "evidence score below the pilot-tuned threshold",
            retrieved_dicts,
            translated,
            timings,
            warnings,
        )
    if not generate_answer:
        return AskResult(
            question=question,
            language=language,
            method=method,
            answer="",
            raw_answer="",
            english_answer="",
            abstained=False,
            abstain_reason=None,
            retrieved=retrieved_dicts,
            citation_ids=[],
            judgements=[],
            dropped=[],
            translated_query=translated,
            timings_ms=timings,
            warnings=warnings,
        )

    budget = max(800, cfg.llm.n_ctx * 3 - cfg.llm.max_new_tokens * 4)
    passages = fit_passages(
        [(item.chunk.chunk_id, item.chunk.text) for item in retrieved],
        cfg.llm.passage_chars,
        budget,
    )
    messages = build_messages(question, language, passages)
    started = time.perf_counter()
    cache_hit = False
    model_path = cfg.llm.gguf_path
    if runtime.generate is not None:
        raw = runtime.generate(messages)
    else:
        raw, cache_hit, model_path = _generate_routed(cfg, language, messages, cache)
    timings["generate"] = (time.perf_counter() - started) * 1000

    passage_map = {item.chunk.chunk_id: item.chunk.text for item in retrieved}
    embedding_matrix = None
    if runtime.embed_claim is not None and index.embeddings is not None:
        embedding_matrix = {}
        for item in retrieved:
            vector = index.vector_for(item.chunk.chunk_id)
            if vector is not None:
                embedding_matrix[item.chunk.chunk_id] = [float(value) for value in vector]

    def embed_text(claim: str):
        vector = runtime.embed_claim(claim)
        return [float(value) for value in vector]

    started = time.perf_counter()
    grounded: GroundingResult = ground_answer(
        raw,
        passage_map,
        lexical_threshold=cfg.grounding.lexical_threshold,
        embedding_threshold=cfg.grounding.embedding_threshold,
        embedding_matrix=embedding_matrix,
        embed_text=embed_text if embedding_matrix is not None else None,
        question=question,
        rank_order=[item.chunk.chunk_id for item in retrieved],
    )
    timings["ground"] = (time.perf_counter() - started) * 1000
    answer_mode = "generated"
    english_answer = grounded.english_answer
    judgements = grounded.judgements
    dropped = grounded.dropped
    if grounded.abstained:
        extracted = None
        if runtime.embed_claim is not None:
            extracted = best_extractive_sentence(
                [(item.chunk.chunk_id, item.chunk.text) for item in retrieved],
                question,
                embed_text,
                cfg.grounding.extractive_min_cosine,
            )
        if extracted is None:
            return _finish_abstain(
                question,
                language,
                method,
                grounded.reason or "abstained",
                retrieved_dicts,
                translated,
                timings,
                warnings,
                raw_answer=raw,
            )
        sentence, chunk_id, support = extracted
        answer_mode = "extractive"
        english_answer = f"{sentence} [{chunk_id}]"
        judgements = []
        dropped = grounded.dropped
        warnings.append(f"Draft was not usable; showing a retrieved sentence (cosine {support:.2f}).")

    final = english_answer
    if language == "ur" and not _has_urdu_words(strip_for_language(final)):
        final = _bilingual_urdu(english_answer, runtime, warnings, timings)
    return AskResult(
        question=question,
        language=language,
        method=method,
        answer=final,
        raw_answer=raw,
        english_answer=english_answer,
        abstained=False,
        abstain_reason=None,
        retrieved=retrieved_dicts,
        citation_ids=parse_citation_ids(final),
        judgements=[_judgement_dict(item) for item in judgements],
        dropped=[_judgement_dict(item) for item in dropped],
        answer_mode=answer_mode,
        translated_query=translated,
        timings_ms=timings,
        warnings=warnings,
        cache_hit=cache_hit,
        model_path=model_path,
    )


def _generate_routed(cfg: Config, language: str, messages: list[dict], cache: GenerationCache | None):
    """Use the Urdu profile for Urdu questions and restore the caller's model."""
    saved = (
        cfg.llm.profile,
        cfg.llm.gguf_path,
        cfg.llm.gguf_repo,
        cfg.llm.gguf_filename,
        cfg.llm.max_new_tokens,
    )
    try:
        if language == "ur" and cfg.llm.urdu_profile:
            apply_profile(cfg, cfg.llm.urdu_profile)
            cfg.llm.max_new_tokens = min(saved[4], cfg.llm.urdu_max_new_tokens or saved[4])
        elif language == "ur":
            cfg.llm.max_new_tokens = min(saved[4], cfg.llm.urdu_max_new_tokens or saved[4])
        text, hit = generate(cfg, messages, cache)
        return text, hit, cfg.llm.gguf_path
    finally:
        cfg.llm.profile, cfg.llm.gguf_path, cfg.llm.gguf_repo, cfg.llm.gguf_filename, cfg.llm.max_new_tokens = saved


def _has_urdu_words(text: str) -> bool:
    """True when the answer uses Urdu words, even if Latin terms outnumber them."""
    from mira.textnorm import tokenize

    urdu_tokens = [tok for tok in tokenize(text) if any("\u0600" <= ch <= "\u06ff" for ch in tok)]
    return len(urdu_tokens) >= 2


def strip_for_language(text: str) -> str:
    from mira.grounding import CITE_RE

    return CITE_RE.sub(" ", text)


def _translation_ok(source: str, translated: str) -> bool:
    import re

    from mira.grounding import strip_citations

    if detect_language(translated) != "ur":
        return False
    if sum(ch.isdigit() for ch in translated) >= 8 or translated.count("0") >= 4:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|\d{2,}", strip_citations(source))
    if not tokens:
        return True
    lowered = translated.lower()
    kept = sum(1 for token in tokens if token.lower() in lowered)
    return kept / len(tokens) >= 0.34


_TERM = re.compile(r"[A-Za-z][A-Za-z0-9.+-]{1,}|\d[\d.,%]*")


def protect_latin_terms(text: str) -> tuple[str, list[str]]:
    """Hide Latin names and numbers so a translator cannot drop them."""
    terms: list[str] = []

    def _replace(match) -> str:
        terms.append(match.group(0))
        return f" XX{len(terms) - 1} "

    return _TERM.sub(_replace, text), terms


def restore_latin_terms(text: str, terms: list[str]) -> str:
    restored = text
    for index, term in enumerate(terms):
        restored = restored.replace(f"XX{index}", term)
    return restored


def _bilingual_urdu(english_answer: str, runtime: Runtime, warnings: list[str], timings: dict[str, float]) -> str:
    from mira.grounding import CITE_RE, strip_citations

    translated = None
    if runtime.translate is not None:
        started = time.perf_counter()
        try:
            translated = _translate_preserving_citations(english_answer, runtime.translate)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Urdu answer translation failed ({exc}).")
        timings["translate_answer"] = (time.perf_counter() - started) * 1000
    if translated and _translation_ok(english_answer, translated):
        return translated
    warnings.append("The Urdu wording was unreliable, so the answer keeps the English source sentence.")
    body = strip_citations(english_answer).strip().rstrip(".")
    cites = " ".join(f"[{cid}]" for cid in CITE_RE.findall(english_answer))
    framed = f"انگریزی ماخذ کے مطابق: {body}."
    return f"{framed} {cites}".strip()


def _translate_preserving_citations(text: str, translate) -> str:
    from mira.grounding import CITE_RE, split_sentences, strip_citations

    pieces = []
    for sentence in split_sentences(text):
        cites = CITE_RE.findall(sentence)
        bare = strip_citations(sentence)
        if not bare:
            continue
        if detect_language(bare) == "ur":
            translated = bare
        else:
            protected, terms = protect_latin_terms(bare)
            translated = restore_latin_terms(translate(protected, "en-ur").strip(), terms)
        if cites:
            translated = translated.rstrip() + " " + " ".join(f"[{cid}]" for cid in cites)
        pieces.append(translated)
    return " ".join(pieces)
