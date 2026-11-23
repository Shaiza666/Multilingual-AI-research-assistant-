"""Run retrieval and answer evaluation. Every number written here is measured."""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from pathlib import Path

from mira.config import Config
from mira.generate import GenerationCache, generate
from mira.grounding import citation_existence_precision, citation_page_precision
from mira.indexing import IndexStore
from mira.metrics import (
    bootstrap_mean_ci,
    exact_match,
    hit_at_k,
    macro_average,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)
from mira.models import release
from mira.pipeline import AskResult, ask
from mira.qa import QAItem, assert_disjoint, load_qa
from mira.report import write_report
from mira.retrieve import METHODS, search

logger = logging.getLogger("mira.evaluate")


def tune_threshold(pairs: list[tuple[float, bool]], min_coverage: float = 1.0) -> dict:
    """Choose a minimum evidence score on the pilot split.

    Scores are not comparable across methods, so this is called once per method.
    When every answerable pilot score is above every unanswerable score, the
    threshold is the midpoint of that gap. That sits below the weakest
    answerable pilot item, instead of on top of it, so a slightly lower test
    score is not refused. If the scores overlap, the gate stays off rather than
    dropping an answerable pilot question.
    """
    if not pairs:
        return {"threshold": None, "pilot_accuracy": None, "enabled": False, "n": 0, "rule": "empty"}
    answerable_scores = [score for score, label in pairs if label and math.isfinite(score)]
    unanswerable_scores = [score for score, label in pairs if not label and math.isfinite(score)]
    answerable_n = sum(1 for _score, label in pairs if label)
    majority = max(answerable_n, len(pairs) - answerable_n) / len(pairs)

    def accuracy(threshold: float) -> float:
        correct = 0
        for score, label in pairs:
            predicted = math.isfinite(score) and score >= threshold
            if predicted == label:
                correct += 1
        return correct / len(pairs)

    threshold = None
    rule = "disabled"
    enabled = False
    if answerable_scores and unanswerable_scores and min(answerable_scores) > max(unanswerable_scores):
        gap_low = max(unanswerable_scores)
        gap_high = min(answerable_scores)
        threshold = gap_low + 0.5 * (gap_high - gap_low)
        rule = "gap_midpoint"
        enabled = True
    pilot_accuracy = accuracy(threshold) if enabled and threshold is not None else None
    if enabled and pilot_accuracy is not None and pilot_accuracy <= majority + 1e-12:
        enabled = False
        threshold = None
        rule = "disabled_below_majority"
        pilot_accuracy = None
    del min_coverage
    return {
        "threshold": threshold if enabled else None,
        "pilot_accuracy": pilot_accuracy,
        "majority_baseline": majority,
        "enabled": enabled,
        "rule": rule,
        "n": len(pairs),
        "gap": (
            [max(unanswerable_scores), min(answerable_scores)]
            if answerable_scores and unanswerable_scores
            else None
        ),
        "pilot_scores": [{"score": score, "answerable": label} for score, label in pairs],
    }


def _pages(result_chunks: list[dict]) -> list[tuple[str, int]]:
    return [(row["doc_id"], int(row["page"])) for row in result_chunks]


def retrieval_row(pages: list[tuple[str, int]], gold: set[tuple[str, int]], ks: list[int]) -> dict:
    row: dict = {"mrr": reciprocal_rank(pages, gold)}
    for k in ks:
        row[f"recall@{k}"] = recall_at_k(pages, gold, k)
        row[f"ndcg@{k}"] = ndcg_at_k(pages, gold, k)
        row[f"hit@{k}"] = hit_at_k(pages, gold, k)
    return row


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _ci(rows: list[dict], key: str) -> list[float] | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    bounds = bootstrap_mean_ci(values)
    if bounds["low"] is None:
        return None
    return [bounds["low"], bounds["high"]]


def _summarize_retrieval(rows: list[dict], ks: list[int]) -> dict:
    summary: dict = {"n": len(rows), "mrr": macro_average(rows, "mrr"), "mrr_ci95": _ci(rows, "mrr")}
    for k in ks:
        summary[f"recall@{k}"] = macro_average(rows, f"recall@{k}")
        summary[f"recall@{k}_ci95"] = _ci(rows, f"recall@{k}")
        summary[f"ndcg@{k}"] = macro_average(rows, f"ndcg@{k}")
        summary[f"ndcg@{k}_ci95"] = _ci(rows, f"ndcg@{k}")
        summary[f"hit@{k}"] = macro_average(rows, f"hit@{k}")
    return summary


def _group(items_and_rows: list[tuple[QAItem, dict]], ks: list[int]) -> dict:
    grouped = {
        "all": [row for _item, row in items_and_rows],
        "en": [row for item, row in items_and_rows if item.language == "en"],
        "ur": [row for item, row in items_and_rows if item.language == "ur"],
        "lexical": [row for item, row in items_and_rows if item.query_type == "lexical"],
        "paraphrase": [row for item, row in items_and_rows if item.query_type == "paraphrase"],
        "english_papers": [row for item, row in items_and_rows if item.corpus != "urdu_doc"],
        "urdu_doc": [row for item, row in items_and_rows if item.corpus == "urdu_doc"],
    }
    return {name: _summarize_retrieval(rows, ks) for name, rows in grouped.items()}


def _summarize_answers(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "f1": macro_average(answerable, "f1"),
        "f1_ci95": _ci(answerable, "f1"),
        "f1_generated": macro_average(answerable, "f1_generated"),
        "semantic": macro_average(answerable, "semantic"),
        "semantic_ci95": _ci(answerable, "semantic"),
        "content_f1": macro_average(answerable, "content_f1"),
        "exact_match": macro_average(answerable, "em"),
        "abstention_accuracy": macro_average(rows, "abstention_correct"),
        "answerable_coverage": macro_average(answerable, "answered"),
        "unanswerable_abstention_rate": macro_average(unanswerable, "abstained_flag"),
        "citation_existence_precision": macro_average(answerable, "citation_existence"),
        "citation_gold_page_precision": macro_average(answerable, "citation_gold"),
        "raw_citation_existence_precision": macro_average(answerable, "raw_citation_existence"),
        "unsupported_rate_raw": macro_average(answerable, "unsupported_raw"),
        "mean_sentences_dropped": macro_average(answerable, "dropped"),
        "citation_rate": macro_average(answerable, "has_citation"),
    }


def _group_answers(items_and_rows: list[tuple[QAItem, dict]]) -> dict:
    grouped = {
        "all": [row for _item, row in items_and_rows],
        "en": [row for item, row in items_and_rows if item.language == "en"],
        "ur": [row for item, row in items_and_rows if item.language == "ur"],
        "english_papers": [row for item, row in items_and_rows if item.corpus != "urdu_doc"],
        "urdu_doc": [row for item, row in items_and_rows if item.corpus == "urdu_doc"],
    }
    return {name: _summarize_answers(rows) for name, rows in grouped.items()}


def _score_answer(item: QAItem, result: AskResult, index: IndexStore) -> dict:
    known = set(index.by_id)
    chunk_pages = {chunk.chunk_id: (chunk.doc_id, chunk.page) for chunk in index.chunks}
    abstained = bool(result.abstained)
    row: dict = {
        "id": item.id,
        "answerable": item.answerable,
        "abstained_flag": 1.0 if abstained else 0.0,
        "answered": 0.0 if abstained else 1.0,
        "abstention_correct": (0.0 if abstained else 1.0) if item.answerable else (1.0 if abstained else 0.0),
    }
    if not item.answerable:
        return row
    if abstained:
        row.update(
            {
                "f1": 0.0,
                "f1_generated": 0.0,
                "content_f1": 0.0,
                "em": 0.0,
                "citation_existence": None,
                "citation_gold": None,
                "raw_citation_existence": citation_existence_precision(result.raw_answer, known),
                "unsupported_raw": None,
                "dropped": None,
                "has_citation": None,
            }
        )
        return row
    judgements = result.judgements
    unsupported = None
    if judgements:
        unsupported = sum(1 for item_j in judgements if not item_j["supported"]) / len(judgements)
    row.update(
        {
            "f1": token_f1(result.answer, item.answer, item.language),
            "f1_generated": (
                token_f1(result.answer, item.answer, item.language)
                if result.answer_mode == "generated"
                else 0.0
            ),
            "content_f1": token_f1(result.english_answer, item.answer_en or item.answer, "en"),
            "em": exact_match(result.answer, item.answer, item.language),
            "citation_existence": citation_existence_precision(result.answer, known),
            "citation_gold": citation_page_precision(result.answer, chunk_pages, item.gold_set()),
            "raw_citation_existence": citation_existence_precision(result.raw_answer, known),
            "unsupported_raw": unsupported,
            "dropped": float(len(result.dropped)),
            "has_citation": 1.0 if result.citation_ids else 0.0,
        }
    )
    return row


def _retrieve_all(index: IndexStore, cfg: Config, items: list[QAItem], method: str, top_k: int) -> list[list[dict]]:
    from mira.models import embed_query_text, make_translator, rerank_pairs

    embed_query = None if method == "bm25" else (lambda text: embed_query_text(cfg, text))
    rerank = None
    if method in {"hybrid_rerank", "hybrid_translate_rerank"}:
        rerank = lambda query, passages: rerank_pairs(
            cfg.rerank.model_name, query, passages, cfg.rerank.batch_size, cfg.llm.n_threads
        )
    translate = make_translator(cfg, "ur-en") if cfg.translation.enabled else None
    rankings = []
    for item in items:
        ranked, translated = search(
            index,
            item.question,
            method=method,
            top_k=top_k,
            candidate_k=cfg.retrieval.candidate_k,
            rrf_k=cfg.retrieval.rrf_k,
            rerank_pool=cfg.rerank.pool,
            embed_query=embed_query,
            rerank=rerank,
            translate=translate,
            query_language=item.language,
        )
        rows = [hit.to_dict() for hit in ranked]
        for row in rows:
            row["translated_query"] = translated
        rankings.append(rows)
    return rankings


def _attach_semantic(records: list[tuple], cfg: Config) -> None:
    """Multilingual cosine between the final answer and the reference. Abstention scores 0."""
    import numpy as np

    from mira.models import encode_texts

    pending = []
    for scored, prediction, gold, answerable, abstained in records:
        if not answerable:
            scored["semantic"] = None
            continue
        if abstained or not str(prediction).strip() or not str(gold).strip():
            scored["semantic"] = 0.0
            continue
        pending.append((scored, str(prediction)[:800], str(gold)[:800]))
    if not pending:
        return
    flat: list[str] = []
    for _scored, prediction, gold in pending:
        flat.append(cfg.embedding.query_prefix + prediction)
        flat.append(cfg.embedding.query_prefix + gold)
    vectors = encode_texts(cfg.embedding.model_name, flat, cfg.embedding.batch_size, cfg.llm.n_threads)
    for index, (scored, _prediction, _gold) in enumerate(pending):
        scored["semantic"] = float(np.dot(vectors[2 * index], vectors[2 * index + 1]))


def run_evaluation(
    index: IndexStore,
    cfg: Config,
    gold_path: Path,
    pilot_path: Path,
    out_dir: Path,
    methods: list[str] | None = None,
    limit: int | None = None,
    skip_generate: bool = False,
    profiles: list[str] | None = None,
) -> dict:
    methods = list(methods or METHODS)
    pilot, pilot_meta = load_qa(pilot_path, "pilot")
    test, test_meta = load_qa(gold_path, "test")
    assert_disjoint(pilot, test)
    if limit is not None:
        test = test[:limit]
    ks = list(cfg.evaluation.ks)
    eval_k = max(max(ks), cfg.retrieval.top_k)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = GenerationCache(out_dir / "cache" / "generations.jsonl")

    logger.info("Retrieving %s test and %s pilot questions", len(test), len(pilot))
    pilot_rankings: dict[str, list[list[dict]]] = {}
    test_rankings: dict[str, list[list[dict]]] = {}
    for method in methods:
        logger.info("Retrieval method %s", method)
        started = time.perf_counter()
        pilot_rankings[method] = _retrieve_all(index, cfg, pilot, method, eval_k)
        test_rankings[method] = _retrieve_all(index, cfg, test, method, eval_k)
        logger.info("Retrieval %s finished in %.1f s", method, time.perf_counter() - started)

    thresholds = {}
    for method in methods:
        pairs = []
        for item, ranking in zip(pilot, pilot_rankings[method]):
            score = float(ranking[0]["score"]) if ranking else float("-inf")
            pairs.append((score, item.answerable))
        thresholds[method] = tune_threshold(pairs, cfg.grounding.min_gate_coverage)
        logger.info(
            "Evidence gate %s enabled=%s accuracy=%.3f",
            method,
            thresholds[method]["enabled"],
            thresholds[method]["pilot_accuracy"] or -1,
        )

    release("reranker")
    retrieval_summary: dict = {}
    for method in methods:
        retrieval_rows = []
        for item, ranking in zip(test, test_rankings[method]):
            if item.answerable:
                retrieval_rows.append((item, retrieval_row(_pages(ranking), item.gold_set(), ks)))
        retrieval_summary[method] = _group(retrieval_rows, ks)

    chosen_profiles = list(profiles or ([cfg.llm.profile] if cfg.llm.profiles else ["default"]))
    by_model: dict = {}
    predictions: list[dict] = []
    errors: list[dict] = []
    from mira.config import apply_profile

    for profile in chosen_profiles:
        if profile != "default":
            apply_profile(cfg, profile)
        release("llm")
        warmup_s = None
        if not skip_generate:
            logger.info("Warming up profile %s (%s)", profile, cfg.llm.gguf_path)
            started = time.perf_counter()
            generate(cfg, [{"role": "user", "content": "Reply with OK."}], cache=None)
            warmup_s = time.perf_counter() - started
            logger.info("Warmup %s took %.1f s", profile, warmup_s)
        answer_summary: dict = {}
        latency: dict = {}
        semantic_rows: list[tuple] = []
        for method in methods:
            answer_rows: list[tuple[QAItem, dict]] = []
            retrieve_times: list[float] = []
            generate_times: list[float] = []
            e2e_times: list[float] = []
            if skip_generate:
                continue
            for item, _ranking in zip(test, test_rankings[method]):
                try:
                    result = ask(
                        item.question,
                        index,
                        cfg,
                        method=method,
                        top_k=cfg.retrieval.top_k,
                        thresholds=thresholds,
                        cache=cache,
                        language=item.language,
                    )
                except Exception as exc:
                    logger.exception("Failed on %s / %s / %s", profile, method, item.id)
                    errors.append({"profile": profile, "method": method, "id": item.id, "error": str(exc)})
                    continue
                scored = _score_answer(item, result, index)
                answer_rows.append((item, scored))
                semantic_rows.append((scored, result.answer, item.answer, item.answerable, result.abstained))
                retrieve_times.append(result.timings_ms.get("retrieve", 0.0))
                if "generate" in result.timings_ms:
                    generate_times.append(result.timings_ms["generate"])
                e2e_times.append(sum(result.timings_ms.values()))
                predictions.append(
                    {
                        "split": "test",
                        "profile": profile,
                        "model": cfg.llm.gguf_path,
                        "method": method,
                        "id": item.id,
                        "language": item.language,
                        "query_type": item.query_type,
                        "answerable": item.answerable,
                        "question": item.question,
                        "gold_answer": item.answer,
                        "gold_pages": [{"doc_id": doc, "page": page} for doc, page in item.gold_pages],
                        "prediction": result.to_dict(),
                        "scores": scored,
                    }
                )
                if len(predictions) % 10 == 0:
                    logger.info("Answered %s rows so far (%s)", len(predictions), profile)
            if not skip_generate:
                answer_summary[method] = answer_rows
                latency[method] = {
                    "retrieve_median_ms": _median(retrieve_times),
                    "retrieve_mean_ms": (sum(retrieve_times) / len(retrieve_times)) if retrieve_times else None,
                    "generate_median_ms": _median(generate_times),
                    "generate_mean_ms": (sum(generate_times) / len(generate_times)) if generate_times else None,
                    "e2e_median_ms": _median(e2e_times),
                    "n_generated": len(generate_times),
                    "n_scored": len(answer_rows),
                }
        if not skip_generate:
            release("llm")
            _attach_semantic(semantic_rows, cfg)
            for method, rows in answer_summary.items():
                answer_summary[method] = _group_answers(rows)
        by_model[profile] = {
            "llm": cfg.llm.gguf_path,
            "warmup_seconds": warmup_s,
            "answers": answer_summary,
            "latency_ms": latency,
        }

    from mira.hardware import collect_hardware

    primary = "quality" if "quality" in by_model else (chosen_profiles[0] if chosen_profiles else "default")
    primary_block = by_model.get(primary, {"answers": {}, "latency_ms": {}, "warmup_seconds": None, "llm": cfg.llm.gguf_path})
    metrics = {
        "task": "mira benchmark",
        "test_size": len(test),
        "pilot_size": len(pilot),
        "test_construction": test_meta.get("construction"),
        "pilot_construction": pilot_meta.get("construction"),
        "methods": methods,
        "profiles": chosen_profiles,
        "primary_profile": primary,
        "ks": ks,
        "primary_k": cfg.evaluation.primary_k,
        "hardware": collect_hardware(),
        "models": {
            "embedding": cfg.embedding.model_name,
            "reranker": cfg.rerank.model_name,
            "translation_ur_en": cfg.translation.ur_en_model,
            "translation_en_ur": cfg.translation.en_ur_model,
            "llm_backend": cfg.llm.backend,
            "llm": primary_block["llm"],
            "profiles": {name: block["llm"] for name, block in by_model.items()},
            "grounding_lexical_threshold": cfg.grounding.lexical_threshold,
            "grounding_embedding_threshold": cfg.grounding.embedding_threshold,
            "extractive_min_cosine": cfg.grounding.extractive_min_cosine,
        },
        "warmup_seconds": primary_block["warmup_seconds"],
        "thresholds": thresholds,
        "retrieval": retrieval_summary,
        "answers": primary_block["answers"],
        "latency_ms": primary_block["latency_ms"],
        "by_model": by_model,
        "errors": errors,
        "skip_generate": skip_generate,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "thresholds.json").write_text(
        json.dumps({method: {k: v for k, v in info.items() if k != "pilot_scores"} | {"pilot_scores": info.get("pilot_scores")} for method, info in thresholds.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if predictions:
        with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(metrics, out_dir)
    return metrics
