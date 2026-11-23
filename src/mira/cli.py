"""Command line interface: ingest, ask, and evaluate."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mira.config import load_config, setup_logging
from mira.errors import MiraError
from mira.hardware import collect_hardware

logger = logging.getLogger("mira.cli")


def _cmd_ingest(args: argparse.Namespace) -> int:
    from mira.indexing import build_index
    from mira.ingest import ingest_paths

    cfg = load_config(args.config)
    chunks = ingest_paths(args.inputs, cfg.chunk.chunk_words, cfg.chunk.overlap_words, cfg.chunk.min_words)
    logger.info("Chunked %s passages from %s files", len(chunks), len({c.doc_id for c in chunks}))
    store = build_index(chunks, args.index, cfg, embed=not args.no_embed)
    logger.info("Wrote index to %s (%s chunks)", store.directory, len(store.chunks))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from mira.generate import GenerationCache
    from mira.indexing import IndexStore
    from mira.pipeline import ask

    cfg = load_config(args.config)
    index = IndexStore.load(args.index)
    thresholds = None
    threshold_path = Path(args.thresholds) if args.thresholds else Path(cfg.results_dir) / "thresholds.json"
    if threshold_path.exists() and not args.no_gate:
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    cache = GenerationCache(Path(args.cache)) if args.cache else None
    result = ask(
        args.question,
        index,
        cfg,
        method=args.method,
        top_k=args.top_k,
        generate_answer=not args.retrieve_only,
        thresholds=thresholds,
        cache=cache,
    )
    print(result.answer)
    if result.abstained:
        print(f"abstained: {result.abstain_reason}")
    if result.citation_ids:
        print("citations: " + ", ".join(result.citation_ids))
    if result.translated_query:
        print("translated query: " + result.translated_query)
    for warning in result.warnings:
        print("warning: " + warning)
    if args.show_passages:
        for hit in result.retrieved:
            print(f"\n[{hit['chunk_id']}] p.{hit['page']} score={hit['score']:.4f}")
            print(hit["text"][:500])
    print(
        "timing_ms: "
        + ", ".join(f"{name}={value:.1f}" for name, value in result.timings_ms.items())
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from mira.evaluate import run_evaluation
    from mira.indexing import IndexStore

    cfg = load_config(args.config)
    index = IndexStore.load(args.index)
    metrics = run_evaluation(
        index,
        cfg,
        gold_path=Path(args.gold),
        pilot_path=Path(args.pilot),
        out_dir=Path(args.out),
        methods=args.methods,
        limit=args.limit,
        skip_generate=args.skip_generate,
    )
    logger.info("Wrote %s", Path(args.out) / "RESULTS.md")
    print(Path(args.out) / "RESULTS.md")
    if metrics.get("errors"):
        logger.warning("%s questions failed", len(metrics["errors"]))
    return 0


def _cmd_hardware(_args: argparse.Namespace) -> int:
    print(json.dumps(collect_hardware(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mira", description="Multilingual research assistant")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Read PDFs and build an index")
    ingest.add_argument("inputs", nargs="+", help="PDF files or directories")
    ingest.add_argument("--index", default="data/indexes/benchmark")
    ingest.add_argument("--config", default="configs/default.yaml")
    ingest.add_argument("--no-embed", action="store_true")
    ingest.set_defaults(func=_cmd_ingest)

    ask = sub.add_parser("ask", help="Ask a question in English or Urdu")
    ask.add_argument("question")
    ask.add_argument("--index", default="data/indexes/benchmark")
    ask.add_argument("--config", default="configs/default.yaml")
    ask.add_argument("--method", default="hybrid_translate_rerank")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--thresholds", default=None)
    ask.add_argument("--no-gate", action="store_true")
    ask.add_argument("--retrieve-only", action="store_true")
    ask.add_argument("--show-passages", action="store_true")
    ask.add_argument("--cache", default="data/cache/generations.jsonl")
    ask.set_defaults(func=_cmd_ask)

    evaluate = sub.add_parser("evaluate", help="Score retrieval and answers on the benchmark")
    evaluate.add_argument("--index", default="data/indexes/benchmark")
    evaluate.add_argument("--config", default="configs/default.yaml")
    evaluate.add_argument("--gold", default="data/qa/gold.json")
    evaluate.add_argument("--pilot", default="data/qa/pilot.json")
    evaluate.add_argument("--out", default="results")
    evaluate.add_argument("--methods", nargs="*", default=None)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--skip-generate", action="store_true")
    evaluate.set_defaults(func=_cmd_evaluate)

    hardware = sub.add_parser("hardware", help="Print CPU and package versions")
    hardware.set_defaults(func=_cmd_hardware)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        return int(args.func(args))
    except MiraError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
