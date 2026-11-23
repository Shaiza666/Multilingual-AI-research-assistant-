"""Print the measured tables from results/metrics.json. Does not invent values."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/metrics.json")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    k = metrics.get("primary_k", 5)
    print("test", metrics.get("test_size"), "pilot", metrics.get("pilot_size"))
    print("primary", metrics.get("primary_profile"), metrics.get("models", {}).get("llm"))
    print("\n## Retrieval")
    print(f"| method | Recall@{k} | English | Urdu | MRR | nDCG@{k} | Recall CI |")
    for method, groups in metrics["retrieval"].items():
        all_g = groups.get("all") or {}
        ci = all_g.get(f"recall@{k}_ci95")
        ci_text = "n/a" if not ci else f"{fmt(ci[0])}–{fmt(ci[1])}"
        print(
            f"| {method} | {fmt(all_g.get(f'recall@{k}'))} | {fmt((groups.get('en') or {}).get(f'recall@{k}'))} | "
            f"{fmt((groups.get('ur') or {}).get(f'recall@{k}'))} | {fmt(all_g.get('mrr'))} | "
            f"{fmt(all_g.get(f'ndcg@{k}'))} | {ci_text} |"
        )
    print("\n## Answers (primary model)")
    print("| method | F1 | CI | English | Urdu | semantic | generated-only F1 | EM | abstention | coverage | unanswerable abstain | citation page |")
    for method, groups in (metrics.get("answers") or {}).items():
        all_g = groups.get("all") or {}
        ci = all_g.get("f1_ci95")
        ci_text = "n/a" if not ci else f"{fmt(ci[0])}–{fmt(ci[1])}"
        print(
            f"| {method} | {fmt(all_g.get('f1'))} | {ci_text} | {fmt((groups.get('en') or {}).get('f1'))} | "
            f"{fmt((groups.get('ur') or {}).get('f1'))} | {fmt(all_g.get('semantic'))} | "
            f"{fmt(all_g.get('f1_generated'))} | {fmt(all_g.get('exact_match'))} | "
            f"{fmt(all_g.get('abstention_accuracy'))} | {fmt(all_g.get('answerable_coverage'))} | "
            f"{fmt(all_g.get('unanswerable_abstention_rate'))} | {fmt(all_g.get('citation_gold_page_precision'))} |"
        )
    print("\n## Models")
    for name, block in (metrics.get("by_model") or {}).items():
        print(name, block.get("llm"), "warmup", block.get("warmup_seconds"))
        latency = block.get("latency_ms") or {}
        for method, row in latency.items():
            print(" ", method, "gen_median", row.get("generate_median_ms"), "e2e", row.get("e2e_median_ms"), "n", row.get("n_generated"))


if __name__ == "__main__":
    main()
