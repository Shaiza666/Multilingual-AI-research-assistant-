"""Write CSV, markdown, and charts from measured metrics. Nothing is imputed."""

from __future__ import annotations

import csv
from pathlib import Path


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bar_groups(path: Path, title: str, methods: list[str], series: dict[str, list[float | None]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(series)
    x = np.arange(len(methods))
    width = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#1F4E79", "#C47B4A", "#2F6F4E", "#8C4A6F", "#4C6A7A", "#B0892E"]
    for index, (name, values) in enumerate(series.items()):
        heights = [0.0 if value is None else value for value in values]
        offset = (index - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, heights, width=width, label=name, color=colors[index % len(colors)])
    ax.set_xticks(x, methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=min(3, len(labels)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _latency_chart(path: Path, methods: list[str], retrieve: list[float | None], generate: list[float | None]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.36
    ax.bar(x, [0 if v is None else v for v in retrieve], width=0.6, label="retrieve median", color="#1F4E79")
    ax.set_xticks(x, methods, rotation=20, ha="right")
    ax.set_ylabel("milliseconds")
    ax.set_title("Median retrieval time per question")
    ax.legend(frameon=False)
    del generate
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _best(metrics: dict, section: str, group: str, key: str) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for method, groups in metrics.get(section, {}).items():
        value = groups.get(group, {}).get(key)
        if value is None:
            continue
        scored.append((method, float(value)))
    if not scored:
        return []
    best = max(value for _method, value in scored)
    return [(method, value) for method, value in scored if abs(value - best) < 1e-9]


def _best_label(winners: list[tuple[str, float]]) -> str:
    names = ", ".join(method for method, _value in winners)
    if len(winners) > 1:
        names = f"{names} (tied)"
    return f"**{names}** at {_fmt(winners[0][1])}"


def _discussion(metrics: dict) -> str:
    lines = []
    k = metrics.get("primary_k", 5)
    recall_key = f"recall@{k}"
    pairs = [
        ("all", recall_key, "page Recall@{k} on the full answerable test set"),
        ("en", recall_key, "page Recall@{k} on English answerable questions"),
        ("ur", recall_key, "page Recall@{k} on Urdu answerable questions"),
        ("lexical", recall_key, "page Recall@{k} on lexical questions"),
        ("paraphrase", recall_key, "page Recall@{k} on paraphrased questions"),
    ]
    lines.append("The comparisons below are computed from the tables in this file.")
    for group, key, label in pairs:
        winners = _best(metrics, "retrieval", group, key)
        if not winners:
            continue
        lines.append(f"- Highest {label.format(k=k)}: {_best_label(winners)}.")
    if not metrics.get("skip_generate"):
        for group, label in (("all", "question-language token F1"), ("en", "English token F1"), ("ur", "Urdu token F1")):
            winners = _best(metrics, "answers", group, "f1")
            if winners:
                lines.append(f"- Highest {label}: {_best_label(winners)}.")
        content = _best(metrics, "answers", "ur", "content_f1")
        surface = _best(metrics, "answers", "ur", "f1")
        if content and surface:
            lines.append(
                "- Urdu content F1 compares the grounded English answer with the English reference "
                f"(best {_best_label(content)}). Urdu token F1 compares the final answer, "
                "after Urdu translation when the draft was English, with the Urdu reference "
                f"(best {_best_label(surface)})."
            )
    bm25_ur = metrics.get("retrieval", {}).get("bm25", {}).get("ur", {}).get(recall_key)
    dense_ur = metrics.get("retrieval", {}).get("dense", {}).get("ur", {}).get(recall_key)
    if bm25_ur is not None and dense_ur is not None:
        if dense_ur > bm25_ur + 0.02:
            lines.append(
                "- On Urdu questions the dense retriever's page recall is higher than BM25. "
                "The papers are in English, so Urdu questions share few exact tokens with the pages, "
                "while the multilingual encoder can still rank a page that states the same fact."
            )
        elif bm25_ur > dense_ur + 0.02:
            lines.append(
                "- On Urdu questions BM25's page recall is higher than the dense retriever on this set. "
                "That can happen when the Urdu questions keep English names, numbers, and dataset titles, "
                "which keyword search matches directly."
            )
        else:
            lines.append("- On Urdu questions BM25 and dense page recall are close on this set.")
    bm25_lex = metrics.get("retrieval", {}).get("bm25", {}).get("lexical", {}).get(recall_key)
    dense_par = metrics.get("retrieval", {}).get("dense", {}).get("paraphrase", {}).get(recall_key)
    bm25_par = metrics.get("retrieval", {}).get("bm25", {}).get("paraphrase", {}).get(recall_key)
    if bm25_lex is not None:
        lines.append(f"- BM25 page Recall@{k} on lexical questions is {_fmt(bm25_lex)}.")
    if bm25_par is not None and dense_par is not None:
        lines.append(
            f"- On paraphrased questions, BM25 page Recall@{k} is {_fmt(bm25_par)} and dense page Recall@{k} is {_fmt(dense_par)}."
        )
    rerank_ur = metrics.get("retrieval", {}).get("hybrid_rerank", {}).get("ur", {}).get(recall_key)
    translate_ur = metrics.get("retrieval", {}).get("hybrid_translate", {}).get("ur", {}).get(recall_key)
    if rerank_ur is not None and translate_ur is not None and dense_ur is not None:
        lines.append(
            f"- Urdu page Recall@{k} was {_fmt(rerank_ur)} for hybrid plus reranking, "
            f"{_fmt(dense_ur)} for dense retrieval, and {_fmt(translate_ur)} for hybrid retrieval "
            "with Urdu-to-English query translation. Reranking used the original question. "
            "Translation adds an English keyword list, which helps only when the translation keeps the technical terms."
        )
    return "\n".join(lines)


def write_report(metrics: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    figures = out_dir / "figures"
    methods = list(metrics.get("methods") or [])
    k = int(metrics.get("primary_k") or 5)
    retrieval_rows = []
    for method in methods:
        for group, summary in metrics.get("retrieval", {}).get(method, {}).items():
            retrieval_rows.append({"method": method, "group": group, **summary})
    _write_csv(out_dir / "retrieval_metrics.csv", retrieval_rows)

    answer_rows = []
    for method in methods:
        for group, summary in metrics.get("answers", {}).get(method, {}).items():
            answer_rows.append({"method": method, "group": group, **summary})
    _write_csv(out_dir / "answer_metrics.csv", answer_rows)

    latency_rows = [{"method": method, **row} for method, row in metrics.get("latency_ms", {}).items()]
    _write_csv(out_dir / "latency.csv", latency_rows)

    if methods and metrics.get("retrieval"):
        _bar_groups(
            figures / "retrieval_metrics.png",
            f"Retrieval on answerable test questions (k={k})",
            methods,
            {
                f"Recall@{k}": [metrics["retrieval"][m]["all"].get(f"recall@{k}") for m in methods],
                "MRR": [metrics["retrieval"][m]["all"].get("mrr") for m in methods],
                f"nDCG@{k}": [metrics["retrieval"][m]["all"].get(f"ndcg@{k}") for m in methods],
            },
        )
        _bar_groups(
            figures / "retrieval_by_language.png",
            f"Page Recall@{k} by question language",
            methods,
            {
                "English": [metrics["retrieval"][m]["en"].get(f"recall@{k}") for m in methods],
                "Urdu": [metrics["retrieval"][m]["ur"].get(f"recall@{k}") for m in methods],
            },
        )
    if methods and metrics.get("answers"):
        _bar_groups(
            figures / "answer_f1.png",
            "Token F1 on answerable questions (abstention scores 0)",
            methods,
            {
                "English F1": [metrics["answers"][m]["en"].get("f1") for m in methods],
                "Urdu F1": [metrics["answers"][m]["ur"].get("f1") for m in methods],
                "Urdu content F1": [metrics["answers"][m]["ur"].get("content_f1") for m in methods],
            },
        )
        _bar_groups(
            figures / "abstention.png",
            "Abstention behaviour",
            methods,
            {
                "accuracy": [metrics["answers"][m]["all"].get("abstention_accuracy") for m in methods],
                "coverage of answerable": [metrics["answers"][m]["all"].get("answerable_coverage") for m in methods],
                "abstain on unanswerable": [metrics["answers"][m]["all"].get("unanswerable_abstention_rate") for m in methods],
            },
        )
    if methods and metrics.get("latency_ms"):
        _latency_chart(
            figures / "latency.png",
            methods,
            [metrics["latency_ms"][m].get("retrieve_median_ms") for m in methods],
            [metrics["latency_ms"][m].get("generate_median_ms") for m in methods],
        )

    lines = ["# Measured results", ""]
    hw = metrics.get("hardware") or {}
    lines.append(
        f"Hardware: {hw.get('processor')} ({hw.get('logical_cpus')} logical CPUs, {hw.get('ram_gib')} GiB RAM), "
        f"Python {hw.get('python')}, device {hw.get('device')}."
    )
    lines.append("")
    lines.append(f"Test questions: {metrics.get('test_size')}. Pilot questions used only for the evidence gate: {metrics.get('pilot_size')}.")
    if metrics.get("warmup_seconds") is not None:
        lines.append(f"Generator warmup (not included in per-question latency): {_fmt(metrics['warmup_seconds'], 1)} s.")
    lines.append("")
    lines.append("## Retrieval")
    lines.append("")
    lines.append(f"| method | group | n | Recall@{k} | MRR | nDCG@{k} | Hit@{k} |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in retrieval_rows:
        if row["group"] not in {"all", "en", "ur", "lexical", "paraphrase"}:
            continue
        lines.append(
            "| {method} | {group} | {n} | {r} | {m} | {nd} | {h} |".format(
                method=row["method"],
                group=row["group"],
                n=row.get("n"),
                r=_fmt(row.get(f"recall@{k}")),
                m=_fmt(row.get("mrr")),
                nd=_fmt(row.get(f"ndcg@{k}")),
                h=_fmt(row.get(f"hit@{k}")),
            )
        )
    lines.append("")
    if answer_rows:
        lines.append("## Answers")
        lines.append("")
        lines.append(
            "| method | group | n | F1 | F1 95% CI | semantic | generated-only F1 | EM | abstention acc. | coverage | unanswerable abstain | citation page precision | raw unsupported rate |"
        )
        lines.append("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in answer_rows:
            ci = row.get("f1_ci95")
            ci_text = "n/a" if not ci else f"{_fmt(ci[0])}–{_fmt(ci[1])}"
            lines.append(
                "| {method} | {group} | {n} | {f1} | {ci} | {sem} | {gen} | {em} | {ab} | {cov} | {ua} | {cit} | {uns} |".format(
                    method=row["method"],
                    group=row["group"],
                    n=row.get("n"),
                    f1=_fmt(row.get("f1")),
                    ci=ci_text,
                    sem=_fmt(row.get("semantic")),
                    gen=_fmt(row.get("f1_generated")),
                    em=_fmt(row.get("exact_match")),
                    ab=_fmt(row.get("abstention_accuracy")),
                    cov=_fmt(row.get("answerable_coverage")),
                    ua=_fmt(row.get("unanswerable_abstention_rate")),
                    cit=_fmt(row.get("citation_gold_page_precision")),
                    uns=_fmt(row.get("unsupported_rate_raw")),
                )
            )
        lines.append("")
        lines.append("Raw F1 and EM above are unchanged. The next table strips citations and the bilingual frame, then folds Urdu letters, digits, and punctuation. `F1 either` is the better of that score against the question-language reference and against the English reference. It is not a replacement for raw F1.")
        lines.append("")
        lines.append("| method | group | F1 norm | F1 norm 95% CI | EM norm | F1 either | F1 either 95% CI |")
        lines.append("| --- | --- | ---: | --- | ---: | ---: | --- |")
        for row in answer_rows:
            if row["group"] not in {"all", "en", "ur", "urdu_doc"}:
                continue
            nci = row.get("f1_norm_ci95")
            eci = row.get("f1_either_ci95")
            lines.append(
                "| {method} | {group} | {fn} | {nci} | {em} | {fe} | {eci} |".format(
                    method=row["method"],
                    group=row["group"],
                    fn=_fmt(row.get("f1_norm")),
                    nci="n/a" if not nci else f"{_fmt(nci[0])}–{_fmt(nci[1])}",
                    em=_fmt(row.get("exact_match_normalized")),
                    fe=_fmt(row.get("f1_either")),
                    eci="n/a" if not eci else f"{_fmt(eci[0])}–{_fmt(eci[1])}",
                )
            )
        lines.append("")
        lines.append("## Latency")
        lines.append("")
        lines.append("Retrieve times below are measured. A generate median near 0.1 ms means the draft was read from the generation cache, not that the model ran in a tenth of a millisecond.")
        lines.append("")
        lines.append("| method | retrieve median ms | generate median ms | English generate ms | Urdu generate ms | end-to-end median ms | generations |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in latency_rows:
            lines.append(
                f"| {row['method']} | {_fmt(row.get('retrieve_median_ms'), 1)} | {_fmt(row.get('generate_median_ms'), 1)} | "
                f"{_fmt(row.get('generate_median_ms_en'), 1)} | {_fmt(row.get('generate_median_ms_ur'), 1)} | "
                f"{_fmt(row.get('e2e_median_ms'), 1)} | {row.get('n_generated')} |"
            )
        lines.append("")
    by_model = metrics.get("by_model") or {}
    if len(by_model) > 1 and not metrics.get("skip_generate"):
        lines.append("## Model comparison")
        lines.append("")
        lines.append("Retrieval is shared. Each profile is a different local GGUF model. F1 is the question-language token F1 of the primary hybrid method when that row exists, otherwise the best method for that model.")
        lines.append("")
        lines.append("| profile | model | best method by F1 | F1 | English F1 | Urdu F1 | semantic | generate median ms |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for name, block in by_model.items():
            answers = block.get("answers") or {}
            best_name = None
            best_f1 = None
            for method, groups in answers.items():
                value = (groups.get("all") or {}).get("f1")
                if value is None:
                    continue
                if best_f1 is None or value > best_f1:
                    best_f1 = value
                    best_name = method
            group = (answers.get(best_name) or {}).get("all") or {} if best_name else {}
            latency = (block.get("latency_ms") or {}).get(best_name) or {}
            model_name = str(block.get("llm") or "").rsplit("/", 1)[-1]
            lines.append(
                f"| {name} | {model_name} | {best_name or 'n/a'} | {_fmt(group.get('f1'))} | "
                f"{_fmt((answers.get(best_name) or {}).get('en', {}).get('f1') if best_name else None)} | "
                f"{_fmt((answers.get(best_name) or {}).get('ur', {}).get('f1') if best_name else None)} | "
                f"{_fmt(group.get('semantic'))} | {_fmt(latency.get('generate_median_ms'), 1)} |"
            )
        lines.append("")
    lines.append("## Reading the numbers")
    lines.append("")
    lines.append(_discussion(metrics))
    lines.append("")
    if metrics.get("errors"):
        lines.append("## Errors")
        lines.append("")
        for error in metrics["errors"]:
            lines.append(f"- `{error['method']}` / `{error['id']}`: {error['error']}")
        lines.append("")
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
