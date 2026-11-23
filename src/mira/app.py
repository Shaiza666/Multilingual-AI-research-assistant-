"""Streamlit interface for asking questions over uploaded or benchmark PDFs."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from mira.config import load_config
from mira.retrieve import METHODS

REPO = Path(__file__).resolve().parents[2]
FONT = Path(__file__).resolve().parent / "assets" / "fonts" / "NotoNaskhArabic-Regular.ttf"


def _css() -> str:
    face = ""
    if FONT.exists():
        encoded = base64.b64encode(FONT.read_bytes()).decode("ascii")
        face = (
            "@font-face { font-family: 'Noto Naskh Arabic'; "
            f"src: url(data:font/truetype;base64,{encoded}) format('truetype'); }}"
        )
    return face + """
    .urdu { direction: rtl; text-align: right; font-family: 'Noto Naskh Arabic', 'Noto Nastaliq Urdu', sans-serif; font-size: 1.15rem; line-height: 1.9; }
    .answer-card { background: #fffdf8; border: 1px solid #e4dccf; border-radius: 12px; padding: 1rem 1.1rem; }
    .passage { background: #f6f3ee; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }
    """


def _load_index(path: str):
    from mira.indexing import IndexStore

    return IndexStore.load(path)


def _render_text(text: str, language: str) -> None:
    css_class = "urdu" if language == "ur" else ""
    st.markdown(f"<div class='answer-card {css_class}'>{text}</div>", unsafe_allow_html=True)


def _results_tab() -> None:
    metrics_path = REPO / "results" / "metrics.json"
    report_path = REPO / "results" / "RESULTS.md"
    if not metrics_path.exists():
        st.info("No measured results yet. From the repository root run `make evaluate`.")
        return
    st.markdown(report_path.read_text(encoding="utf-8") if report_path.exists() else "")
    figures = REPO / "results" / "figures"
    for image in sorted(figures.glob("*.png")) if figures.exists() else []:
        st.image(str(image), caption=image.stem.replace("_", " "))


def main() -> None:
    st.set_page_config(page_title="MIRA research assistant", page_icon="📄", layout="wide")
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)
    cfg = load_config(REPO / "configs" / "default.yaml")
    st.title("MIRA")
    st.caption("Ask an English or Urdu question about indexed research PDFs. Answers cite a document, page, and chunk.")

    with st.sidebar:
        st.header("Index")
        default_index = "data/indexes/benchmark"
        index_path = st.text_input("Index directory", value=st.session_state.get("index_path", default_index))
        uploads = st.file_uploader("Add PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Index uploaded PDFs", disabled=not uploads):
            upload_dir = REPO / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            for uploaded in uploads:
                (upload_dir / uploaded.name).write_bytes(uploaded.getvalue())
            from mira.indexing import build_index
            from mira.ingest import ingest_paths

            with st.spinner("Reading PDFs and building the index on CPU…"):
                chunks = ingest_paths([upload_dir], cfg.chunk.chunk_words, cfg.chunk.overlap_words, cfg.chunk.min_words)
                out = REPO / "data" / "indexes" / "uploads"
                build_index(chunks, out, cfg, embed=True)
            st.session_state["index_path"] = "data/indexes/uploads"
            index_path = "data/indexes/uploads"
            st.success(f"Indexed {len(chunks)} chunks.")
        method = st.selectbox("Retrieval", METHODS, index=METHODS.index("hybrid_rerank"))
        top_k = st.slider("Passages", 1, 8, cfg.retrieval.top_k)
        use_gate = st.checkbox("Use pilot-tuned evidence gate", value=True)
        st.caption("The first question loads the CPU models and can take a few minutes.")

    ask_tab, results_tab = st.tabs(["Ask", "Measured results"])
    with results_tab:
        _results_tab()
    with ask_tab:
        question = st.text_area("Question", height=110, placeholder="What does MISH stand for?   /   MISH کس کا مخفف ہے؟")
        if st.button("Answer", type="primary", disabled=not question.strip()):
            from mira.generate import GenerationCache
            from mira.pipeline import ask

            path = Path(index_path)
            if not path.is_absolute():
                path = REPO / path
            if not (path / "chunks.jsonl").exists():
                st.error("No index at that directory. Upload PDFs or run make ingest.")
            else:
                thresholds = None
                threshold_file = REPO / "results" / "thresholds.json"
                if use_gate and threshold_file.exists():
                    thresholds = json.loads(threshold_file.read_text(encoding="utf-8"))
                with st.spinner("Retrieving passages and drafting a grounded answer…"):
                    result = ask(
                        question.strip(),
                        _load_index(str(path)),
                        cfg,
                        method=method,
                        top_k=top_k,
                        thresholds=thresholds,
                        cache=GenerationCache(REPO / "data" / "cache" / "generations.jsonl"),
                    )
                st.session_state["result"] = result
        result = st.session_state.get("result")
        if result is None:
            st.markdown("Index a PDF, then ask in English or Urdu. Weak evidence is refused instead of guessed.")
        else:
            st.markdown(f"Language **{result.language}** · method `{result.method}`")
            _render_text(result.answer, result.language)
            if result.abstained:
                st.warning(result.abstain_reason or "Abstained")
            for warning in result.warnings:
                st.warning(warning)
            if result.translated_query:
                st.caption("Urdu question translated for keyword retrieval: " + result.translated_query)
            if result.citation_ids:
                choice = st.radio("Citations", result.citation_ids, horizontal=True)
                match = next((hit for hit in result.retrieved if hit["chunk_id"] == choice), None)
                if match:
                    st.markdown(
                        f"**{match['doc_id']}** · page {match['page']} · chunk {match['chunk_index']} · score {match['score']:.4f}"
                    )
                    _render_text(match["text"], match.get("language") or "en")
            with st.expander("Retrieved passages"):
                for hit in result.retrieved:
                    st.markdown(
                        f"<div class='passage'><b>{hit['chunk_id']}</b> · score {hit['score']:.4f}<br>{hit['text']}</div>",
                        unsafe_allow_html=True,
                    )
            with st.expander("Citation check"):
                if not result.judgements:
                    st.write("No sentences were checked.")
                for row in result.judgements:
                    flag = "kept" if row["supported"] else "dropped"
                    st.write(
                        f"{flag} · lexical {row['lexical']:.2f} · embedding {row['embedding'] if row['embedding'] is not None else 'n/a'} · {row['sentence']}"
                    )
            timing = ", ".join(f"{name} {value:.0f} ms" for name, value in result.timings_ms.items())
            st.caption(timing + (" · cached generation" if result.cache_hit else ""))


if __name__ == "__main__":
    main()
else:
    # `streamlit run` executes the file as a script, so __name__ is "__main__".
    # Importing the module for tests does not launch the interface.
    pass
