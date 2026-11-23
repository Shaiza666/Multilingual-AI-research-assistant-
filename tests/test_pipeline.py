from pathlib import Path

import numpy as np

from mira.config import load_config
from mira.grounding import ABSTAIN_TOKEN
from mira.indexing import IndexStore
from mira.pipeline import Runtime, _has_urdu_words, ask
from mira.types import Chunk


def _index() -> IndexStore:
    chunks = [
        Chunk(
            "paper::p2::c0",
            "paper",
            "Paper",
            2,
            0,
            "Multi-Index Semantic Hashing learns binary hash codes for document similarity search.",
            "en",
            "paper.pdf",
        )
    ]
    return IndexStore(Path("."), chunks, np.ones((1, 4), dtype=np.float32), {})


def test_fake_generator_is_grounded():
    calls = {"n": 0}

    def generate(_messages):
        calls["n"] += 1
        return (
            "Multi-Index Semantic Hashing learns binary hash codes for document similarity search. "
            "Photosynthesis was not studied."
        )

    result = ask(
        "What does the hashing model learn?",
        _index(),
        load_config("configs/default.yaml"),
        method="bm25",
        generate_answer=True,
        runtime=Runtime(generate=generate),
    )
    assert calls["n"] == 1
    assert result.abstained is False
    assert "paper::p2::c0" in result.answer
    assert "Photosynthesis" not in result.answer


def test_evidence_gate_skips_generation():
    calls = {"n": 0}

    def generate(_messages):
        calls["n"] += 1
        return "should not be called"

    result = ask(
        "What does the hashing model learn?",
        _index(),
        load_config("configs/default.yaml"),
        method="bm25",
        runtime=Runtime(generate=generate),
        thresholds={"bm25": {"enabled": True, "threshold": 10_000.0}},
    )
    assert calls["n"] == 0
    assert result.abstained is True
    assert "pilot-tuned" in (result.abstain_reason or "")


def test_not_found_from_generator_abstains():
    result = ask(
        "What does the hashing model learn?",
        _index(),
        load_config("configs/default.yaml"),
        method="bm25",
        runtime=Runtime(generate=lambda _messages: ABSTAIN_TOKEN),
    )
    assert result.abstained is True


def test_latin_technical_terms_do_not_hide_urdu_words():
    mixed = "BERT کا مخفف Bidirectional Encoder Representations from Transformers ہے۔"
    assert _has_urdu_words(mixed) is True
    assert _has_urdu_words("BERT stands for Bidirectional Encoder Representations from Transformers.") is False
