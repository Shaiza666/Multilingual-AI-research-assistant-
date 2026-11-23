from mira.grounding import (
    citation_existence_precision,
    citation_page_precision,
    ground_answer,
    is_supported,
    lexical_support,
)


def test_supported_sentence_is_kept_and_cited():
    passages = {
        "paper::p1::c0": "Batch normalization uses the mean and variance of the current mini-batch.",
    }
    raw = (
        "Batch normalization uses the mean and variance of the current mini-batch. "
        "The moon is made of cheese."
    )
    result = ground_answer(raw, passages, lexical_threshold=0.35, embedding_threshold=0.82)
    assert result.abstained is False
    assert "paper::p1::c0" in result.answer
    assert "moon" not in result.answer.lower()
    assert len(result.dropped) == 1


def test_abstain_token_and_unsupported_answer():
    passages = {"paper::p1::c0": "Hash codes are binary vectors."}
    assert ground_answer("NOT_FOUND", passages).abstained is True
    refused = ground_answer("The experiment measured photosynthesis in spinach.", passages)
    assert refused.abstained is True
    assert refused.reason == "no sentence was supported by the retrieved passages"


def test_lexical_support_and_thresholds():
    evidence = "MISH stands for Multi-Index Semantic Hashing and is unsupervised."
    claim = "MISH stands for Multi-Index Semantic Hashing."
    score = lexical_support(claim, evidence)
    assert score > 0.6
    assert is_supported(score, None, 0.35, 0.82) is True
    assert is_supported(0.1, 0.9, 0.35, 0.82) is True
    assert is_supported(0.1, 0.5, 0.35, 0.82) is False


def test_citation_precision_counts_hallucinated_ids():
    answer = "A fact. [paper::p1::c0] Another. [missing::p9::c3]"
    known = {"paper::p1::c0"}
    pages = {"paper::p1::c0": ("paper", 1)}
    assert citation_existence_precision(answer, known) == 0.5
    assert citation_page_precision(answer, pages, {("paper", 1)}) == 0.5
