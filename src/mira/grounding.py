"""Citation checks and sentence-level grounding.

A sentence is supported when enough of its content words occur in a retrieved
passage, or when its embedding is close to that passage. Unsupported sentences
are reported and removed from the answer that is shown. If nothing remains,
the system abstains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mira.textnorm import content_tokens

CITE_RE = re.compile(r"\[([A-Za-z0-9.\-]+::p\d+::c\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?\u06d4])\s+")
ABSTAIN_TOKEN = "NOT_FOUND"

_ABSTAIN_PHRASES = (
    "not found in the documents",
    "not found in the provided",
    "not in the documents",
    "دستاویزات میں نہیں",
    "جواب نہیں ملا",
)


@dataclass
class SentenceJudgement:
    sentence: str
    supported: bool
    best_chunk_id: str | None
    lexical: float
    embedding: float | None
    model_citations: list[str] = field(default_factory=list)


@dataclass
class GroundingResult:
    answer: str
    english_answer: str
    abstained: bool
    reason: str | None
    judgements: list[SentenceJudgement]
    dropped: list[SentenceJudgement]

    @property
    def citation_ids(self) -> list[str]:
        return CITE_RE.findall(self.answer)

    def unsupported_rate(self, judgements: list[SentenceJudgement] | None = None) -> float | None:
        rows = self.judgements if judgements is None else judgements
        if not rows:
            return None
        bad = sum(1 for row in rows if not row.supported)
        return bad / len(rows)


def is_abstain_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    first = stripped.splitlines()[0].strip().lower()
    compact = re.sub(r"[^a-z_]+", "", first)
    if compact.startswith("not_found") or compact == "notfound":
        rest = stripped.splitlines()[1:]
        rest_text = " ".join(rest).strip()
        if not rest_text or is_abstain_text(rest_text):
            return True
        if len(rest_text) < 8:
            return True
    lowered = stripped.lower()
    return any(phrase in lowered for phrase in _ABSTAIN_PHRASES) and len(stripped) < 280


def split_sentences(text: str) -> list[str]:
    pieces = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()]
    return pieces


def lexical_support(claim: str, evidence: str) -> float:
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(content_tokens(evidence))
    hits = sum(1 for token in claim_tokens if token in evidence_tokens)
    return hits / len(claim_tokens)


def is_supported(lexical: float, embedding: float | None, lexical_threshold: float, embedding_threshold: float) -> bool:
    if lexical >= lexical_threshold:
        return True
    if embedding is not None and embedding >= embedding_threshold:
        return True
    return False


def strip_citations(text: str) -> str:
    return re.sub(r"\s+", " ", CITE_RE.sub(" ", text)).strip()


def parse_citation_ids(text: str) -> list[str]:
    return CITE_RE.findall(text)


def judge_sentences(
    answer: str,
    passages: dict[str, str],
    lexical_threshold: float,
    embedding_threshold: float,
    embed_claim: dict[str, float] | None = None,
) -> list[SentenceJudgement]:
    """Score each sentence against the retrieved passages.

    ``embed_claim`` maps a chunk id to the cosine similarity of that chunk with
    the sentence currently being judged when the caller computes it outside.
    This function recomputes embedding scores per sentence when ``embed_fn`` is
    not used; pass ``chunk_embeddings`` via ``score_sentence`` instead.
    """
    del embed_claim
    judgements: list[SentenceJudgement] = []
    for sentence in split_sentences(answer):
        if is_abstain_text(sentence) and len(split_sentences(answer)) == 1:
            continue
        bare = strip_citations(sentence)
        if not content_tokens(bare):
            continue
        model_citations = parse_citation_ids(sentence)
        best_id: str | None = None
        best_lex = 0.0
        for chunk_id, text in passages.items():
            score = lexical_support(bare, text)
            if score > best_lex:
                best_lex = score
                best_id = chunk_id
        judgements.append(
            SentenceJudgement(
                sentence=bare,
                supported=is_supported(best_lex, None, lexical_threshold, embedding_threshold),
                best_chunk_id=best_id,
                lexical=best_lex,
                embedding=None,
                model_citations=model_citations,
            )
        )
    return judgements


def score_sentence(
    sentence: str,
    passages: dict[str, str],
    lexical_threshold: float,
    embedding_threshold: float,
    embedding_scores: dict[str, float] | None = None,
) -> SentenceJudgement:
    bare = strip_citations(sentence)
    best_id: str | None = None
    best_lex = -1.0
    best_emb: float | None = None
    best_combined = -1.0
    for chunk_id, text in passages.items():
        lex = lexical_support(bare, text)
        emb = None if not embedding_scores else embedding_scores.get(chunk_id)
        combined = lex
        if emb is not None:
            combined = max(combined, emb)
        if combined > best_combined:
            best_combined = combined
            best_lex = lex
            best_emb = emb
            best_id = chunk_id
    if best_lex < 0:
        best_lex = 0.0
    return SentenceJudgement(
        sentence=bare,
        supported=is_supported(best_lex, best_emb, lexical_threshold, embedding_threshold),
        best_chunk_id=best_id,
        lexical=best_lex,
        embedding=best_emb,
        model_citations=parse_citation_ids(sentence),
    )


def ground_answer(
    raw_answer: str,
    passages: dict[str, str],
    lexical_threshold: float = 0.35,
    embedding_threshold: float = 0.82,
    embedding_matrix: dict[str, list[float]] | None = None,
    embed_text=None,
) -> GroundingResult:
    """Keep supported sentences and attach a verified chunk id to each one.

    ``embed_text`` maps a claim string to a unit vector. ``embedding_matrix``
    maps chunk ids to unit vectors stored at index time.
    """
    if is_abstain_text(raw_answer):
        return GroundingResult(
            answer="",
            english_answer="",
            abstained=True,
            reason="model abstained",
            judgements=[],
            dropped=[],
        )
    judgements: list[SentenceJudgement] = []
    for sentence in split_sentences(raw_answer):
        bare = strip_citations(sentence)
        if not content_tokens(bare):
            continue
        embedding_scores = None
        if embed_text is not None and embedding_matrix:
            vector = embed_text(bare)
            embedding_scores = {}
            for chunk_id, stored in embedding_matrix.items():
                embedding_scores[chunk_id] = _dot(vector, stored)
        judgements.append(
            score_sentence(
                sentence,
                passages,
                lexical_threshold,
                embedding_threshold,
                embedding_scores,
            )
        )
    if not judgements:
        return GroundingResult(
            answer="",
            english_answer="",
            abstained=True,
            reason="no content sentences",
            judgements=[],
            dropped=[],
        )
    kept: list[str] = []
    dropped: list[SentenceJudgement] = []
    for item in judgements:
        if item.supported and item.best_chunk_id:
            kept.append(f"{item.sentence} [{item.best_chunk_id}]")
        else:
            dropped.append(item)
    if not kept:
        return GroundingResult(
            answer="",
            english_answer="",
            abstained=True,
            reason="no sentence was supported by the retrieved passages",
            judgements=judgements,
            dropped=dropped,
        )
    text = " ".join(kept)
    return GroundingResult(
        answer=text,
        english_answer=text,
        abstained=False,
        reason=None,
        judgements=judgements,
        dropped=dropped,
    )


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def citation_existence_precision(answer: str, known_ids: set[str]) -> float | None:
    cited = parse_citation_ids(answer)
    if not cited:
        return None
    return sum(1 for cid in cited if cid in known_ids) / len(cited)


def citation_page_precision(answer: str, chunk_pages: dict[str, tuple[str, int]], gold_pages: set[tuple[str, int]]) -> float | None:
    cited = parse_citation_ids(answer)
    if not cited:
        return None
    hits = 0
    for cid in cited:
        page = chunk_pages.get(cid)
        if page is not None and page in gold_pages:
            hits += 1
    return hits / len(cited)
