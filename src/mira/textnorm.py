"""Unicode normalisation, Urdu folding, language detection, and tokenisation.

Arabic-script text is treated as Urdu, which is the second language of this
project. Display text keeps the original letters. The index form folds a few
keyboard variants so that Arabic kaf/yeh still match Urdu kaf/yeh.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic yeh / alef maksura -> Farsi yeh; Arabic kaf -> keheh; hamza-alefs -> alef.
# Arabic heh -> heh goal. Bari yeh (ے) and heh doachashmee (ھ) are kept.
_FOLD = str.maketrans(
    {
        "\u064a": "\u06cc",
        "\u0649": "\u06cc",
        "\u0643": "\u06a9",
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0671": "\u0627",
        "\u0647": "\u06c1",
    }
)
_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
_BIDI = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_TOKEN = re.compile(r"[A-Za-z0-9]+|[\u0600-\u06FF]+")
_EDGE_PUNCT = ".,;:!?()[]{}\"'`،؛؟۔“”«»…/\\-–—"

_EN_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "to",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "with",
        "that",
        "this",
        "it",
        "as",
        "at",
        "from",
        "which",
        "into",
        "their",
        "its",
        "can",
        "may",
        "we",
        "our",
        "than",
    }
)
_UR_STOP = frozenset(
    {
        "کا",
        "کے",
        "کی",
        "ہے",
        "ہیں",
        "کو",
        "سے",
        "میں",
        "اور",
        "یہ",
        "وہ",
        "پر",
        "نے",
        "ایک",
        "کہ",
        "جو",
        "تو",
        "بھی",
        "نہیں",
        "ہو",
        "یا",
        "تک",
        "ہی",
        "اس",
        "ان",
    }
)
STOPWORDS = _EN_STOP | _UR_STOP


def _fold_digits(text: str) -> str:
    chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x06F0 <= code <= 0x06F9:
            chars.append(chr(ord("0") + code - 0x06F0))
        elif 0x0660 <= code <= 0x0669:
            chars.append(chr(ord("0") + code - 0x0660))
        else:
            chars.append(ch)
    return "".join(chars)


def normalize_display(text: str) -> str:
    """Clean text for storage and display without folding Urdu letters."""
    # NFKC turns Arabic presentation forms from PDF extraction back into letters.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    text = _BIDI.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\u0640", "")
    text = re.sub(r"[\u2000-\u200b]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_for_index(text: str) -> str:
    """Fold Urdu keyboard variants and digits so retrieval can match them."""
    text = normalize_display(text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_FOLD)
    text = _fold_digits(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def script_counts(text: str) -> tuple[int, int]:
    latin = 0
    arabic = 0
    for ch in text:
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            latin += 1
        elif "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f":
            arabic += 1
    return latin, arabic


def detect_language(text: str) -> str:
    """Return ``en``, ``ur``, or ``unknown``. Arabic script is labelled Urdu."""
    latin, arabic = script_counts(text)
    total = latin + arabic
    if total == 0:
        return "unknown"
    if arabic >= 8 and arabic / total >= 0.25:
        return "ur"
    if arabic > latin:
        return "ur"
    return "en"


def tokenize(text: str) -> list[str]:
    folded = normalize_for_index(text)
    tokens: list[str] = []
    for raw in _TOKEN.findall(folded):
        tok = raw.strip(_EDGE_PUNCT)
        if not tok:
            continue
        if any("\u0600" <= ch <= "\u06ff" for ch in tok):
            tok = tok.strip("\u060c\u061b\u061f\u06d4")
        if not tok:
            continue
        tokens.append(tok)
    return tokens


def content_tokens(text: str) -> list[str]:
    kept: list[str] = []
    for tok in tokenize(text):
        if tok in STOPWORDS:
            continue
        if tok.isascii() and tok.isalpha() and len(tok) <= 1:
            continue
        kept.append(tok)
    return kept
