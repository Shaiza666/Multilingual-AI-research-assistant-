from mira.textnorm import detect_language, normalize_for_index, tokenize


def test_urdu_letter_folding_matches_keyboard_variants():
    arabic = tokenize("كتاب يہ")
    urdu = tokenize("کتاب یہ")
    assert arabic == urdu
    assert arabic[0].startswith("ک")


def test_digits_tatweel_and_bidi_marks():
    folded = normalize_for_index("نمبر ۱۲۳\u0640 ہے\u200f")
    assert "123" in folded
    assert "\u0640" not in folded
    assert "\u200f" not in folded


def test_bari_yeh_is_not_folded_into_farsi_yeh():
    assert "ے" in normalize_for_index("کے")
    assert tokenize("کے") != tokenize("کی")


def test_language_detection():
    assert detect_language("What does MISH stand for?") == "en"
    assert detect_language("MISH کس کا مخفف ہے؟") == "ur"
    assert detect_language("12345") == "unknown"


def test_pdf_urdu_repairs_presentation_and_cmap_errors():
    from mira.textnorm import normalize_display

    broken = "کمپیوڻر ص\u01c0\u061dحیت ǂعداد \u0208"
    repaired = normalize_display(broken)
    assert "کمپیوٹر" in repaired
    assert "صلاحیت" in repaired
    assert "لاعداد" in repaired or "لا" in repaired
    assert "\u06c2" in repaired
    assert "\u01c0" not in repaired
    assert "\ufb50" not in normalize_display("ﭘ")
    assert normalize_display("ﭘ") == "پ"


def test_display_text_keeps_arabic_kaf():
    from mira.textnorm import normalize_display

    assert "ك" in normalize_display("كتاب")
