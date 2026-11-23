"""Detect permissive Creative Commons statements printed on a paper."""

from __future__ import annotations

import re


def classify_license(text: str) -> str | None:
    """Return a permissive licence name, or None when the text does not grant one.

    CC BY and CC BY-SA are accepted. Non-commercial and no-derivatives licences
    are not. A restrictive URL suppresses a looser phrase unless a plain BY or
    BY-SA 4.0 URL is also present.
    """
    lowered = text.lower().replace("–", "-").replace("—", "-")
    permissive = [
        (r"creativecommons\.org/licenses/by-sa/4\.0", "CC BY-SA 4.0"),
        (r"creativecommons\.org/licenses/by/4\.0", "CC BY 4.0"),
        (r"cc-by-sa 4\.0", "CC BY-SA 4.0"),
        (r"cc-by 4\.0", "CC BY 4.0"),
        (r"creative commons attribution-sharealike 4\.0", "CC BY-SA 4.0"),
        (r"creative commons attribution 4\.0", "CC BY 4.0"),
    ]
    restrictive = re.search(r"creativecommons\.org/licenses/by-(nc|nd)", lowered)
    plain = (
        "creativecommons.org/licenses/by/4.0" in lowered
        or "creativecommons.org/licenses/by-sa/4.0" in lowered
    )
    if restrictive and not plain:
        return None
    for pattern, name in permissive:
        if re.search(pattern, lowered):
            return name
    return None
