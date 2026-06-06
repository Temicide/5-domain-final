from __future__ import annotations

import re
import unicodedata


FILLERS = ("เอ่อ", "อืม", "อ่า", "อะ", "แบบว่า")
POLICIES = {"raw", "single_space", "no_spaces", "thai_chars_only_light", "remove_fillers"}


def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch.isspace())


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _light_allowed_chars(text: str) -> str:
    kept = []
    for ch in text:
        codepoint = ord(ch)
        is_thai = 0x0E00 <= codepoint <= 0x0E7F
        is_ascii_letter = "A" <= ch <= "Z" or "a" <= ch <= "z"
        is_digit = ch.isdigit()
        if is_thai or is_ascii_letter or is_digit or ch.isspace():
            kept.append(ch)
        else:
            kept.append(" ")
    return _collapse_spaces("".join(kept))


def normalize_text(text: str, policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError(f"Unknown normalization policy: {policy}")

    cleaned = _strip_control_chars(str(text)).strip()
    if policy == "raw":
        return cleaned
    if policy == "single_space":
        return _collapse_spaces(cleaned)
    if policy == "no_spaces":
        return re.sub(r"\s+", "", cleaned)
    if policy == "thai_chars_only_light":
        return _light_allowed_chars(cleaned)
    if policy == "remove_fillers":
        without_fillers = cleaned
        for filler in FILLERS:
            without_fillers = re.sub(rf"(^|\s){re.escape(filler)}(?=\s|$)", " ", without_fillers)
        return _collapse_spaces(without_fillers)

    raise ValueError(f"Unknown normalization policy: {policy}")
