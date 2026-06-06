from __future__ import annotations

from dataclasses import dataclass
import re


THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
PREFIX_PATTERNS = [
    r"answer",
    r"final answer",
    r"the answer is",
    r"คำตอบคือ",
    r"คำตอบ",
    r"ตอบ",
]
UNUSABLE_PATTERN = re.compile(r"(cannot|can't|unable|sorry|ไม่สามารถ|ขอโทษ)", re.IGNORECASE)


@dataclass(frozen=True)
class CleanedAnswer:
    answer: str
    used_fallback: bool


def _candidate_lines(raw_text: str) -> list[str]:
    lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in {"```", "```text", "```markdown", "```python"}:
            continue
        lines.append(stripped)
    if not lines and raw_text.strip():
        lines.append(raw_text.strip())
    return lines


def _remove_prefix(text: str) -> str:
    cleaned = text
    for prefix in PREFIX_PATTERNS:
        cleaned = re.sub(rf"^\s*{prefix}\s*[:：\-]?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _clean_line(line: str) -> str:
    cleaned = line.strip().strip("`")
    cleaned = _remove_prefix(cleaned)
    cleaned = cleaned.strip().strip("\"'“”‘’")
    cleaned = cleaned.rstrip(".,;:。")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.translate(THAI_DIGITS)
    return cleaned.strip()


def clean_model_answer(raw_output: object, fallback_answer: str) -> CleanedAnswer:
    raw_text = "" if raw_output is None else str(raw_output)
    for line in _candidate_lines(raw_text):
        cleaned = _clean_line(line)
        if cleaned and not UNUSABLE_PATTERN.search(cleaned):
            return CleanedAnswer(answer=cleaned, used_fallback=False)
    return CleanedAnswer(answer=str(fallback_answer), used_fallback=True)
