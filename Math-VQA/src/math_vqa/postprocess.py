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
FINAL_ANSWER_PATTERNS = [
    re.compile(
        r"(?:final answer)(?:\s+in\s+this\s+case)?\s+is\s*(?:[:：]|\s+-\s+)?\s*[\"'“”‘’]*(?P<answer>[^\n;。]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:final answer|the answer is|answer is|คำตอบคือ|ตอบ)\s*(?:[:：]|\s+-\s+)?\s*[\"'“”‘’]*(?P<answer>[^\n;。]+)",
        re.IGNORECASE,
    ),
]
UNUSABLE_PATTERN = re.compile(
    r"("
    r"cannot|can't|unable|sorry|"
    r"too small|blurry|not clear|not possible|"
    r"provide a clearer|please provide|does not provide|no math problem|"
    r"ไม่สามารถ|ขอโทษ|ไม่ชัด"
    r")",
    re.IGNORECASE,
)


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
    cleaned = cleaned.strip().strip("\"'“”‘’")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.translate(THAI_DIGITS)
    return cleaned.strip()


def _looks_like_short_answer(text: str) -> bool:
    if not text:
        return False
    if text.lower() in {"is", "answer", "final answer", "คำตอบ", "ตอบ"}:
        return False
    if re.match(r"^\d+\.\s+\D", text):
        return False
    if UNUSABLE_PATTERN.search(text):
        return False
    if len(text) > 80:
        return False
    if len(text.split()) > 12:
        return False
    return True


def _trim_answer_span(text: str) -> str:
    return re.split(r"(?<=[0-9A-Za-z๐-๙\"'”’])\.\s+", text.strip(), maxsplit=1)[0]


def _extract_declared_final_answer(raw_text: str) -> str | None:
    for pattern in FINAL_ANSWER_PATTERNS:
        for match in pattern.finditer(raw_text):
            cleaned = _clean_line(_trim_answer_span(match.group("answer")))
            if _looks_like_short_answer(cleaned):
                return cleaned
    return None


def clean_model_answer(raw_output: object, fallback_answer: str) -> CleanedAnswer:
    raw_text = "" if raw_output is None else str(raw_output)
    declared_final_answer = _extract_declared_final_answer(raw_text)
    if declared_final_answer is not None:
        return CleanedAnswer(answer=declared_final_answer, used_fallback=False)
    for line in _candidate_lines(raw_text):
        cleaned = _clean_line(line)
        if _looks_like_short_answer(cleaned):
            return CleanedAnswer(answer=cleaned, used_fallback=False)
    return CleanedAnswer(answer=str(fallback_answer), used_fallback=True)
