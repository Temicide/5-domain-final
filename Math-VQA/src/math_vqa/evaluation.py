from __future__ import annotations

import re
from typing import Sequence


THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
UNIT_WORDS = [
    "square centimeters",
    "ตารางเซนติเมตร",
    "ลูกบาศก์หน่วย",
    "เซนติเมตร",
    "degrees",
    "years old",
    "ร้อยละ",
    "ดอลลาร์",
    "องศา",
    "หน่วย",
    "จำนวน",
    "วิธี",
    "แบบ",
    "ค่า",
    "บาท",
]
LATEX_REPLACEMENTS = {
    r"\pi": "pi",
    r"\times": "*",
    r"\cdot": "*",
    r"\div": "/",
    r"\pm": "+-",
    r"\left": "",
    r"\right": "",
    r"\,": "",
    r"\;": "",
    r"\:": "",
    r"\!": "",
}


def normalize_answer(value: object) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip().translate(THAI_DIGITS)
    text = text.replace("$", "")
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", lambda m: f"{m.group(1)}/{m.group(2)}", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", lambda m: f"sqrt{m.group(1)}", text)
    for macro in ("overrightarrow", "overline", "vec"):
        text = re.sub(rf"\\{macro}\s*\{{([^{{}}]+)\}}", r"\1", text)
    for source, replacement in LATEX_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    for unit in sorted(UNIT_WORDS, key=len, reverse=True):
        text = re.sub(re.escape(unit), "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[{}\\,]", "", text)
    text = re.sub(r"\(([-+]?\d+)\)", r"\1", text)
    text = re.sub(r"(?<![a-z0-9])([-+]?\d+)\.0+(?![a-z0-9])", r"\1", text)
    return text


def normalized_accuracy(predictions: Sequence[object], truths: Sequence[object]) -> float:
    if len(predictions) != len(truths):
        raise ValueError("predictions and truths must have the same length")
    if len(predictions) == 0:
        return 0.0
    matches = sum(
        normalize_answer(prediction) == normalize_answer(truth)
        for prediction, truth in zip(predictions, truths, strict=True)
    )
    return matches / len(predictions)
