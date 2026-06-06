# Thai Math VQA Kaggle Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, Kaggle-ready LLaVA/Ollama notebook that reads Thai/English math problem images and writes validated `submission.csv` and `raw_predictions.csv`.

**Architecture:** Put reusable logic in a small `Math-VQA/src/math_vqa` Python package, then generate a self-contained Kaggle notebook from those modules. The notebook materializes the tested package code inside `/kaggle/working`, runs conservative preprocessing, calls Ollama LLaVA through HTTP, postprocesses short answers, validates the submission contract, and records raw predictions for review.

**Tech Stack:** Python 3.10+, pandas, Pillow, requests, tqdm, pytest, nbformat, Ollama LLaVA.

---

## Scope Check

The spec describes one cohesive subsystem: a Kaggle inference pipeline for Thai Math VQA. It does not need to be split into multiple plans because each task below contributes to one testable notebook deliverable.

## Source References

- Spec: `Math-VQA/spec.md`
- Ollama Linux install command: `https://docs.ollama.com/linux`
- Ollama LLaVA model page and model name: `https://ollama.com/library/llava`

## File Structure

- Create `Math-VQA/pyproject.toml`: local package metadata, runtime dependencies, pytest configuration.
- Create `Math-VQA/README.md`: commands for local tests, notebook generation, Kaggle use, and Hugging Face fallback documentation.
- Create `Math-VQA/src/math_vqa/__init__.py`: package marker and version.
- Create `Math-VQA/src/math_vqa/data.py`: competition path discovery, CSV loading, image path resolution, data integrity checks, fallback answer prior.
- Create `Math-VQA/src/math_vqa/evaluation.py`: Kaggle-style answer normalization and local normalized accuracy.
- Create `Math-VQA/src/math_vqa/postprocess.py`: raw model output cleanup and fallback tracking.
- Create `Math-VQA/src/math_vqa/preprocessing.py`: RGB image loading, conservative preprocessing variants, image-id based variant selection, saving preprocessed images for Ollama.
- Create `Math-VQA/src/math_vqa/prompts.py`: base, diagram, and strict-format prompt templates plus prompt selection.
- Create `Math-VQA/src/math_vqa/ollama_client.py`: Ollama readiness checks and LLaVA image query client.
- Create `Math-VQA/src/math_vqa/submission.py`: prediction record model, submission/raw-log dataframe construction, validation, output writing.
- Create `Math-VQA/scripts/build_notebook.py`: builds `Math-VQA/notebooks/thai_math_vqa_ollama_llava.ipynb` with embedded package source and Kaggle execution cells.
- Create `Math-VQA/notebooks/thai_math_vqa_ollama_llava.ipynb`: generated notebook artifact.
- Create `Math-VQA/tests/test_data_contracts.py`: unit tests for CSV contracts and local image path resolution.
- Create `Math-VQA/tests/test_evaluation.py`: unit tests for Kaggle-style normalization and accuracy.
- Create `Math-VQA/tests/test_postprocess.py`: unit tests for model output cleanup and fallback behavior.
- Create `Math-VQA/tests/test_preprocessing.py`: unit tests for RGB loading, aspect ratio preservation, preprocessing selectors, and saved images.
- Create `Math-VQA/tests/test_prompts_and_ollama.py`: unit tests for prompt selection and Ollama HTTP payloads.
- Create `Math-VQA/tests/test_submission.py`: unit tests for submission ordering, required columns, non-empty answers, and raw prediction logs.
- Create `Math-VQA/tests/test_notebook_build.py`: unit test that builds the notebook and verifies required Kaggle sections and outputs are present.

### Task 0: Project Scaffold

**Files:**
- Create: `Math-VQA/pyproject.toml`
- Create: `Math-VQA/src/math_vqa/__init__.py`

- [ ] **Step 1: Create package metadata**

Create `Math-VQA/pyproject.toml`:

```toml
[project]
name = "math-vqa"
version = "0.1.0"
description = "Kaggle-ready Thai Math VQA inference pipeline"
requires-python = ">=3.10"
dependencies = [
  "nbformat>=5.9",
  "pandas>=2.0",
  "pillow>=10.0",
  "requests>=2.31",
  "tqdm>=4.66"
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0"
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Create package marker**

Create `Math-VQA/src/math_vqa/__init__.py`:

```python
"""Thai Math VQA Kaggle inference helpers."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Install editable package locally**

Run:

```bash
cd Math-VQA
python -m pip install -e ".[dev]"
```

Expected: package installs successfully and pip reports `Successfully installed math-vqa`.

- [ ] **Step 4: Verify pytest is available**

Run:

```bash
cd Math-VQA
python -m pytest --version
```

Expected: prints a pytest version and exits with code 0.

- [ ] **Step 5: Commit**

```bash
git add Math-VQA/pyproject.toml Math-VQA/src/math_vqa/__init__.py
git commit -m "chore: scaffold math vqa package"
```

### Task 1: Data Contracts

**Files:**
- Create: `Math-VQA/tests/test_data_contracts.py`
- Create: `Math-VQA/src/math_vqa/data.py`

- [ ] **Step 1: Write failing data contract tests**

Create `Math-VQA/tests/test_data_contracts.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from math_vqa.data import (
    CompetitionPaths,
    answer_prior,
    load_competition_frames,
    resolve_image_path,
    validate_data_files,
)


def make_competition_root(tmp_path: Path) -> CompetitionPaths:
    root = tmp_path / "competition"
    (root / "images" / "images").mkdir(parents=True)
    (root / "images" / "images" / "0.jpg").write_bytes(b"image-0")
    (root / "images" / "images" / "1.jpg").write_bytes(b"image-1")
    (root / "images" / "images" / "2.jpg").write_bytes(b"image-2")
    (root / "train.csv").write_text(
        "id,image_path,answer\n0,images/0.jpg,๒\n2,images/2.jpg,2\n",
        encoding="utf-8",
    )
    (root / "test.csv").write_text(
        "id,image_path\n1,images/1.jpg\n",
        encoding="utf-8",
    )
    (root / "sample_submission.csv").write_text(
        "id,answer\n1,2\n",
        encoding="utf-8",
    )
    return CompetitionPaths(root)


def test_load_competition_frames_casts_ids_to_strings(tmp_path: Path) -> None:
    paths = make_competition_root(tmp_path)

    train_df, test_df, sample_df = load_competition_frames(paths)

    assert train_df.loc[0, "id"] == "0"
    assert test_df.loc[0, "id"] == "1"
    assert sample_df.loc[0, "id"] == "1"


def test_validate_data_files_accepts_expected_contract(tmp_path: Path) -> None:
    paths = make_competition_root(tmp_path)
    train_df, test_df, sample_df = load_competition_frames(paths)

    validate_data_files(paths, train_df, test_df, sample_df)


def test_validate_data_files_rejects_missing_image(tmp_path: Path) -> None:
    paths = make_competition_root(tmp_path)
    (paths.root / "images" / "images" / "1.jpg").unlink()
    train_df, test_df, sample_df = load_competition_frames(paths)

    with pytest.raises(ValueError, match="missing image"):
        validate_data_files(paths, train_df, test_df, sample_df)


def test_resolve_image_path_matches_nested_kaggle_image_dir(tmp_path: Path) -> None:
    paths = make_competition_root(tmp_path)

    resolved = resolve_image_path(paths, "images/1.jpg")

    assert resolved == paths.root / "images" / "images" / "1.jpg"


def test_answer_prior_prefers_most_common_simple_answer() -> None:
    train_df = pd.DataFrame({"answer": ["20 ตารางเซนติเมตร", "๒", "2", "คำตอบ"]})

    assert answer_prior(train_df) == "2"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_data_contracts.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'math_vqa.data'`.

- [ ] **Step 3: Implement data contracts**

Create `Math-VQA/src/math_vqa/data.py`:

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


COMPETITION_DIR_NAME = "super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


@dataclass(frozen=True)
class CompetitionPaths:
    root: Path

    @property
    def train_csv(self) -> Path:
        return self.root / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.root / "test.csv"

    @property
    def sample_submission_csv(self) -> Path:
        return self.root / "sample_submission.csv"

    @classmethod
    def auto(cls) -> "CompetitionPaths":
        kaggle_root = Path("/kaggle/input/competitions") / COMPETITION_DIR_NAME
        if kaggle_root.exists():
            return cls(kaggle_root)
        local_root = Path(__file__).resolve().parents[2] / "data" / COMPETITION_DIR_NAME
        return cls(local_root)


def load_competition_frames(paths: CompetitionPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(paths.train_csv, dtype={"id": str, "image_path": str, "answer": str}, keep_default_na=False)
    test_df = pd.read_csv(paths.test_csv, dtype={"id": str, "image_path": str}, keep_default_na=False)
    sample_df = pd.read_csv(paths.sample_submission_csv, dtype={"id": str, "answer": str}, keep_default_na=False)
    return train_df, test_df, sample_df


def resolve_image_path(paths: CompetitionPaths, image_path: str | Path) -> Path:
    image_path = Path(image_path)
    if image_path.is_absolute() and image_path.exists():
        return image_path
    candidates = [
        paths.root / image_path,
        paths.root / "images" / image_path,
        paths.root / "images" / "images" / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_data_files(
    paths: CompetitionPaths,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> None:
    if not paths.root.exists():
        raise ValueError(f"data root does not exist: {paths.root}")
    for csv_path in (paths.train_csv, paths.test_csv, paths.sample_submission_csv):
        if not csv_path.exists():
            raise ValueError(f"missing csv file: {csv_path}")

    expected_columns = {
        "train.csv": (train_df, ["id", "image_path", "answer"]),
        "test.csv": (test_df, ["id", "image_path"]),
        "sample_submission.csv": (sample_df, ["id", "answer"]),
    }
    for name, (frame, columns) in expected_columns.items():
        if list(frame.columns) != columns:
            raise ValueError(f"{name} columns must be exactly {columns}, got {list(frame.columns)}")
        if frame["id"].duplicated().any():
            duplicated = frame.loc[frame["id"].duplicated(), "id"].tolist()
            raise ValueError(f"{name} has duplicate ids: {duplicated[:5]}")

    if set(train_df["id"]) & set(test_df["id"]):
        raise ValueError("train and test ids must not overlap")
    if set(sample_df["id"]) != set(test_df["id"]):
        raise ValueError("sample_submission ids must match test ids")

    missing_images: list[str] = []
    for frame_name, frame in (("train.csv", train_df), ("test.csv", test_df)):
        for row in frame.itertuples(index=False):
            resolved = resolve_image_path(paths, row.image_path)
            if not resolved.exists():
                missing_images.append(f"{frame_name}:{row.id}:{row.image_path}")
    if missing_images:
        raise ValueError(f"missing image files: {missing_images[:10]}")


def answer_prior(train_df: pd.DataFrame) -> str:
    answers = [str(answer).strip() for answer in train_df["answer"].tolist() if str(answer).strip()]
    simple_answers = [
        answer.translate(THAI_DIGITS)
        for answer in answers
        if re.fullmatch(r"[0-9๐-๙]+", answer)
    ]
    if simple_answers:
        return Counter(simple_answers).most_common(1)[0][0]
    if answers:
        return Counter(answers).most_common(1)[0][0]
    return "2"
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_data_contracts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Math-VQA/src/math_vqa/data.py Math-VQA/tests/test_data_contracts.py
git commit -m "feat: add math vqa data contracts"
```

### Task 2: Kaggle Answer Normalization

**Files:**
- Create: `Math-VQA/tests/test_evaluation.py`
- Create: `Math-VQA/src/math_vqa/evaluation.py`

- [ ] **Step 1: Write failing normalization tests**

Create `Math-VQA/tests/test_evaluation.py`:

```python
import pytest

from math_vqa.evaluation import normalize_answer, normalized_accuracy


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("20 ตารางเซนติเมตร", "20"),
        ("30 องศา", "30"),
        (r"$6\sqrt{3}$", "6sqrt3"),
        (r"$\frac{17}{10}$", "17/10"),
        ("๒๕", "25"),
        ("2.0", "2"),
        (r"\overline{AB}", "ab"),
        (r"3 \times 4", "3*4"),
    ],
)
def test_normalize_answer_matches_competition_examples(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_normalized_accuracy_compares_after_normalization() -> None:
    predictions = ["20 ตารางเซนติเมตร", "๒๕", "4"]
    truths = ["20", "25", "5"]

    assert normalized_accuracy(predictions, truths) == pytest.approx(2 / 3)


def test_normalized_accuracy_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        normalized_accuracy(["1"], ["1", "2"])
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_evaluation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'math_vqa.evaluation'`.

- [ ] **Step 3: Implement normalization**

Create `Math-VQA/src/math_vqa/evaluation.py`:

```python
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
    if not predictions:
        return 0.0
    matches = sum(
        normalize_answer(prediction) == normalize_answer(truth)
        for prediction, truth in zip(predictions, truths, strict=True)
    )
    return matches / len(predictions)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_evaluation.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Math-VQA/src/math_vqa/evaluation.py Math-VQA/tests/test_evaluation.py
git commit -m "feat: add kaggle answer normalization"
```

### Task 3: Model Output Postprocessing

**Files:**
- Create: `Math-VQA/tests/test_postprocess.py`
- Create: `Math-VQA/src/math_vqa/postprocess.py`

- [ ] **Step 1: Write failing postprocessing tests**

Create `Math-VQA/tests/test_postprocess.py`:

```python
from math_vqa.postprocess import clean_model_answer


def test_clean_model_answer_removes_prefix_punctuation_and_thai_digits() -> None:
    cleaned = clean_model_answer("Answer: ๒๕.\nExplanation: ignored", fallback_answer="2")

    assert cleaned.answer == "25"
    assert cleaned.used_fallback is False


def test_clean_model_answer_removes_thai_prefix_and_latex_delimiters() -> None:
    cleaned = clean_model_answer("```text\nคำตอบคือ $6\\sqrt{3}$\n```", fallback_answer="2")

    assert cleaned.answer == r"6\sqrt{3}"
    assert cleaned.used_fallback is False


def test_clean_model_answer_uses_fallback_for_empty_output() -> None:
    cleaned = clean_model_answer("   \n```", fallback_answer="2")

    assert cleaned.answer == "2"
    assert cleaned.used_fallback is True


def test_clean_model_answer_uses_fallback_for_refusal_output() -> None:
    cleaned = clean_model_answer("I cannot solve this image.", fallback_answer="1")

    assert cleaned.answer == "1"
    assert cleaned.used_fallback is True
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_postprocess.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'math_vqa.postprocess'`.

- [ ] **Step 3: Implement postprocessing**

Create `Math-VQA/src/math_vqa/postprocess.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_postprocess.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Math-VQA/src/math_vqa/postprocess.py Math-VQA/tests/test_postprocess.py
git commit -m "feat: clean vqa model answers"
```

### Task 4: Conservative Image Preprocessing

**Files:**
- Create: `Math-VQA/tests/test_preprocessing.py`
- Create: `Math-VQA/src/math_vqa/preprocessing.py`

- [ ] **Step 1: Write failing preprocessing tests**

Create `Math-VQA/tests/test_preprocessing.py`:

```python
from pathlib import Path

from PIL import Image

from math_vqa.preprocessing import (
    load_rgb_image,
    preprocess_image,
    save_preprocessed_image,
    select_preprocess_name,
)


def make_image(path: Path, size: tuple[int, int] = (20, 10), mode: str = "L") -> Path:
    image = Image.new(mode, size, color=200)
    image.save(path)
    return path


def test_load_rgb_image_converts_to_rgb(tmp_path: Path) -> None:
    image_path = make_image(tmp_path / "gray.jpg", mode="L")

    image = load_rgb_image(image_path)

    assert image.mode == "RGB"
    assert image.size == (20, 10)


def test_upscale_preserves_aspect_ratio(tmp_path: Path) -> None:
    image_path = make_image(tmp_path / "small.jpg", size=(20, 10), mode="RGB")

    result = preprocess_image(image_path, "upscale")

    assert result.name == "upscale"
    assert result.image.size == (1024, 512)
    assert result.final_size == (1024, 512)


def test_select_preprocess_name_uses_specified_image_ids() -> None:
    assert select_preprocess_name("94") == "upscale"
    assert select_preprocess_name("156") == "contrast"
    assert select_preprocess_name("451") == "high_res"
    assert select_preprocess_name("7") == "raw"


def test_save_preprocessed_image_writes_rgb_jpeg(tmp_path: Path) -> None:
    image_path = make_image(tmp_path / "raw.jpg", size=(20, 10), mode="RGB")
    result = preprocess_image(image_path, "raw")

    saved = save_preprocessed_image(result, tmp_path / "prepared", image_id="7")

    assert saved.exists()
    assert saved.name == "7_raw.jpg"
    assert Image.open(saved).mode == "RGB"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_preprocessing.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'math_vqa.preprocessing'`.

- [ ] **Step 3: Implement preprocessing**

Create `Math-VQA/src/math_vqa/preprocessing.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance


LOW_RES_IMAGE_IDS = {"94", "101", "134", "140", "162", "200"}
CONTRAST_IMAGE_IDS = {"156"}
HIGH_RES_IMAGE_IDS = {"451", "569"}
RESAMPLE = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)


@dataclass(frozen=True)
class PreprocessResult:
    image: Image.Image
    name: str
    final_size: tuple[int, int]


def load_rgb_image(image_path: str | Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _resize_by_max_side(image: Image.Image, max_side: int, upscale: bool) -> Image.Image:
    width, height = image.size
    current_max = max(width, height)
    if current_max == 0:
        raise ValueError("image has zero-sized dimension")
    if current_max == max_side:
        return image.copy()
    if current_max > max_side or upscale:
        scale = max_side / current_max
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(new_size, RESAMPLE)
    return image.copy()


def preprocess_image(image_path: str | Path, variant: str = "raw") -> PreprocessResult:
    image = load_rgb_image(image_path)
    if variant == "raw":
        processed = image
    elif variant == "upscale":
        processed = _resize_by_max_side(image, max_side=1024, upscale=True)
    elif variant == "contrast":
        processed = ImageEnhance.Contrast(image).enhance(1.6)
    elif variant == "high_res":
        processed = _resize_by_max_side(image, max_side=1568, upscale=False)
    else:
        raise ValueError(f"unknown preprocessing variant: {variant}")
    return PreprocessResult(image=processed, name=variant, final_size=processed.size)


def select_preprocess_name(image_id: object) -> str:
    image_id_text = str(image_id)
    if image_id_text in LOW_RES_IMAGE_IDS:
        return "upscale"
    if image_id_text in CONTRAST_IMAGE_IDS:
        return "contrast"
    if image_id_text in HIGH_RES_IMAGE_IDS:
        return "high_res"
    return "raw"


def save_preprocessed_image(result: PreprocessResult, output_dir: str | Path, image_id: object) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_id}_{result.name}.jpg"
    result.image.save(output_path, format="JPEG", quality=95)
    return output_path
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_preprocessing.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Math-VQA/src/math_vqa/preprocessing.py Math-VQA/tests/test_preprocessing.py
git commit -m "feat: add conservative image preprocessing"
```

### Task 5: Prompts and Ollama Client

**Files:**
- Create: `Math-VQA/tests/test_prompts_and_ollama.py`
- Create: `Math-VQA/src/math_vqa/prompts.py`
- Create: `Math-VQA/src/math_vqa/ollama_client.py`

- [ ] **Step 1: Write failing prompt and Ollama tests**

Create `Math-VQA/tests/test_prompts_and_ollama.py`:

```python
from pathlib import Path

import pytest

from math_vqa.ollama_client import assert_ollama_ready, query_llava
from math_vqa.prompts import build_prompt, select_prompt_name


def test_build_prompt_contains_short_answer_instruction() -> None:
    prompt = build_prompt("base")

    assert "Return only the final answer." in prompt
    assert "Do not explain." in prompt


def test_select_prompt_name_uses_diagram_prompt_for_visual_cases() -> None:
    assert select_prompt_name("451") == "diagram"
    assert select_prompt_name("7") == "base"


def test_query_llava_sends_image_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"abc")
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": "42"}}

    def fake_post(url: str, json: dict, timeout: int) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("math_vqa.ollama_client.requests.post", fake_post)

    response = query_llava(image_path, "Solve this", model="llava")

    assert response == "42"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["model"] == "llava"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"][0]["images"]


def test_assert_ollama_ready_raises_readable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRequests:
        @staticmethod
        def get(url: str, timeout: int) -> object:
            raise OSError("connection refused")

    monkeypatch.setattr("math_vqa.ollama_client.requests", FakeRequests)

    with pytest.raises(RuntimeError, match="Ollama is not ready"):
        assert_ollama_ready()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_prompts_and_ollama.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `math_vqa.ollama_client` or `math_vqa.prompts`.

- [ ] **Step 3: Implement prompt templates**

Create `Math-VQA/src/math_vqa/prompts.py`:

```python
from __future__ import annotations


BASE_PROMPT = """You are solving a Thai/English math problem from an image.
Read all text, diagrams, graphs, shapes, and answer choices carefully.
Return only the final answer.
Do not explain.
Do not include "answer:".
Use Arabic digits when possible."""

DIAGRAM_PROMPT = """You are solving a math problem from an image.
The image may contain Thai text, English text, diagrams, graphs, shapes, or visual answer choices.
Use the visual information, not only OCR text.
Return only the final answer."""

STRICT_FORMAT_PROMPT = """Return exactly one short answer string.
No explanation.
No units unless the unit is necessary to distinguish the answer.
Use plain fractions such as 17/10.
Use sqrt notation for radicals."""

DIAGRAM_IMAGE_IDS = {"451", "569"}
PROMPTS = {
    "base": f"{BASE_PROMPT}\n\n{STRICT_FORMAT_PROMPT}",
    "diagram": f"{DIAGRAM_PROMPT}\n\n{STRICT_FORMAT_PROMPT}",
    "strict": STRICT_FORMAT_PROMPT,
}


def select_prompt_name(image_id: object) -> str:
    if str(image_id) in DIAGRAM_IMAGE_IDS:
        return "diagram"
    return "base"


def build_prompt(prompt_name: str = "base", few_shot_examples: list[dict[str, str]] | None = None) -> str:
    if prompt_name not in PROMPTS:
        raise ValueError(f"unknown prompt name: {prompt_name}")
    prompt = PROMPTS[prompt_name]
    if not few_shot_examples:
        return prompt
    example_lines = ["Examples of accepted final-answer strings:"]
    for example in few_shot_examples:
        example_lines.append(f"- id {example['id']}: {example['answer']}")
    return f"{prompt}\n\n" + "\n".join(example_lines)
```

- [ ] **Step 4: Implement Ollama client**

Create `Math-VQA/src/math_vqa/ollama_client.py`:

```python
from __future__ import annotations

import base64
from pathlib import Path

import requests


DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def encode_image_base64(image_path: str | Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def assert_ollama_ready(host: str = DEFAULT_OLLAMA_HOST, timeout: int = 5) -> None:
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Ollama is not ready at {host}: {exc}") from exc


def query_llava(
    image_path: str | Path,
    prompt: str,
    model: str = "llava",
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: int = 180,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [encode_image_base64(image_path)],
            }
        ],
    }
    response = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    message = body.get("message", {})
    return str(message.get("content", body.get("response", "")))
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_prompts_and_ollama.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Math-VQA/src/math_vqa/prompts.py Math-VQA/src/math_vqa/ollama_client.py Math-VQA/tests/test_prompts_and_ollama.py
git commit -m "feat: add llava prompts and ollama client"
```

### Task 6: Submission Validation and Raw Logs

**Files:**
- Create: `Math-VQA/tests/test_submission.py`
- Create: `Math-VQA/src/math_vqa/submission.py`

- [ ] **Step 1: Write failing submission tests**

Create `Math-VQA/tests/test_submission.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from math_vqa.submission import (
    PredictionRecord,
    build_submission_frames,
    validate_submission,
    write_outputs,
)


def test_build_submission_frames_preserves_sample_order() -> None:
    sample_df = pd.DataFrame({"id": ["2", "1"], "answer": ["2", "2"]})
    records = [
        PredictionRecord("1", "images/1.jpg", "raw one", "1", "base", "raw", False),
        PredictionRecord("2", "images/2.jpg", "raw two", "2", "diagram", "high_res", False),
    ]

    submission_df, raw_df = build_submission_frames(records, sample_df)

    assert submission_df.to_dict("list") == {"id": ["2", "1"], "answer": ["2", "1"]}
    assert list(raw_df.columns) == [
        "id",
        "image_path",
        "raw_prediction",
        "clean_answer",
        "prompt_name",
        "preprocess_name",
        "used_fallback",
    ]


def test_validate_submission_rejects_empty_answer() -> None:
    sample_df = pd.DataFrame({"id": ["1"], "answer": ["2"]})
    submission_df = pd.DataFrame({"id": ["1"], "answer": [" "]})

    with pytest.raises(ValueError, match="empty answers"):
        validate_submission(sample_df, submission_df)


def test_validate_submission_rejects_wrong_order() -> None:
    sample_df = pd.DataFrame({"id": ["1", "2"], "answer": ["2", "2"]})
    submission_df = pd.DataFrame({"id": ["2", "1"], "answer": ["2", "1"]})

    with pytest.raises(ValueError, match="row order"):
        validate_submission(sample_df, submission_df)


def test_write_outputs_writes_submission_and_raw_predictions(tmp_path: Path) -> None:
    sample_df = pd.DataFrame({"id": ["1"], "answer": ["2"]})
    records = [PredictionRecord("1", "images/1.jpg", "raw", "7", "base", "raw", False)]

    submission_path, raw_path = write_outputs(records, sample_df, tmp_path)

    assert submission_path == tmp_path / "submission.csv"
    assert raw_path == tmp_path / "raw_predictions.csv"
    assert submission_path.read_text(encoding="utf-8").startswith("id,answer\n1,7\n")
    assert "raw_prediction" in raw_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_submission.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'math_vqa.submission'`.

- [ ] **Step 3: Implement submission validation and output writing**

Create `Math-VQA/src/math_vqa/submission.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PredictionRecord:
    id: str
    image_path: str
    raw_prediction: str
    clean_answer: str
    prompt_name: str
    preprocess_name: str
    used_fallback: bool


def build_submission_frames(
    records: list[PredictionRecord],
    sample_submission: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_by_id = {str(record.id): record for record in records}
    missing_ids = [str(image_id) for image_id in sample_submission["id"].tolist() if str(image_id) not in record_by_id]
    if missing_ids:
        raise ValueError(f"missing predictions for ids: {missing_ids[:10]}")

    submission_rows = [
        {"id": str(image_id), "answer": str(record_by_id[str(image_id)].clean_answer)}
        for image_id in sample_submission["id"].tolist()
    ]
    raw_rows = [asdict(record) for record in records]
    submission_df = pd.DataFrame(submission_rows, columns=["id", "answer"])
    raw_df = pd.DataFrame(
        raw_rows,
        columns=[
            "id",
            "image_path",
            "raw_prediction",
            "clean_answer",
            "prompt_name",
            "preprocess_name",
            "used_fallback",
        ],
    )
    return submission_df, raw_df


def validate_submission(sample_submission: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["id", "answer"]:
        raise ValueError(f"submission columns must be exactly ['id', 'answer'], got {list(submission.columns)}")
    if len(submission) != len(sample_submission):
        raise ValueError(f"submission row count must be {len(sample_submission)}, got {len(submission)}")
    sample_ids = [str(value) for value in sample_submission["id"].tolist()]
    submission_ids = [str(value) for value in submission["id"].tolist()]
    if submission_ids != sample_ids:
        raise ValueError("submission row order must match sample_submission.csv")
    if len(set(submission_ids)) != len(submission_ids):
        raise ValueError("submission ids must be unique")
    if submission["answer"].isna().any():
        raise ValueError("submission has null answers")
    empty_mask = submission["answer"].astype(str).str.strip().eq("")
    if empty_mask.any():
        empty_ids = submission.loc[empty_mask, "id"].astype(str).tolist()
        raise ValueError(f"submission has empty answers for ids: {empty_ids[:10]}")


def write_outputs(
    records: list[PredictionRecord],
    sample_submission: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_df, raw_df = build_submission_frames(records, sample_submission)
    validate_submission(sample_submission, submission_df)
    submission_path = output_dir / "submission.csv"
    raw_path = output_dir / "raw_predictions.csv"
    submission_df.to_csv(submission_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    return submission_path, raw_path
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_submission.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Math-VQA/src/math_vqa/submission.py Math-VQA/tests/test_submission.py
git commit -m "feat: validate math vqa submissions"
```

### Task 7: Self-Contained Kaggle Notebook Builder

**Files:**
- Create: `Math-VQA/tests/test_notebook_build.py`
- Create: `Math-VQA/scripts/build_notebook.py`
- Create: `Math-VQA/notebooks/thai_math_vqa_ollama_llava.ipynb`

- [ ] **Step 1: Write failing notebook builder test**

Create `Math-VQA/tests/test_notebook_build.py`:

```python
from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_notebook_creates_required_kaggle_sections() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_notebook.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    notebook_path = PROJECT_ROOT / "notebooks" / "thai_math_vqa_ollama_llava.ipynb"
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "\n".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "curl -fsSL https://ollama.com/install.sh | sh" in source
    assert "ollama pull" in source
    assert "MATERIALIZED_MODULES" in source
    assert "/kaggle/working/submission.csv" in source
    assert "/kaggle/working/raw_predictions.csv" in source
    assert "experiment_log.csv" in source
    assert "Hugging Face fallback" in source
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_notebook_build.py -q
```

Expected: FAIL because `scripts/build_notebook.py` does not exist.

- [ ] **Step 3: Implement notebook builder**

Create `Math-VQA/scripts/build_notebook.py`:

```python
from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "thai_math_vqa_ollama_llava.ipynb"
MODULE_FILES = [
    PROJECT_ROOT / "src" / "math_vqa" / "__init__.py",
    PROJECT_ROOT / "src" / "math_vqa" / "data.py",
    PROJECT_ROOT / "src" / "math_vqa" / "evaluation.py",
    PROJECT_ROOT / "src" / "math_vqa" / "postprocess.py",
    PROJECT_ROOT / "src" / "math_vqa" / "preprocessing.py",
    PROJECT_ROOT / "src" / "math_vqa" / "prompts.py",
    PROJECT_ROOT / "src" / "math_vqa" / "ollama_client.py",
    PROJECT_ROOT / "src" / "math_vqa" / "submission.py",
]


def code_cell(source: str):
    return new_code_cell(textwrap.dedent(source).strip() + "\n")


def materialize_modules_cell() -> str:
    modules = {
        str(path.relative_to(PROJECT_ROOT / "src")): path.read_text(encoding="utf-8")
        for path in MODULE_FILES
    }
    return f"""
    from pathlib import Path
    import sys

    RUNTIME_SRC = Path("/kaggle/working/math_vqa_runtime")
    if not Path("/kaggle").exists():
        RUNTIME_SRC = Path.cwd() / "math_vqa_runtime"
    MATERIALIZED_MODULES = {modules!r}

    for relative_path, source in MATERIALIZED_MODULES.items():
        target = RUNTIME_SRC / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")

    sys.path.insert(0, str(RUNTIME_SRC))
    print(f"Materialized {{len(MATERIALIZED_MODULES)}} math_vqa module files into {{RUNTIME_SRC}}")
    """


def build_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = new_notebook(
        cells=[
            new_markdown_cell(
                "# Thai Math VQA LLaVA/Ollama Kaggle Notebook\n\n"
                "This notebook reads the Kaggle Thai Math VQA test images and writes "
                "`/kaggle/working/submission.csv` plus `/kaggle/working/raw_predictions.csv`.\n\n"
                "Hugging Face fallback: if Ollama cannot be installed or cannot pull `llava`, "
                "attach a Kaggle dataset containing a compatible vision-language model such as "
                "`llava-hf/llava-1.5-7b-hf` or `Qwen/Qwen2.5-VL-3B-Instruct`, then replace "
                "`query_llava(...)` in the inference cells with the corresponding Transformers pipeline call. "
                "Keep `clean_model_answer(...)`, `PredictionRecord`, and `write_outputs(...)` unchanged."
            ),
            code_cell(
                """
                from pathlib import Path
                import os
                import shutil
                import subprocess
                import sys
                import time

                import requests

                COMPETITION_DIR_NAME = "super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen"
                KAGGLE_DATA_ROOT = Path("/kaggle/input/competitions") / COMPETITION_DIR_NAME
                LOCAL_DATA_ROOT = Path("../data") / COMPETITION_DIR_NAME
                DATA_ROOT = KAGGLE_DATA_ROOT if KAGGLE_DATA_ROOT.exists() else LOCAL_DATA_ROOT.resolve()
                OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle").exists() else Path("../outputs").resolve()
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                MODEL_NAME = os.getenv("OLLAMA_MODEL", "llava")
                OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

                print(f"DATA_ROOT={DATA_ROOT}")
                print(f"OUTPUT_DIR={OUTPUT_DIR}")
                print(f"MODEL_NAME={MODEL_NAME}")

                def run_checked(command, *, shell=False, timeout=None):
                    result = subprocess.run(
                        command,
                        shell=shell,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                        check=False,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            "Command failed: "
                            + (command if isinstance(command, str) else " ".join(command))
                            + "\\nSTDOUT:\\n"
                            + result.stdout
                            + "\\nSTDERR:\\n"
                            + result.stderr
                        )
                    return result

                run_checked([sys.executable, "-m", "pip", "install", "-q", "pillow", "requests", "tqdm", "pandas"])

                if shutil.which("ollama") is None:
                    run_checked("curl -fsSL https://ollama.com/install.sh | sh", shell=True, timeout=600)

                server_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                ready = False
                for _ in range(60):
                    try:
                        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
                        if response.ok:
                            ready = True
                            break
                    except requests.RequestException:
                        time.sleep(2)
                if not ready:
                    raise RuntimeError("Ollama setup failed: server did not become ready on http://localhost:11434")

                run_checked(["ollama", "pull", MODEL_NAME], timeout=1800)
                print("Ollama is ready.")
                """
            ),
            code_cell(materialize_modules_cell()),
            code_cell(
                """
                import pandas as pd

                from math_vqa.data import CompetitionPaths, answer_prior, load_competition_frames, validate_data_files

                paths = CompetitionPaths(DATA_ROOT)
                train_df, test_df, sample_df = load_competition_frames(paths)
                validate_data_files(paths, train_df, test_df, sample_df)
                fallback_answer = answer_prior(train_df)

                print(f"train rows: {len(train_df)}")
                print(f"test rows: {len(test_df)}")
                print(f"sample rows: {len(sample_df)}")
                print(f"fallback answer: {fallback_answer}")
                """
            ),
            code_cell(
                """
                answer_summary = train_df["answer"].astype(str).value_counts().head(10)
                print("Top train answers:")
                print(answer_summary)
                print(f"Unique train answers: {train_df['answer'].nunique()}")
                """
            ),
            code_cell(
                """
                from tqdm.auto import tqdm

                from math_vqa.data import resolve_image_path
                from math_vqa.evaluation import normalized_accuracy
                from math_vqa.ollama_client import assert_ollama_ready, query_llava
                from math_vqa.postprocess import clean_model_answer
                from math_vqa.preprocessing import preprocess_image, save_preprocessed_image, select_preprocess_name
                from math_vqa.prompts import build_prompt, select_prompt_name
                from math_vqa.submission import PredictionRecord, write_outputs

                assert_ollama_ready(OLLAMA_HOST)
                PREPARED_IMAGE_DIR = OUTPUT_DIR / "prepared_images"
                """
            ),
            code_cell(
                """
                smoke_row = train_df.iloc[0]
                smoke_image = resolve_image_path(paths, smoke_row["image_path"])
                smoke_preprocess_name = select_preprocess_name(smoke_row["id"])
                smoke_prompt_name = select_prompt_name(smoke_row["id"])
                smoke_result = preprocess_image(smoke_image, smoke_preprocess_name)
                smoke_prepared_path = save_preprocessed_image(smoke_result, PREPARED_IMAGE_DIR, smoke_row["id"])
                smoke_raw = query_llava(
                    smoke_prepared_path,
                    build_prompt(smoke_prompt_name),
                    model=MODEL_NAME,
                    host=OLLAMA_HOST,
                )
                smoke_cleaned = clean_model_answer(smoke_raw, fallback_answer)
                print(
                    {
                        "id": smoke_row["id"],
                        "truth": smoke_row["answer"],
                        "raw_prediction": smoke_raw,
                        "clean_answer": smoke_cleaned.answer,
                        "preprocess_name": smoke_preprocess_name,
                        "final_size": smoke_result.final_size,
                    }
                )
                """
            ),
            code_cell(
                """
                RUN_HOLDOUT_VALIDATION = False
                HOLDOUT_ROWS = 20

                if RUN_HOLDOUT_VALIDATION:
                    holdout_df = train_df.sample(n=min(HOLDOUT_ROWS, len(train_df)), random_state=42)
                    holdout_predictions = []
                    holdout_truths = []
                    for row in tqdm(holdout_df.itertuples(index=False), total=len(holdout_df), desc="holdout"):
                        image_path = resolve_image_path(paths, row.image_path)
                        preprocess_name = select_preprocess_name(row.id)
                        prompt_name = select_prompt_name(row.id)
                        preprocess_result = preprocess_image(image_path, preprocess_name)
                        prepared_path = save_preprocessed_image(preprocess_result, PREPARED_IMAGE_DIR, row.id)
                        raw_prediction = query_llava(
                            prepared_path,
                            build_prompt(prompt_name),
                            model=MODEL_NAME,
                            host=OLLAMA_HOST,
                        )
                        cleaned = clean_model_answer(raw_prediction, fallback_answer)
                        holdout_predictions.append(cleaned.answer)
                        holdout_truths.append(row.answer)
                    print(f"Holdout normalized accuracy: {normalized_accuracy(holdout_predictions, holdout_truths):.4f}")
                else:
                    print("Holdout validation disabled for this run; train smoke test above was executed.")
                """
            ),
            code_cell(
                """
                records = []
                for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc="test inference"):
                    image_path = resolve_image_path(paths, row.image_path)
                    preprocess_name = select_preprocess_name(row.id)
                    prompt_name = select_prompt_name(row.id)
                    preprocess_result = preprocess_image(image_path, preprocess_name)
                    prepared_path = save_preprocessed_image(preprocess_result, PREPARED_IMAGE_DIR, row.id)
                    raw_prediction = query_llava(
                        prepared_path,
                        build_prompt(prompt_name),
                        model=MODEL_NAME,
                        host=OLLAMA_HOST,
                    )
                    cleaned = clean_model_answer(raw_prediction, fallback_answer)
                    records.append(
                        PredictionRecord(
                            id=str(row.id),
                            image_path=str(row.image_path),
                            raw_prediction=raw_prediction,
                            clean_answer=cleaned.answer,
                            prompt_name=prompt_name,
                            preprocess_name=preprocess_name,
                            used_fallback=cleaned.used_fallback,
                        )
                    )

                submission_path, raw_path = write_outputs(records, sample_df, OUTPUT_DIR)
                print(f"Wrote {submission_path}")
                print(f"Wrote {raw_path}")
                assert str(submission_path) == "/kaggle/working/submission.csv" or submission_path.name == "submission.csv"
                assert str(raw_path) == "/kaggle/working/raw_predictions.csv" or raw_path.name == "raw_predictions.csv"
                """
            ),
            code_cell(
                """
                experiment_log = pd.DataFrame(
                    [
                        {
                            "run": "001",
                            "model": MODEL_NAME,
                            "setup": "Ollama HTTP API",
                            "preprocessing": "image-id selector: raw/upscale/contrast/high_res",
                            "prompt": "image-id selector: base/diagram + strict formatting",
                            "postprocessing": "prefix cleanup, Thai digit conversion, empty-output fallback",
                            "local_score": "",
                            "public_lb": "",
                            "notes": f"fallback_count={sum(record.used_fallback for record in records)}; output_dir={OUTPUT_DIR}",
                        }
                    ]
                )
                experiment_log_path = OUTPUT_DIR / "experiment_log.csv"
                experiment_log.to_csv(experiment_log_path, index=False)
                print(f"Wrote {experiment_log_path}")
                """
            ),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
    )
    nbformat.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    build_notebook()
    print(f"Wrote {NOTEBOOK_PATH}")
```

- [ ] **Step 4: Generate notebook**

Run:

```bash
cd Math-VQA
python scripts/build_notebook.py
```

Expected: prints `Wrote .../Math-VQA/notebooks/thai_math_vqa_ollama_llava.ipynb`.

- [ ] **Step 5: Run notebook builder test to verify pass**

Run:

```bash
cd Math-VQA
python -m pytest tests/test_notebook_build.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Math-VQA/scripts/build_notebook.py Math-VQA/notebooks/thai_math_vqa_ollama_llava.ipynb Math-VQA/tests/test_notebook_build.py
git commit -m "feat: generate kaggle llava notebook"
```

### Task 8: README and Fallback Documentation

**Files:**
- Create: `Math-VQA/README.md`

- [ ] **Step 1: Create usage documentation**

Create `Math-VQA/README.md`:

```markdown
# Thai Math VQA Kaggle Pipeline

This project builds a Kaggle-ready notebook for the Super AI Engineer SS6 Thai Math VQA challenge. The notebook uses LLaVA through Ollama, produces `/kaggle/working/submission.csv`, and saves `/kaggle/working/raw_predictions.csv` for error analysis.

## Local Setup

```bash
cd Math-VQA
python -m pip install -e ".[dev]"
python -m pytest
python scripts/build_notebook.py
```

The generated notebook is `notebooks/thai_math_vqa_ollama_llava.ipynb`.

## Kaggle Setup

1. Upload or create the generated notebook in Kaggle.
2. Attach the competition dataset.
3. Enable internet for the first Ollama install and `llava` pull.
4. Run the notebook.
5. Submit `/kaggle/working/submission.csv`.

The notebook uses:

```text
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llava
```

If internet or model download is unreliable, package the Ollama model cache as a Kaggle dataset and adjust the setup cell to point Ollama at the attached model directory.

## Hugging Face Fallback

If Ollama cannot run in the Kaggle environment, attach a Kaggle dataset containing a compatible Hugging Face vision-language model such as `llava-hf/llava-1.5-7b-hf` or `Qwen/Qwen2.5-VL-3B-Instruct`. Replace the notebook call to `query_llava(...)` with a Transformers image-to-text call that returns one raw answer string. Keep these pieces unchanged:

- `clean_model_answer(...)`
- `PredictionRecord`
- `write_outputs(...)`
- `submission.csv` validation
- `raw_predictions.csv` logging

## Output Contract

The final submission must contain exactly these columns:

```csv
id,answer
```

The row order must match `sample_submission.csv`, ids are strings, answers are strings, and empty answers are rejected before the file is written.
```

- [ ] **Step 2: Verify README contains required commands and fallback**

Run:

```bash
cd Math-VQA
python - <<'PY'
from pathlib import Path
readme = Path("README.md").read_text(encoding="utf-8")
assert "python scripts/build_notebook.py" in readme
assert "ollama pull llava" in readme
assert "Hugging Face Fallback" in readme
assert "submission.csv" in readme
print("README checks passed")
PY
```

Expected: prints `README checks passed`.

- [ ] **Step 3: Commit**

```bash
git add Math-VQA/README.md
git commit -m "docs: document math vqa kaggle workflow"
```

### Task 9: Full Local Verification

**Files:**
- Modify: generated files from previous tasks only if verification exposes a concrete failure.

- [ ] **Step 1: Run the full unit test suite**

Run:

```bash
cd Math-VQA
python -m pytest
```

Expected: PASS for all tests.

- [ ] **Step 2: Regenerate the notebook from tested source**

Run:

```bash
cd Math-VQA
python scripts/build_notebook.py
```

Expected: prints `Wrote .../Math-VQA/notebooks/thai_math_vqa_ollama_llava.ipynb`.

- [ ] **Step 3: Inspect generated notebook for required output paths**

Run:

```bash
cd Math-VQA
python - <<'PY'
import json
from pathlib import Path
path = Path("notebooks/thai_math_vqa_ollama_llava.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
source = "\n".join("\n".join(cell.get("source", [])) for cell in notebook["cells"])
checks = [
    "/kaggle/working/submission.csv",
    "/kaggle/working/raw_predictions.csv",
    "experiment_log.csv",
    "clean_model_answer",
    "write_outputs",
]
missing = [check for check in checks if check not in source]
if missing:
    raise SystemExit(f"missing notebook checks: {missing}")
print("notebook contract checks passed")
PY
```

Expected: prints `notebook contract checks passed`.

- [ ] **Step 4: Run local data integrity check against the included competition data**

Run:

```bash
cd Math-VQA
python - <<'PY'
from math_vqa.data import CompetitionPaths, load_competition_frames, validate_data_files, answer_prior
paths = CompetitionPaths.auto()
train_df, test_df, sample_df = load_competition_frames(paths)
validate_data_files(paths, train_df, test_df, sample_df)
print({"train_rows": len(train_df), "test_rows": len(test_df), "sample_rows": len(sample_df), "fallback": answer_prior(train_df)})
PY
```

Expected: prints `{'train_rows': 280, 'test_rows': 420, 'sample_rows': 420, 'fallback': '2'}` or the same row counts with a different most-common simple fallback if the local data changes.

- [ ] **Step 5: Commit final generated notebook state**

```bash
git add Math-VQA/notebooks/thai_math_vqa_ollama_llava.ipynb
git commit -m "chore: verify generated math vqa notebook"
```

## Self-Review

- Spec coverage: Tasks 1 and 6 cover input/output CSV contracts and validation. Task 2 covers Kaggle normalization for local validation. Task 3 covers answer cleanup and fallback logging. Task 4 covers conservative RGB preprocessing variants and records final image size. Task 5 covers short-answer prompts and Ollama LLaVA calls. Task 7 covers Kaggle setup, smoke test, holdout switch, test inference loop, raw predictions, submission output, and experiment log. Task 8 documents the Ollama setup and Hugging Face fallback.
- Placeholder scan: The plan contains concrete files, concrete commands, concrete expected outputs, and complete code blocks for every code-writing step.
- Type consistency: `PredictionRecord` fields match `raw_predictions.csv`; `PreprocessResult.final_size` is used by the notebook smoke test; `clean_model_answer(...)` returns `CleanedAnswer.answer` and `CleanedAnswer.used_fallback`; notebook calls use the same function signatures defined in package tasks.

