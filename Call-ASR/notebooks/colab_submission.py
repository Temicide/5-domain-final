# %% [markdown]
# # Thai Call Center ASR Submission
# This notebook downloads Kaggle competition data, generates /content/submission.csv,
# and stops before any upload or submission step.

# %%
from pathlib import Path
from dataclasses import dataclass
import json
import os
import shutil
import subprocess
import sys
import zipfile

COLAB_REQUIRED_PACKAGES = [
    "kaggle",
    "pandas>=2.2.2",
    "numpy>=1.26.4",
    "soundfile>=0.12.1",
    "librosa>=0.10.2.post1",
    "jiwer>=3.0.4",
    "accelerate>=0.31.0",
    "transformers>=4.41.2",
]
COMPETITION_SLUG = "individual-test-thai-call-center-asr"
CONTENT_ROOT = Path("/content")
INPUT_DIR = Path("/content/input/individual-test-thai-call-center-asr")
WORKING_DIR = Path("/content/working")
SUBMISSION_PATH = CONTENT_ROOT / "submission.csv"


def install_colab_dependencies() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *COLAB_REQUIRED_PACKAGES],
        check=True,
    )


def configure_kaggle_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"

    uploaded_json = Path("/content/kaggle.json")
    if uploaded_json.exists():
        shutil.copyfile(uploaded_json, kaggle_json_path)
    else:
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if not username or not key:
            try:
                from google.colab import userdata

                username = username or userdata.get("KAGGLE_USERNAME")
                key = key or userdata.get("KAGGLE_KEY")
            except Exception:
                pass
        if not username or not key:
            raise RuntimeError(
                "Provide Kaggle credentials via uploaded /content/kaggle.json "
                "or Colab secrets/environment variables KAGGLE_USERNAME and KAGGLE_KEY."
            )
        kaggle_json_path.write_text(
            json.dumps({"username": username, "key": key}),
            encoding="utf-8",
        )
    kaggle_json_path.chmod(0o600)


def download_and_extract_competition_data() -> Path:
    configure_kaggle_credentials()
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_submission = INPUT_DIR / "sample_submission.csv"
    audio_dir = INPUT_DIR / "audio_final" / "audio"
    if sample_submission.exists() and audio_dir.is_dir():
        return INPUT_DIR

    archive_path = WORKING_DIR / f"{COMPETITION_SLUG}.zip"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION_SLUG,
            "-p",
            str(WORKING_DIR),
            "--force",
        ],
        check=True,
    )
    if not archive_path.exists():
        raise FileNotFoundError(f"Downloaded archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as zip_file:
        zip_file.extractall(INPUT_DIR)
    return INPUT_DIR


# %%
install_colab_dependencies()

import pandas as pd


class SubmissionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CompetitionPaths:
    input_dir: Path
    audio_dir: Path
    sample_submission: Path
    working_dir: Path
    submissions_dir: Path
    is_colab: bool


def _has_competition_files(input_dir: Path) -> bool:
    return (input_dir / "sample_submission.csv").is_file() and (input_dir / "audio_final" / "audio").is_dir()


def resolve_competition_paths(colab_input_root: Path = Path("/content/input")) -> CompetitionPaths:
    colab_competition_dir = colab_input_root / COMPETITION_SLUG
    if not _has_competition_files(colab_competition_dir):
        raise FileNotFoundError(
            "Competition files not found. Expected sample_submission.csv and "
            f"audio_final/audio under {colab_competition_dir}. Run download_and_extract_competition_data() first."
        )
    return CompetitionPaths(
        input_dir=colab_competition_dir,
        audio_dir=colab_competition_dir / "audio_final" / "audio",
        sample_submission=colab_competition_dir / "sample_submission.csv",
        working_dir=WORKING_DIR,
        submissions_dir=WORKING_DIR,
        is_colab=True,
    )


def load_sample_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"file_name": "string", "text": "string"}, keep_default_na=False)
    if list(df.columns) != ["file_name", "text"]:
        raise SubmissionValidationError(
            f"Expected sample submission columns ['file_name', 'text'], got {list(df.columns)}"
        )
    if df["file_name"].duplicated().any():
        duplicates = df.loc[df["file_name"].duplicated(), "file_name"].tolist()
        raise SubmissionValidationError(f"Duplicate file_name values: {', '.join(duplicates[:10])}")
    return df


def validate_audio_coverage(sample_df: pd.DataFrame, audio_dir: Path) -> None:
    missing = [name for name in sample_df["file_name"].tolist() if not (audio_dir / name).is_file()]
    if missing:
        raise SubmissionValidationError(f"Missing audio files: {', '.join(missing[:20])}")


def validate_submission_frame(submission_df: pd.DataFrame, allow_empty_files: set[str] | None = None) -> None:
    allow_empty_files = allow_empty_files or set()
    if list(submission_df.columns) != ["file_name", "text"]:
        raise SubmissionValidationError(
            f"Expected submission columns ['file_name', 'text'], got {list(submission_df.columns)}"
        )
    if submission_df["file_name"].duplicated().any():
        duplicates = submission_df.loc[submission_df["file_name"].duplicated(), "file_name"].tolist()
        raise SubmissionValidationError(f"Duplicate submission rows: {', '.join(duplicates[:10])}")
    for row in submission_df.itertuples(index=False):
        text = "" if pd.isna(row.text) else str(row.text)
        if text == "" and row.file_name not in allow_empty_files:
            raise SubmissionValidationError(f"Empty transcript for {row.file_name}")


def write_submission_csv(
    sample_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_path: Path,
    allow_empty_files: set[str] | None = None,
) -> pd.DataFrame:
    required_prediction_columns = {"file_name", "normalized_text"}
    missing_columns = required_prediction_columns - set(predictions_df.columns)
    if missing_columns:
        raise SubmissionValidationError(f"Prediction columns missing: {', '.join(sorted(missing_columns))}")

    merged = sample_df[["file_name"]].merge(
        predictions_df[["file_name", "normalized_text"]],
        on="file_name",
        how="left",
        validate="one_to_one",
    )
    if merged["normalized_text"].isna().any():
        missing = merged.loc[merged["normalized_text"].isna(), "file_name"].tolist()
        raise SubmissionValidationError(f"Missing predictions: {', '.join(missing[:20])}")

    submission_df = merged.rename(columns={"normalized_text": "text"})[["file_name", "text"]]
    submission_df["text"] = submission_df["text"].fillna("").astype(str)
    validate_submission_frame(submission_df, allow_empty_files=allow_empty_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return submission_df


def normalize_text(text: str, policy: str) -> str:
    normalized = str(text).strip()
    if policy == "none":
        return normalized
    if policy == "single_space":
        return " ".join(normalized.split())
    raise ValueError(f"Unknown normalization policy: {policy}")


PREDICTION_COLUMNS = [
    "file_name",
    "raw_text",
    "normalized_text",
    "model_name",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
    "error",
]


@dataclass(frozen=True)
class AsrResult:
    text: str
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class WhisperPipelineBackend:
    def __init__(
        self,
        model_name: str,
        chunk_length_s: int,
        batch_size: int,
        condition_on_previous_text: bool,
        device: str,
    ):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. In Colab, select Runtime > Change runtime type > A100 GPU.")
        self.model_name = model_name
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        device_index = 0 if device == "cuda" else -1
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(model_name)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            device=device_index,
            return_timestamps=False,
        )
        self.generate_kwargs = {
            "language": "thai",
            "task": "transcribe",
            "condition_on_prev_tokens": condition_on_previous_text,
        }

    def transcribe(self, audio_path: Path) -> AsrResult:
        output = self.pipe(str(audio_path), generate_kwargs=self.generate_kwargs)
        text = str(output.get("text", ""))
        return AsrResult(text=text, avg_logprob=0.0, compression_ratio=0.0, no_speech_prob=0.0)


def _load_existing(output_csv: Path) -> pd.DataFrame:
    if output_csv.is_file():
        return pd.read_csv(output_csv, keep_default_na=False)
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _write_log(log_jsonl: Path, row: dict) -> None:
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with log_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_inference(
    sample_df: pd.DataFrame,
    audio_dir: Path,
    output_csv: Path,
    log_jsonl: Path,
    backend: WhisperPipelineBackend,
    normalization_policy: str,
    resume: bool,
) -> pd.DataFrame:
    existing = _load_existing(output_csv) if resume else pd.DataFrame(columns=PREDICTION_COLUMNS)
    completed = set(existing["file_name"].tolist()) if not existing.empty else set()
    rows = existing.to_dict("records")

    for file_name in sample_df["file_name"].tolist():
        if file_name in completed:
            continue
        audio_path = audio_dir / file_name
        try:
            result = backend.transcribe(audio_path)
            raw_text = result.text
            normalized_text = normalize_text(raw_text, normalization_policy)
            row = {
                "file_name": file_name,
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "model_name": backend.model_name,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
                "error": "",
            }
        except Exception as exc:
            row = {
                "file_name": file_name,
                "raw_text": "",
                "normalized_text": "",
                "model_name": backend.model_name,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        _write_log(log_jsonl, row)
        rows.append(row)
        pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(output_csv, index=False)

    output = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    order = {file_name: position for position, file_name in enumerate(sample_df["file_name"].tolist())}
    output["_order"] = output["file_name"].map(order)
    output = output.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    output.to_csv(output_csv, index=False)
    return output


def require_cuda_runtime() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. In Colab, select Runtime > Change runtime type > A100 GPU.")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# %%
MODEL_NAME = "typhoon-ai/typhoon-whisper-large-v3"
NORMALIZATION_POLICY = "single_space"
CHUNK_LENGTH_SECONDS = 30
BATCH_SIZE = 4
CONDITION_ON_PREVIOUS_TEXT = False

# %%
download_and_extract_competition_data()
paths = resolve_competition_paths(colab_input_root=CONTENT_ROOT / "input")
sample_df = load_sample_submission(paths.sample_submission)
validate_audio_coverage(sample_df, paths.audio_dir)
print(f"Rows in sample submission: {len(sample_df)}")
print(f"Audio directory: {paths.audio_dir}")
print(f"Output submission path: {SUBMISSION_PATH}")

# %%
require_cuda_runtime()
backend = WhisperPipelineBackend(
    model_name=MODEL_NAME,
    chunk_length_s=CHUNK_LENGTH_SECONDS,
    batch_size=BATCH_SIZE,
    condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
    device="cuda",
)

# %%
safe_model_name = MODEL_NAME.replace("/", "__")
predictions_path = paths.working_dir / f"predictions_{safe_model_name}_{NORMALIZATION_POLICY}.csv"
log_path = paths.working_dir / f"run_{safe_model_name}_{NORMALIZATION_POLICY}.jsonl"
predictions_df = run_inference(
    sample_df=sample_df,
    audio_dir=paths.audio_dir,
    output_csv=predictions_path,
    log_jsonl=log_path,
    backend=backend,
    normalization_policy=NORMALIZATION_POLICY,
    resume=True,
)
print(f"Wrote predictions: {predictions_path}")
print(f"Wrote logs: {log_path}")
print(f"Prediction rows: {len(predictions_df)}")
print(f"Inference errors: {(predictions_df['error'] != '').sum()}")

# %%
submission_df = write_submission_csv(
    sample_df=sample_df,
    predictions_df=predictions_df,
    output_path=SUBMISSION_PATH,
    allow_empty_files=set(),
)
validate_submission_frame(submission_df, allow_empty_files=set())
print(f"Wrote validated submission: {SUBMISSION_PATH}")
print(submission_df.head().to_string(index=False))
