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
    modules: dict[str, str] = {}
    missing_modules: list[str] = []
    for path in MODULE_FILES:
        relative_path = str(path.relative_to(PROJECT_ROOT / "src"))
        if path.exists():
            modules[relative_path] = path.read_text(encoding="utf-8")
        else:
            missing_modules.append(relative_path)

    return f"""
    from pathlib import Path
    import sys

    RUNTIME_SRC = Path("/content/math_vqa_runtime") if Path("/content").exists() else Path.cwd() / "math_vqa_runtime"
    MATERIALIZED_MODULES = {modules!r}
    MISSING_MATERIALIZED_MODULES = {missing_modules!r}

    for relative_path, source in MATERIALIZED_MODULES.items():
        target = RUNTIME_SRC / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")

    sys.path.insert(0, str(RUNTIME_SRC))
    print(f"Materialized {{len(MATERIALIZED_MODULES)}} math_vqa module files into {{RUNTIME_SRC}}")
    if MISSING_MATERIALIZED_MODULES:
        print("Missing planned module sources at notebook-build time:", MISSING_MATERIALIZED_MODULES)
        print("Rerun scripts/build_notebook.py after Tasks 1-5 complete to embed the full runtime package.")
    """


def build_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = new_notebook(
        cells=[
            new_markdown_cell(
                "# Thai Math VQA LLaVA/Ollama Colab Notebook\n\n"
                "This notebook downloads the Kaggle Thai Math VQA competition data into Google Colab, "
                "runs local Ollama/LLaVA inference, and writes "
                "`/content/math-vqa-output/submission.csv` plus `/content/math-vqa-output/raw_predictions.csv`.\n\n"
                "Hugging Face fallback: if Ollama cannot be installed or cannot pull `llava`, "
                "attach a Kaggle dataset containing a compatible vision-language model such as "
                "`llava-hf/llava-1.5-7b-hf` or `Qwen/Qwen2.5-VL-3B-Instruct`, then replace "
                "`query_llava(...)` in the inference cells with the corresponding Transformers pipeline call. "
                "Keep `clean_model_answer(...)`, `PredictionRecord`, and `write_outputs(...)` unchanged."
            ),
            code_cell(
                """
                from pathlib import Path
                import json
                import os
                import shutil
                import subprocess
                import sys
                import time
                import zipfile

                import requests

                COMPETITION_SLUG = "super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen"
                IS_COLAB = Path("/content").exists()
                DATA_PARENT = Path("/content/math-vqa-data") if IS_COLAB else Path("../data").resolve()
                DATA_ROOT = DATA_PARENT / COMPETITION_SLUG
                OUTPUT_DIR = Path("/content/math-vqa-output") if IS_COLAB else Path("../outputs").resolve()
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                MODEL_NAME = os.getenv("OLLAMA_MODEL", "llava")
                OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
                OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))
                OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "1"))

                print(f"IS_COLAB={IS_COLAB}")
                print(f"DATA_ROOT={DATA_ROOT}")
                print(f"OUTPUT_DIR={OUTPUT_DIR}")
                print(f"MODEL_NAME={MODEL_NAME}")
                print(f"OLLAMA_REQUEST_TIMEOUT={OLLAMA_REQUEST_TIMEOUT}")
                print(f"OLLAMA_RETRIES={OLLAMA_RETRIES}")

                def run_command(command, *, shell=False, timeout=None):
                    return subprocess.run(
                        command,
                        shell=shell,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                        check=False,
                    )

                def run_checked(command, *, shell=False, timeout=None):
                    result = run_command(command, shell=shell, timeout=timeout)
                    if result.returncode != 0:
                        rendered_command = command if isinstance(command, str) else " ".join(command)
                        raise RuntimeError(
                            f"Command failed: {rendered_command}\\nSTDOUT:\\n{result.stdout}\\nSTDERR:\\n{result.stderr}"
                        )
                    return result

                def ensure_zstd_for_ollama():
                    if shutil.which("zstd") is not None:
                        print("zstd is available.")
                        return
                    if shutil.which("apt-get") is None:
                        raise RuntimeError(
                            "Ollama install requires zstd, but `zstd` is not on PATH and `apt-get` is unavailable. "
                            "Use a Kaggle/runtime image that includes zstd, package Ollama/model assets as a dataset, "
                            "or switch this notebook to the documented Hugging Face fallback."
                        )
                    print("Installing zstd because the Ollama installer requires it for .tar.zst extraction.")
                    run_checked(
                        "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends zstd",
                        shell=True,
                        timeout=600,
                    )
                    if shutil.which("zstd") is None:
                        raise RuntimeError(
                            "Installed the zstd apt package, but the `zstd` command is still not on PATH. "
                            "Ollama cannot be installed in this runtime; use packaged assets or the Hugging Face fallback."
                        )

                def data_files_ready(root):
                    required_paths = [
                        root / "train.csv",
                        root / "test.csv",
                        root / "sample_submission.csv",
                        root / "images" / "images",
                    ]
                    return all(path.exists() for path in required_paths)

                def configure_kaggle_credentials():
                    kaggle_dir = Path.home() / ".kaggle"
                    token_path = kaggle_dir / "kaggle.json"
                    kaggle_dir.mkdir(parents=True, exist_ok=True)

                    username = os.getenv("KAGGLE_USERNAME")
                    key = os.getenv("KAGGLE_KEY")
                    if username and key:
                        token_path.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
                        token_path.chmod(0o600)
                        print("Configured Kaggle credentials from KAGGLE_USERNAME/KAGGLE_KEY.")
                        return

                    if token_path.exists():
                        token_path.chmod(0o600)
                        print(f"Using existing Kaggle credential file at {token_path}.")
                        return

                    try:
                        from google.colab import files
                    except Exception as exc:
                        raise RuntimeError(
                            "Kaggle credentials are required to download competition data. "
                            "Set KAGGLE_USERNAME and KAGGLE_KEY, or place kaggle.json at ~/.kaggle/kaggle.json."
                        ) from exc

                    print("Upload your Kaggle API token file named kaggle.json.")
                    uploaded = files.upload()
                    if "kaggle.json" not in uploaded:
                        raise RuntimeError("Expected an uploaded file named kaggle.json.")
                    token_path.write_bytes(uploaded["kaggle.json"])
                    token_path.chmod(0o600)
                    print(f"Saved Kaggle credentials to {token_path}.")

                def extract_competition_archive(archive_path):
                    DATA_ROOT.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(archive_path) as archive:
                        archive.extractall(DATA_ROOT)
                    if not data_files_ready(DATA_ROOT):
                        raise RuntimeError(
                            f"Downloaded archive extracted to {DATA_ROOT}, but required train/test/sample/image files were not found."
                        )

                def download_competition_data():
                    if data_files_ready(DATA_ROOT):
                        print(f"Competition data already exists at {DATA_ROOT}.")
                        return

                    configure_kaggle_credentials()
                    DATA_PARENT.mkdir(parents=True, exist_ok=True)
                    command_variants = [
                        ["kaggle", "competitions", "download", "-c", COMPETITION_SLUG, "-p", str(DATA_PARENT), "-o"],
                        ["kaggle", "competitions", "download", COMPETITION_SLUG, "-p", str(DATA_PARENT), "-o"],
                    ]
                    failures = []
                    for command in command_variants:
                        result = run_command(command, timeout=1800)
                        if result.returncode == 0:
                            break
                        failures.append(" ".join(command) + "\\nSTDOUT:\\n" + result.stdout + "\\nSTDERR:\\n" + result.stderr)
                    else:
                        raise RuntimeError(
                            "Kaggle competition data download failed. Confirm that your Kaggle API token is valid "
                            "and that you accepted the competition rules on Kaggle.\\n\\n" + "\\n\\n".join(failures)
                        )

                    archive_candidates = [
                        DATA_PARENT / f"{COMPETITION_SLUG}.zip",
                        DATA_PARENT / "super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen.zip",
                    ]
                    archive_candidates.extend(sorted(DATA_PARENT.glob("*.zip")))
                    archive_path = next((candidate for candidate in archive_candidates if candidate.exists()), None)
                    if archive_path is None:
                        raise RuntimeError(f"Kaggle download completed, but no zip archive was found in {DATA_PARENT}.")
                    extract_competition_archive(archive_path)
                    print(f"Downloaded and extracted competition data to {DATA_ROOT}.")

                run_checked([sys.executable, "-m", "pip", "install", "-q", "kaggle", "pillow", "requests", "tqdm", "pandas"])
                download_competition_data()

                if shutil.which("ollama") is None:
                    ensure_zstd_for_ollama()
                    run_checked(
                        "bash -lc 'set -o pipefail; curl -fsSL https://ollama.com/install.sh | sh'",
                        shell=True,
                        timeout=600,
                    )
                if shutil.which("ollama") is None:
                    raise RuntimeError("Ollama install failed: the `ollama` command is still not on PATH after install.")

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

                # Equivalent shell command: ollama pull $OLLAMA_MODEL
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
                import time

                from math_vqa.data import resolve_image_path
                from math_vqa.evaluation import normalized_accuracy
                from math_vqa.ollama_client import assert_ollama_ready, query_llava
                from math_vqa.postprocess import clean_model_answer
                from math_vqa.preprocessing import preprocess_image, save_preprocessed_image, select_preprocess_name
                from math_vqa.prompts import build_prompt, select_prompt_name
                from math_vqa.submission import PredictionRecord, write_outputs

                assert_ollama_ready(OLLAMA_HOST)
                PREPARED_IMAGE_DIR = OUTPUT_DIR / "prepared_images"

                def predict_image_record(image_id, image_path_value):
                    started_at = time.time()
                    image_path = resolve_image_path(paths, image_path_value)
                    preprocess_name = select_preprocess_name(image_id)
                    prompt_name = select_prompt_name(image_id)
                    preprocess_result = preprocess_image(image_path, preprocess_name)
                    prepared_path = save_preprocessed_image(preprocess_result, PREPARED_IMAGE_DIR, image_id)
                    inference_error = ""
                    try:
                        raw_prediction = query_llava(
                            prepared_path,
                            build_prompt(prompt_name),
                            model=MODEL_NAME,
                            host=OLLAMA_HOST,
                            timeout=OLLAMA_REQUEST_TIMEOUT,
                            retries=OLLAMA_RETRIES,
                        )
                        cleaned = clean_model_answer(raw_prediction, fallback_answer)
                    except Exception as exc:
                        inference_error = f"{type(exc).__name__}: {exc}"
                        raw_prediction = f"ERROR: {inference_error}"
                        cleaned = clean_model_answer("", fallback_answer)
                    runtime_seconds = round(time.time() - started_at, 3)
                    return PredictionRecord(
                        id=str(image_id),
                        image_path=str(image_path_value),
                        raw_prediction=raw_prediction,
                        clean_answer=cleaned.answer,
                        prompt_name=prompt_name,
                        preprocess_name=preprocess_name,
                        final_size=f"{preprocess_result.final_size[0]}x{preprocess_result.final_size[1]}",
                        runtime_seconds=runtime_seconds,
                        inference_error=inference_error,
                        used_fallback=cleaned.used_fallback,
                    )
                """
            ),
            code_cell(
                """
                smoke_row = train_df.iloc[0]
                smoke_record = predict_image_record(smoke_row["id"], smoke_row["image_path"])
                print(
                    {
                        "id": smoke_row["id"],
                        "truth": smoke_row["answer"],
                        "raw_prediction": smoke_record.raw_prediction,
                        "clean_answer": smoke_record.clean_answer,
                        "preprocess_name": smoke_record.preprocess_name,
                        "final_size": smoke_record.final_size,
                        "runtime_seconds": smoke_record.runtime_seconds,
                        "inference_error": smoke_record.inference_error,
                        "used_fallback": smoke_record.used_fallback,
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
                        record = predict_image_record(row.id, row.image_path)
                        holdout_predictions.append(record.clean_answer)
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
                    records.append(predict_image_record(row.id, row.image_path))

                submission_path, raw_path = write_outputs(records, sample_df, OUTPUT_DIR)
                print(f"Wrote {submission_path}")
                print(f"Wrote {raw_path}")
                assert str(submission_path) == "/content/math-vqa-output/submission.csv" or submission_path.name == "submission.csv"
                assert str(raw_path) == "/content/math-vqa-output/raw_predictions.csv" or raw_path.name == "raw_predictions.csv"
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
                            "notes": (
                                f"fallback_count={sum(record.used_fallback for record in records)}; "
                                f"error_count={sum(bool(record.inference_error) for record in records)}; "
                                f"timeout_seconds={OLLAMA_REQUEST_TIMEOUT}; retries={OLLAMA_RETRIES}; output_dir={OUTPUT_DIR}"
                            ),
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
