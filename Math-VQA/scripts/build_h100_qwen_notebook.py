from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "thai_math_vqa_h100_qwen_vl.ipynb"
MODULE_FILES = [
    PROJECT_ROOT / "src" / "math_vqa" / "__init__.py",
    PROJECT_ROOT / "src" / "math_vqa" / "data.py",
    PROJECT_ROOT / "src" / "math_vqa" / "evaluation.py",
    PROJECT_ROOT / "src" / "math_vqa" / "postprocess.py",
    PROJECT_ROOT / "src" / "math_vqa" / "preprocessing.py",
    PROJECT_ROOT / "src" / "math_vqa" / "prompts.py",
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
        print("Rerun scripts/build_h100_qwen_notebook.py after source modules are complete.")
    """


def build_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = new_notebook(
        cells=[
            new_markdown_cell(
                "# Thai Math VQA H100 Qwen-VL Colab Notebook\n\n"
                "This notebook is the research-backed replacement for the Ollama/LLaVA path. "
                "It downloads the Kaggle competition data, loads a modern Qwen vision-language model "
                "with Transformers, runs fixed-seed train holdout validation, then writes "
                "`/content/math-vqa-output/submission.csv` and `/content/math-vqa-output/raw_predictions.csv`.\n\n"
                "Recommended H100 start: `VLM_MODEL_ID=Qwen/Qwen3-VL-32B-Instruct`, BF16, bounded visual "
                "tokens, deterministic generation. If BF16 does not fit, set `LOAD_IN_4BIT=1` or use "
                "`Qwen/Qwen2.5-VL-32B-Instruct` as a fallback."
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

                COMPETITION_SLUG = "super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen"
                IS_COLAB = Path("/content").exists()
                DATA_PARENT = Path("/content/math-vqa-data") if IS_COLAB else Path("../data").resolve()
                DATA_ROOT = DATA_PARENT / COMPETITION_SLUG
                OUTPUT_DIR = Path("/content/math-vqa-output") if IS_COLAB else Path("../outputs").resolve()
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "Qwen/Qwen3-VL-32B-Instruct")
                LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "0") == "1"
                INSTALL_FLASH_ATTN = os.getenv("INSTALL_FLASH_ATTN", "0") == "1"
                ENABLE_FLASH_ATTN = os.getenv("ENABLE_FLASH_ATTN", "1") == "1"
                MIN_PIXELS = int(os.getenv("VLM_MIN_PIXELS", str(256 * 28 * 28)))
                MAX_PIXELS = int(os.getenv("VLM_MAX_PIXELS", str(2048 * 28 * 28)))
                MAX_NEW_TOKENS = int(os.getenv("VLM_MAX_NEW_TOKENS", "32"))
                HOLDOUT_ROWS = int(os.getenv("HOLDOUT_ROWS", "56"))
                RUN_HOLDOUT_VALIDATION = os.getenv("RUN_HOLDOUT_VALIDATION", "1") == "1"
                RUN_TEST_INFERENCE = os.getenv("RUN_TEST_INFERENCE", "1") == "1"

                print(f"IS_COLAB={IS_COLAB}")
                print(f"DATA_ROOT={DATA_ROOT}")
                print(f"OUTPUT_DIR={OUTPUT_DIR}")
                print(f"VLM_MODEL_ID={VLM_MODEL_ID}")
                print(f"LOAD_IN_4BIT={LOAD_IN_4BIT}")
                print(f"MIN_PIXELS={MIN_PIXELS}")
                print(f"MAX_PIXELS={MAX_PIXELS}")
                print(f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
                print(f"RUN_HOLDOUT_VALIDATION={RUN_HOLDOUT_VALIDATION}")
                print(f"RUN_TEST_INFERENCE={RUN_TEST_INFERENCE}")

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

                    archive_candidates = [DATA_PARENT / f"{COMPETITION_SLUG}.zip"]
                    archive_candidates.extend(sorted(DATA_PARENT.glob("*.zip")))
                    archive_path = next((candidate for candidate in archive_candidates if candidate.exists()), None)
                    if archive_path is None:
                        raise RuntimeError(f"Kaggle download completed, but no zip archive was found in {DATA_PARENT}.")
                    extract_competition_archive(archive_path)
                    print(f"Downloaded and extracted competition data to {DATA_ROOT}.")

                setup_packages = [
                    "kaggle",
                    "pillow",
                    "requests",
                    "tqdm",
                    "pandas",
                    "accelerate",
                    "qwen-vl-utils",
                ]
                run_checked([sys.executable, "-m", "pip", "install", "-q", *setup_packages], timeout=900)
                run_checked([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/huggingface/transformers"], timeout=1800)
                if LOAD_IN_4BIT:
                    run_checked([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], timeout=900)
                if INSTALL_FLASH_ATTN:
                    run_checked([sys.executable, "-m", "pip", "install", "-q", "flash-attn", "--no-build-isolation"], timeout=3600)

                download_competition_data()
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
                print("Top train answers:")
                print(train_df["answer"].astype(str).value_counts().head(12))
                """
            ),
            code_cell(
                """
                import importlib.util
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                if not torch.cuda.is_available():
                    raise RuntimeError("A GPU runtime is required. In Colab, select an H100/A100/L4/T4 GPU runtime before running this notebook.")

                attn_implementation = None
                if ENABLE_FLASH_ATTN and importlib.util.find_spec("flash_attn") is not None:
                    attn_implementation = "flash_attention_2"

                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                }
                if attn_implementation:
                    model_kwargs["attn_implementation"] = attn_implementation

                if LOAD_IN_4BIT:
                    from transformers import BitsAndBytesConfig

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )

                print("Loading processor...")
                processor = AutoProcessor.from_pretrained(
                    VLM_MODEL_ID,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                    trust_remote_code=True,
                )
                print("Loading model...")
                model = AutoModelForImageTextToText.from_pretrained(
                    VLM_MODEL_ID,
                    trust_remote_code=True,
                    **model_kwargs,
                )
                model.eval()

                print(model.__class__.__name__)
                print(f"attn_implementation={attn_implementation or 'default/sdpa'}")
                print(f"cuda_device={torch.cuda.get_device_name(0)}")
                """
            ),
            code_cell(
                """
                from pathlib import Path
                import time

                from PIL import Image
                from qwen_vl_utils import process_vision_info
                from tqdm.auto import tqdm

                from math_vqa.data import resolve_image_path
                from math_vqa.evaluation import normalized_accuracy
                from math_vqa.postprocess import clean_model_answer
                from math_vqa.preprocessing import preprocess_image, save_preprocessed_image, select_preprocess_name
                from math_vqa.prompts import build_prompt, select_prompt_name
                from math_vqa.submission import PredictionRecord, write_outputs

                PREPARED_IMAGE_DIR = OUTPUT_DIR / "prepared_images_qwen"

                def build_h100_prompt(prompt_name: str) -> str:
                    return (
                        build_prompt(prompt_name)
                        + "\\nIf the image is low resolution, infer the most likely answer anyway. "
                        + "Never ask for a clearer image."
                    )

                def query_qwen_vl(image_path: str | Path, prompt: str) -> str:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(image_path)},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device)
                    with torch.inference_mode():
                        generated_ids = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=MAX_NEW_TOKENS,
                            use_cache=True,
                        )
                    generated_ids_trimmed = [
                        output_ids[len(input_ids):]
                        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    return processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()

                def predict_image_record(image_id, image_path_value):
                    started_at = time.time()
                    image_path = resolve_image_path(paths, image_path_value)
                    preprocess_name = select_preprocess_name(image_id)
                    prompt_name = select_prompt_name(image_id)
                    preprocess_result = preprocess_image(image_path, preprocess_name)
                    prepared_path = save_preprocessed_image(preprocess_result, PREPARED_IMAGE_DIR, image_id)
                    inference_error = ""
                    try:
                        raw_prediction = query_qwen_vl(prepared_path, build_h100_prompt(prompt_name))
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
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                if RUN_HOLDOUT_VALIDATION:
                    holdout_df = train_df.sample(n=min(HOLDOUT_ROWS, len(train_df)), random_state=42)
                    holdout_records = []
                    holdout_truths = []
                    for row in tqdm(holdout_df.itertuples(index=False), total=len(holdout_df), desc="holdout"):
                        record = predict_image_record(row.id, row.image_path)
                        holdout_records.append(record)
                        holdout_truths.append(row.answer)
                    holdout_predictions = [record.clean_answer for record in holdout_records]
                    holdout_score = normalized_accuracy(holdout_predictions, holdout_truths)
                    holdout_raw_path = OUTPUT_DIR / "holdout_raw_predictions.csv"
                    pd.DataFrame([record.__dict__ for record in holdout_records]).to_csv(holdout_raw_path, index=False)
                    print(f"Holdout normalized accuracy: {holdout_score:.4f}")
                    print(f"Holdout fallback count: {sum(record.used_fallback for record in holdout_records)}")
                    print(f"Wrote {holdout_raw_path}")
                else:
                    print("Holdout validation disabled.")

                if torch.cuda.is_available():
                    print(f"Peak GPU memory allocated GB: {torch.cuda.max_memory_allocated() / 1024**3:.2f}")
                """
            ),
            code_cell(
                """
                if RUN_TEST_INFERENCE:
                    records = []
                    for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc="test inference"):
                        records.append(predict_image_record(row.id, row.image_path))

                    submission_path, raw_path = write_outputs(records, sample_df, OUTPUT_DIR)
                    print(f"Wrote {submission_path}")
                    print(f"Wrote {raw_path}")
                    print(f"fallback_count={sum(record.used_fallback for record in records)}")
                    print(f"error_count={sum(bool(record.inference_error) for record in records)}")
                    assert str(submission_path) == "/content/math-vqa-output/submission.csv" or submission_path.name == "submission.csv"
                    assert str(raw_path) == "/content/math-vqa-output/raw_predictions.csv" or raw_path.name == "raw_predictions.csv"
                else:
                    print("Test inference disabled. Set RUN_TEST_INFERENCE=1 to write submission.csv.")
                """
            ),
            code_cell(
                """
                experiment_log = pd.DataFrame(
                    [
                        {
                            "run": "h100-qwen-001",
                            "model": VLM_MODEL_ID,
                            "setup": "Transformers AutoModelForImageTextToText",
                            "preprocessing": "image-id selector: raw/upscale/contrast/high_res",
                            "prompt": "Qwen-VL answer-only prompt + no clearer-image request",
                            "postprocessing": "strict short-answer cleanup, refusal fallback, declared-final-answer extraction",
                            "local_score": locals().get("holdout_score", ""),
                            "public_lb": "",
                            "notes": (
                                f"load_in_4bit={LOAD_IN_4BIT}; min_pixels={MIN_PIXELS}; max_pixels={MAX_PIXELS}; "
                                f"max_new_tokens={MAX_NEW_TOKENS}; output_dir={OUTPUT_DIR}"
                            ),
                        }
                    ]
                )
                experiment_log_path = OUTPUT_DIR / "experiment_log_h100_qwen.csv"
                experiment_log.to_csv(experiment_log_path, index=False)
                print(f"Wrote {experiment_log_path}")
                """
            ),
        ]
    )
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
