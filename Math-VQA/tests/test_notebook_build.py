from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_notebook_creates_required_colab_sections() -> None:
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
    assert "/content/math-vqa-data" in source
    assert "/content/math-vqa-output/submission.csv" in source
    assert "pip\", \"install\", \"-q\", \"kaggle\"" in source
    assert "configure_kaggle_credentials" in source
    assert "google.colab" in source
    assert "kaggle\", \"competitions\", \"download\"" in source
    assert "accepted the competition rules" in source
    assert "curl -fsSL https://ollama.com/install.sh | sh" in source
    assert "ensure_zstd_for_ollama" in source
    assert "apt-get install -y -qq --no-install-recommends zstd" in source
    assert "Ollama install requires zstd" in source
    assert "OLLAMA_REQUEST_TIMEOUT" in source
    assert "OLLAMA_RETRIES" in source
    assert "predict_image_record" in source
    assert "inference_error" in source
    assert "ollama pull" in source
    assert "MATERIALIZED_MODULES" in source
    assert "/kaggle/working" not in source
    assert "/content/math-vqa-output/submission.csv" in source
    assert "/content/math-vqa-output/raw_predictions.csv" in source
    assert "experiment_log.csv" in source
    assert "Hugging Face fallback" in source


def test_build_h100_qwen_notebook_creates_qwen_validation_sections() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_h100_qwen_notebook.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    notebook_path = PROJECT_ROOT / "notebooks" / "thai_math_vqa_h100_qwen_vl.ipynb"
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "\n".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "Qwen/Qwen3-VL-32B-Instruct" in source
    assert "Qwen/Qwen2.5-VL-32B-Instruct" in source
    assert "AutoModelForImageTextToText" in source
    assert "qwen-vl-utils" in source
    assert "process_vision_info" in source
    assert "LOAD_IN_4BIT" in source
    assert "ENABLE_FLASH_ATTN" in source
    assert "RUN_HOLDOUT_VALIDATION" in source
    assert "Holdout normalized accuracy" in source
    assert "/content/math-vqa-output/submission.csv" in source
    assert "experiment_log_h100_qwen.csv" in source
