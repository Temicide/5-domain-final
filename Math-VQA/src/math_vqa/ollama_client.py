from __future__ import annotations

import base64
from pathlib import Path
import time
from typing import Any

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


def _extract_text_response(body: dict[str, Any]) -> str:
    message = body.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])
    if body.get("response") is not None:
        return str(body["response"])
    raise RuntimeError("Ollama response did not include message.content or response")


def query_llava(
    image_path: str | Path,
    prompt: str,
    model: str = "llava",
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: int = 600,
    retries: int = 1,
    retry_sleep: float = 5.0,
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
    attempts = max(0, retries) + 1
    for attempt in range(attempts):
        try:
            response = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise RuntimeError("Ollama response was not a JSON object")
            return _extract_text_response(body)
        except requests.Timeout as exc:
            if attempt < attempts - 1:
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(
                f"Ollama request timed out after {timeout} seconds for {image_path}. "
                "Increase OLLAMA_REQUEST_TIMEOUT, reduce image resolution/model size, or rely on fallback logging."
            ) from exc
        except requests.RequestException as exc:
            if attempt < attempts - 1:
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(f"Ollama request failed for {image_path}: {exc}") from exc

    raise RuntimeError("Ollama request failed without a captured exception")
