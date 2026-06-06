from pathlib import Path

import pytest
import requests

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


def test_query_llava_retries_timeout_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"abc")
    calls = {"count": 0}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": "24"}}

    def fake_post(url: str, json: dict, timeout: int) -> FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.ReadTimeout("slow image")
        return FakeResponse()

    monkeypatch.setattr("math_vqa.ollama_client.requests.post", fake_post)
    monkeypatch.setattr("math_vqa.ollama_client.time.sleep", lambda seconds: None)

    response = query_llava(image_path, "Solve this", timeout=3, retries=1, retry_sleep=0)

    assert response == "24"
    assert calls["count"] == 2


def test_query_llava_raises_readable_timeout_after_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"abc")

    def fake_post(url: str, json: dict, timeout: int) -> object:
        raise requests.ReadTimeout("slow image")

    monkeypatch.setattr("math_vqa.ollama_client.requests.post", fake_post)
    monkeypatch.setattr("math_vqa.ollama_client.time.sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="Ollama request timed out after 3 seconds"):
        query_llava(image_path, "Solve this", timeout=3, retries=1, retry_sleep=0)


def test_assert_ollama_ready_raises_readable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRequests:
        @staticmethod
        def get(url: str, timeout: int) -> object:
            raise OSError("connection refused")

    monkeypatch.setattr("math_vqa.ollama_client.requests", FakeRequests)

    with pytest.raises(RuntimeError, match="Ollama is not ready"):
        assert_ollama_ready()
