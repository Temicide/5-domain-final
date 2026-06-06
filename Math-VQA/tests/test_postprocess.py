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


def test_clean_model_answer_uses_fallback_for_blurry_image_request() -> None:
    cleaned = clean_model_answer(
        "The image appears to be too small and blurry. Please provide a clearer image.",
        fallback_answer="7",
    )

    assert cleaned.answer == "7"
    assert cleaned.used_fallback is True


def test_clean_model_answer_extracts_declared_final_answer_from_verbose_output() -> None:
    cleaned = clean_model_answer(
        'The image contains a math problem. The final answer in this case is "-2".',
        fallback_answer="1",
    )

    assert cleaned.answer == "-2"
    assert cleaned.used_fallback is False


def test_clean_model_answer_extracts_final_answer_from_next_line() -> None:
    cleaned = clean_model_answer("The final answer is:\n\n2.1886", fallback_answer="1")

    assert cleaned.answer == "2.1886"
    assert cleaned.used_fallback is False
