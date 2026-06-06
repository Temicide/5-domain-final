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
