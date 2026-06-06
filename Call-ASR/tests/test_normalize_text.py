import pytest

from call_asr.normalize_text import normalize_text


@pytest.mark.parametrize(
    ("policy", "raw", "expected"),
    [
        ("raw", "  สวัสดีค่ะ!!!  ", "สวัสดีค่ะ!!!"),
        ("single_space", "สวัสดี   ค่ะ\nโทร  123", "สวัสดี ค่ะ โทร 123"),
        ("no_spaces", "สวัสดี   ค่ะ โทร 123", "สวัสดีค่ะโทร123"),
        ("thai_chars_only_light", "ค่ะ! โทร ABC 123 😊", "ค่ะ โทร ABC 123"),
        ("remove_fillers", "เอ่อ สวัสดี อืม ค่ะ", "สวัสดี ค่ะ"),
    ],
)
def test_normalize_text_named_policies(policy, raw, expected):
    assert normalize_text(raw, policy) == expected


def test_normalize_text_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unknown normalization policy: missing"):
        normalize_text("สวัสดี", "missing")
