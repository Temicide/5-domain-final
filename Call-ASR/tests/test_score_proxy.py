import pandas as pd

from call_asr.score_proxy import character_error_rate, score_predictions


def test_character_error_rate_uses_edit_distance_over_reference_length():
    assert character_error_rate("abc", "axbc") == 1 / 3
    assert character_error_rate("สวัสดี", "สวัสดี") == 0.0


def test_score_predictions_returns_overall_and_by_prefix():
    reference = pd.DataFrame(
        {
            "file_name": ["RSP_001_audio.wav", "AU_001_audio.wav"],
            "text": ["abc", "สวัสดี"],
        }
    )
    predictions = pd.DataFrame(
        {
            "file_name": ["AU_001_audio.wav", "RSP_001_audio.wav"],
            "normalized_text": ["สวัสดี", "axbc"],
        }
    )

    result = score_predictions(reference, predictions)

    assert round(result["overall_cer"], 4) == round(1 / 9, 4)
    assert result["by_prefix"]["AU"]["count"] == 1
    assert result["by_prefix"]["AU"]["cer"] == 0.0
    assert result["by_prefix"]["RSP"]["count"] == 1
    assert round(result["by_prefix"]["RSP"]["cer"], 4) == round(1 / 3, 4)
