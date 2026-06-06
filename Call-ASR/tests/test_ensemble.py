import pandas as pd

from call_asr.ensemble import confidence_select, prefix_route


def test_prefix_route_chooses_configured_model_by_file_prefix():
    typhoon = pd.DataFrame(
        {
            "file_name": ["AU_001_audio.wav", "RSP_001_audio.wav"],
            "normalized_text": ["typhoon au", "typhoon rsp"],
            "model_name": ["typhoon", "typhoon"],
            "avg_logprob": [-0.1, -0.1],
            "compression_ratio": [1.0, 1.0],
            "no_speech_prob": [0.01, 0.01],
        }
    )
    pathumma = pd.DataFrame(
        {
            "file_name": ["AU_001_audio.wav", "RSP_001_audio.wav"],
            "normalized_text": ["pathumma au", "pathumma rsp"],
            "model_name": ["pathumma", "pathumma"],
            "avg_logprob": [-0.2, -0.2],
            "compression_ratio": [1.0, 1.0],
            "no_speech_prob": [0.01, 0.01],
        }
    )

    routed = prefix_route(
        candidates={"typhoon": typhoon, "pathumma": pathumma},
        route_by_prefix={"AU": "pathumma"},
        default_model="typhoon",
    )

    assert routed[["file_name", "normalized_text", "selected_model"]].to_dict("records") == [
        {"file_name": "AU_001_audio.wav", "normalized_text": "pathumma au", "selected_model": "pathumma"},
        {"file_name": "RSP_001_audio.wav", "normalized_text": "typhoon rsp", "selected_model": "typhoon"},
    ]


def test_confidence_select_prefers_higher_logprob_then_lower_no_speech():
    model_a = pd.DataFrame(
        {
            "file_name": ["RSP_001_audio.wav"],
            "normalized_text": ["ก"],
            "model_name": ["a"],
            "avg_logprob": [-0.5],
            "compression_ratio": [1.1],
            "no_speech_prob": [0.01],
        }
    )
    model_b = pd.DataFrame(
        {
            "file_name": ["RSP_001_audio.wav"],
            "normalized_text": ["ข"],
            "model_name": ["b"],
            "avg_logprob": [-0.2],
            "compression_ratio": [1.2],
            "no_speech_prob": [0.03],
        }
    )

    selected = confidence_select({"a": model_a, "b": model_b})

    assert selected.loc[0, "normalized_text"] == "ข"
    assert selected.loc[0, "selected_model"] == "b"
