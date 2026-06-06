import pandas as pd

from call_asr.proxy_data import degrade_manifest_for_call_center, validate_proxy_manifest


def test_validate_proxy_manifest_accepts_required_columns(tmp_path):
    manifest = tmp_path / "proxy.csv"
    pd.DataFrame(
        {
            "file_name": ["cv_th_001.wav"],
            "audio_path": ["/tmp/cv_th_001.wav"],
            "text": ["สวัสดีค่ะ"],
            "source": ["common_voice_th"],
            "split": ["validation"],
        }
    ).to_csv(manifest, index=False)

    df = validate_proxy_manifest(manifest)

    assert df["source"].tolist() == ["common_voice_th"]


def test_degrade_manifest_for_call_center_adds_transform_columns():
    df = pd.DataFrame(
        {
            "file_name": ["cv_th_001.wav"],
            "audio_path": ["/tmp/cv_th_001.wav"],
            "text": ["สวัสดีค่ะ"],
            "source": ["common_voice_th"],
            "split": ["validation"],
        }
    )

    degraded = degrade_manifest_for_call_center(df)

    assert degraded.loc[0, "transform"] == "call_center_degraded"
    assert degraded.loc[0, "target_sample_rate"] == 16000
    assert degraded.loc[0, "bandpass_hz"] == "300-3400"
