import numpy as np
import soundfile as sf

from call_asr.audio import audio_prefix, load_audio_16khz, probe_audio_file


def test_audio_prefix_returns_text_before_first_underscore():
    assert audio_prefix("AU_001_audio.wav") == "AU"
    assert audio_prefix("RSP_101_audio.wav") == "RSP"


def test_probe_audio_file_reports_duration_sample_rate_and_channels(tmp_path):
    wav_path = tmp_path / "SDB_001_audio.wav"
    samples = np.zeros(8000, dtype=np.float32)
    sf.write(wav_path, samples, 8000)

    metadata = probe_audio_file(wav_path)

    assert metadata.file_name == "SDB_001_audio.wav"
    assert metadata.prefix == "SDB"
    assert metadata.sample_rate == 8000
    assert metadata.channels == 1
    assert round(metadata.duration_seconds, 2) == 1.0
    assert metadata.decode_ok is True
    assert metadata.error == ""


def test_load_audio_16khz_resamples_and_returns_mono(tmp_path):
    wav_path = tmp_path / "AU_001_audio.wav"
    samples = np.zeros((4000, 2), dtype=np.float32)
    samples[:, 0] = 0.1
    samples[:, 1] = -0.1
    sf.write(wav_path, samples, 8000)

    audio, sample_rate = load_audio_16khz(wav_path)

    assert sample_rate == 16000
    assert audio.ndim == 1
    assert 7900 <= len(audio) <= 8100
