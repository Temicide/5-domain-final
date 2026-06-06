from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session", autouse=True)
def write_audio_fixtures():
    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "audio"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    timeline = np.linspace(0, 0.25, int(sample_rate * 0.25), endpoint=False)
    signal = 0.05 * np.sin(2 * np.pi * 440 * timeline)
    for file_name in ["RSP_001_audio.wav", "AU_001_audio.wav", "SDB_001_audio.wav"]:
        sf.write(fixture_dir / file_name, signal, sample_rate)
