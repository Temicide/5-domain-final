from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioMetadata:
    file_name: str
    prefix: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    decode_ok: bool
    error: str


def audio_prefix(file_name: str) -> str:
    return file_name.split("_", 1)[0]


def probe_audio_file(path: Path) -> AudioMetadata:
    try:
        info = sf.info(path)
        channels = int(info.channels)
        sample_rate = int(info.samplerate)
        frames = int(info.frames)
        duration_seconds = float(frames / sample_rate) if sample_rate else 0.0
        return AudioMetadata(
            file_name=path.name,
            prefix=audio_prefix(path.name),
            sample_rate=sample_rate,
            channels=channels,
            frames=frames,
            duration_seconds=duration_seconds,
            decode_ok=True,
            error="",
        )
    except Exception as exc:
        return AudioMetadata(
            file_name=path.name,
            prefix=audio_prefix(path.name),
            sample_rate=0,
            channels=0,
            frames=0,
            duration_seconds=0.0,
            decode_ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def load_audio_16khz(path: Path) -> tuple[np.ndarray, int]:
    audio, source_sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_sample_rate != TARGET_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=source_sample_rate, target_sr=TARGET_SAMPLE_RATE)
    return np.asarray(mono, dtype=np.float32), TARGET_SAMPLE_RATE
