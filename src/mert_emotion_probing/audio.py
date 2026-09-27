"""Audio loading and preprocessing for MERT input.

Core tensor conventions:

- loaded waveform: ``[channels, source_samples]``;
- mono waveform: ``[source_samples]``;
- resampled mono waveform: ``[target_samples]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as audio_functional


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded audio and source metadata.

    ``waveform`` is a float32 tensor with axes ``[channels, samples]``.
    """

    path: Path
    waveform: torch.Tensor
    sample_rate: int
    duration_seconds: float

    @property
    def channels(self) -> int:
        return int(self.waveform.shape[0])

    @property
    def samples(self) -> int:
        return int(self.waveform.shape[1])


def load_audio(path: Path) -> LoadedAudio:
    """Decode an audio file into ``[channels, samples]`` float32 waveform data."""

    samples_by_channels, sample_rate = sf.read(
        path, dtype="float32", always_2d=True
    )
    waveform = torch.from_numpy(samples_by_channels.T.copy())
    return LoadedAudio(
        path=path,
        waveform=waveform,
        sample_rate=int(sample_rate),
        duration_seconds=waveform.shape[1] / float(sample_rate),
    )


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Return ``[samples]`` audio using arithmetic channel mean when needed.

    A single-channel input is only squeezed; its values are not averaged or
    otherwise modified.
    """

    if waveform.ndim != 2:
        raise ValueError(
            f"Expected [channels, samples], received shape {tuple(waveform.shape)}"
        )
    if waveform.shape[0] < 1:
        raise ValueError("Waveform must contain at least one channel")
    if waveform.shape[0] == 1:
        return waveform[0]
    return waveform.mean(dim=0)


def resample_mono(
    waveform: torch.Tensor, source_rate: int, target_rate: int = 24_000
) -> torch.Tensor:
    """Resample a mono ``[samples]`` waveform while preserving duration."""

    if waveform.ndim != 1:
        raise ValueError(f"Expected mono [samples], received {tuple(waveform.shape)}")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if source_rate == target_rate:
        return waveform
    return audio_functional.resample(waveform, source_rate, target_rate)
