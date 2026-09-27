"""Minimal frozen MERT loading, extraction, and temporal pooling utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor


DEFAULT_MODEL_ID = "m-a-p/MERT-v1-95M"
EXPECTED_REPRESENTATIONS = 13
EXPECTED_HIDDEN_DIMENSION = 768


def load_frozen_mert(
    model_id: str = DEFAULT_MODEL_ID, device: str | torch.device = "cuda"
) -> tuple[Wav2Vec2FeatureExtractor, torch.nn.Module]:
    """Load the official MERT feature extractor and a frozen evaluation model."""

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_id, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return feature_extractor, model


def prepare_mert_inputs(
    waveform: torch.Tensor,
    feature_extractor: Wav2Vec2FeatureExtractor,
    sample_rate: int = 24_000,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Create batched MERT inputs from one unpadded mono waveform.

    The input ``waveform`` has shape ``[samples]``. The returned
    ``input_values`` has shape ``[batch=1, samples]``. The official feature
    extractor performs its configured zero-mean/unit-variance normalization;
    this function applies no additional waveform normalization.
    """

    if waveform.ndim != 1:
        raise ValueError(f"Expected [samples], received {tuple(waveform.shape)}")
    inputs = feature_extractor(
        waveform.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt"
    )
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def extract_hidden_states(
    model: torch.nn.Module, model_inputs: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """Run frozen inference and return 13 tensors shaped ``[B, T, 768]``."""

    with torch.inference_mode():
        outputs = model(**model_inputs, output_hidden_states=True)
    hidden_states = tuple(outputs.hidden_states)
    if len(hidden_states) != EXPECTED_REPRESENTATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_REPRESENTATIONS} representations, "
            f"received {len(hidden_states)}"
        )
    for index, state in enumerate(hidden_states):
        if state.ndim != 3 or state.shape[-1] != EXPECTED_HIDDEN_DIMENSION:
            raise RuntimeError(
                f"Representation {index} has invalid shape {tuple(state.shape)}"
            )
    return hidden_states


def temporal_mean_pool(
    hidden_states: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Mean-pool unpadded time axes and stack layers as ``[B, 13, 768]``."""

    if not hidden_states:
        raise ValueError("No hidden states were provided")
    pooled = torch.stack([state.mean(dim=1) for state in hidden_states], dim=1)
    if pooled.ndim != 3 or pooled.shape[1:] != (
        EXPECTED_REPRESENTATIONS,
        EXPECTED_HIDDEN_DIMENSION,
    ):
        raise RuntimeError(f"Unexpected pooled shape {tuple(pooled.shape)}")
    return pooled
