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


def prepare_mert_batch(
    waveforms: Sequence[torch.Tensor],
    feature_extractor: Wav2Vec2FeatureExtractor,
    sample_rate: int = 24_000,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Dynamically pad mono waveforms and return inputs plus attention masks.

    Each input waveform has shape ``[samples]``. Padding is applied only to the
    longest waveform in this batch. The returned attention mask is one for real
    waveform samples and zero for batch-local padding.
    """

    if not waveforms:
        raise ValueError("At least one waveform is required")
    for index, waveform in enumerate(waveforms):
        if waveform.ndim != 1:
            raise ValueError(
                f"Waveform {index} must have shape [samples], "
                f"received {tuple(waveform.shape)}"
            )
    inputs = feature_extractor(
        [waveform.cpu().numpy() for waveform in waveforms],
        sampling_rate=sample_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
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


def feature_frame_mask(
    model: torch.nn.Module,
    input_attention_mask: torch.Tensor,
    feature_sequence_length: int,
) -> torch.Tensor:
    """Convert a waveform attention mask to MERT feature-frame validity."""

    if input_attention_mask.ndim != 2:
        raise ValueError(
            "Expected input attention mask [B, input_samples], received "
            f"{tuple(input_attention_mask.shape)}"
        )
    mask = model._get_feature_vector_attention_mask(
        feature_sequence_length, input_attention_mask
    )
    if mask.ndim != 2 or mask.shape[0] != input_attention_mask.shape[0]:
        raise RuntimeError(f"Unexpected feature-frame mask shape {tuple(mask.shape)}")
    return mask.bool()


def masked_temporal_mean_pool(
    hidden_states: Sequence[torch.Tensor], feature_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool valid frames and stack representations as ``[B, 13, 768]``."""

    if not hidden_states:
        raise ValueError("No hidden states were provided")
    expected_prefix = hidden_states[0].shape[:2]
    if tuple(feature_mask.shape) != tuple(expected_prefix):
        raise ValueError(
            f"Feature mask {tuple(feature_mask.shape)} does not match "
            f"hidden states {tuple(expected_prefix)}"
        )
    valid_counts = feature_mask.sum(dim=1)
    if bool((valid_counts == 0).any().item()):
        raise ValueError("Every sample must contain at least one valid feature frame")

    pooled_states = []
    for index, state in enumerate(hidden_states):
        if state.shape[:2] != expected_prefix:
            raise ValueError(
                f"Hidden state {index} has inconsistent shape {tuple(state.shape)}"
            )
        mask = feature_mask.unsqueeze(-1).to(dtype=state.dtype)
        pooled_states.append(
            (state * mask).sum(dim=1) / valid_counts.unsqueeze(-1).to(state.dtype)
        )
    pooled = torch.stack(pooled_states, dim=1)
    if pooled.shape[1:] != (
        EXPECTED_REPRESENTATIONS,
        EXPECTED_HIDDEN_DIMENSION,
    ):
        raise RuntimeError(f"Unexpected pooled shape {tuple(pooled.shape)}")
    return pooled
