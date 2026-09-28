# Stage B1 Report — MERT Representation Inspection / Sanity Check

Date: 2026-09-28

Status: Passed

## 1. What was inspected

Stage B1 reused the Module A audio and frozen-MERT implementation and inspected three real DEAM excerpts. The check covered decoded waveform shape, channel handling, 24 kHz resampling, model input construction, returned hidden-state count and shapes, implementation-level hidden-state/layer mapping, temporal mean pooling, valid temporal positions, dtype/device, numerical finiteness, gradient state, parameter freezing, and evaluation mode.

No dataset-wide extraction, embedding cache, probe, baseline, split change, or new research decision was introduced.

## 2. Files created or modified

- Modified `scripts/sanity_check_deam_mert.py` to accept explicit sample IDs and an optional local model snapshot, and to record layer-mapping evidence, forward-hook mapping checks, attention-mask/valid-position checks, tensor dtype/device/gradient state, and hidden-state NaN/Inf checks.
- Created this report: `docs/codex_reports/module_b_stage1_mert_representation_inspection.md`.
- Created the small, Git-ignored inspection log `data/raw/deam/verification/stage_b1_mert_representation_inspection.json`. It contains metadata and numerical summaries only; no embeddings were saved.

The reusable Module A extraction and pooling functions in `src/mert_emotion_probing/` were not changed.

## 3. Samples used

| DEAM ID | Reason | Decoded waveform `[C, N]` | Source rate | Decoded duration | Mono/resampled samples | Model input |
|---:|---|---:|---:|---:|---:|---:|
| 10 | Ordinary stereo excerpt | `[2, 1,987,190]` | 44,100 Hz | 45.060998 s | 1,081,464 | `[1, 1,081,464]` |
| 811 | Mono passthrough branch | `[1, 1,984,896]` | 44,100 Hz | 45.008980 s | 1,080,216 | `[1, 1,080,216]` |
| 1024 | 16 kHz stereo resampling branch | `[2, 720,001]` | 16,000 Hz | 45.000063 s | 1,080,002 | `[1, 1,080,002]` |

Stereo inputs were converted to mono by arithmetic channel mean. The mono input was only squeezed and remained exactly equal to its original channel values. All excerpts retained their actually decoded durations; none was cropped or padded to exactly 45 seconds.

## 4. Observed tensor shapes

Every sample returned 13 hidden states. All 13 states within a sample had the same observed shape:

| DEAM ID | Hidden states | Each batched hidden state | Each item hidden state | Valid `T` | Hidden dimension | Final item matrix |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 13 | `[1, 3379, 768]` | `[3379, 768]` | 3379 | 768 | `[13, 768]` |
| 811 | 13 | `[1, 3375, 768]` | `[3375, 768]` | 3375 | 768 | `[13, 768]` |
| 1024 | 13 | `[1, 3374, 768]` | `[3374, 768]` | 3374 | 768 | `[13, 768]` |

Waveforms, model inputs, hidden states, and pooled outputs were `torch.float32`. Model inputs and hidden states were on `cuda:0` (NVIDIA GeForce RTX 4060 Laptop GPU). All hidden states and all pooled vectors were finite, with no NaN or Inf, and no pooled vector was all zero.

## 5. Hidden-state / layer mapping

The mapping was verified from the loaded implementation and from the actual forward execution, rather than assumed from the number of returned tensors.

- Loaded model class: `MERTModel` from the cached MERT remote implementation.
- Encoder class: Transformers 4.24.0 `HubertEncoder`.
- Model configuration: `num_hidden_layers = 12`, `hidden_size = 768`.
- Actual encoder module count: 12 Transformer layers.
- In the loaded encoder implementation, positional convolution, layer normalization, and dropout are applied before the layer loop. The current state is appended to `all_hidden_states` immediately before each Transformer layer, and the final state is appended after the loop.
- Forward hooks on sample 10 verified exact tensor equality between `hidden_states[0]` and the input to Transformer Layer 1. They also verified exact equality between each `hidden_states[i]`, for `i = 1..12`, and the output of Transformer Layer `i`.

Therefore the project mapping is correct:

| Hidden-state index | Conceptual representation |
|---:|---|
| 0 | Pre-Transformer representation: encoder state after positional convolution/layer normalization/dropout and immediately before Transformer Layer 1 |
| 1 | Transformer Layer 1 output |
| 2 | Transformer Layer 2 output |
| ... | ... |
| 12 | Transformer Layer 12 output |

## 6. Pooling verification

For each sample and each representation, temporal mean pooling changed the item tensor from `[T, 768]` to `[768]`. Stacking the 13 pooled vectors produced `[13, 768]`.

The official feature extractor returned an input attention mask for each single-item batch. Every mask value was one, its valid-input count equaled the complete resampled waveform length, and the model-derived feature mask marked all `T` output frames as valid. No padding was applied.

The existing unmasked mean over all `T` positions was compared with an explicit valid-mask mean. They were numerically equal within `rtol=1e-6, atol=1e-7` for all three samples. The maximum absolute difference was `2.384185791015625e-07`, attributable to FP32 reduction order; there was no invalid-frame contribution.

## 7. Protocol consistency

The inspected implementation and run were consistent with the locked Module A protocol:

- DEAM excerpts and one audio item per sample;
- actual decoded excerpt length retained;
- arithmetic-mean mono conversion for multichannel audio;
- on-the-fly resampling to 24 kHz;
- `m-a-p/MERT-v1-95M`, FP32, frozen parameters, evaluation mode;
- `torch.inference_mode()` with gradients disabled during the encoder forward;
- Pre-Transformer plus Transformer Layers 1–12;
- hidden dimension 768;
- temporal mean over all valid positions;
- `[13, 768]` pooled representation per item.

The run used Transformers 4.24.0 and PyTorch 2.14.0+cu130, matching the Module A compatibility environment. All model parameters had `requires_grad=False`; all returned hidden states had `requires_grad=False`; the hook observed `torch.is_grad_enabled() == False` and inference mode enabled inside the encoder; and `model.training == False`.

## 8. Warnings or unexpected findings

- The feature extractor returned an all-ones attention mask even for an unpadded single-item input. This is not a protocol conflict: all input samples and all resulting frames were valid, and masked versus ordinary pooling agreed within FP32 tolerance.
- The explicit valid-mask sum/divide and `Tensor.mean` were not bitwise identical; their maximum difference was `2.384185791015625e-07`. They passed the stated numerical tolerance.
- Loading the already cached local snapshot emitted non-fatal warnings recommending an explicit remote-code revision and noting PyTorch's legacy weight-normalization API. No missing, unused, unexpected, or newly initialized weight warning appeared.
- The local cached snapshot was commit `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`. Offline loading was used to prevent a network check or model-version drift during this inspection; the canonical checkpoint remained `m-a-p/MERT-v1-95M`.

## 9. Exact commands used

Syntax check:

```powershell
& 'C:\Users\Rinshinozaki\miniconda3\Scripts\conda.exe' run -n mert-emotion python -m py_compile scripts\sanity_check_deam_mert.py
```

Final Stage B1 inspection run:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_MODULES_CACHE=Join-Path $env:TEMP 'mert-b1-hf-modules'
$snapshot='C:\Users\Rinshinozaki\.cache\huggingface\hub\models--m-a-p--MERT-v1-95M\snapshots\12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
& 'C:\Users\Rinshinozaki\miniconda3\Scripts\conda.exe' run -n mert-emotion python scripts\sanity_check_deam_mert.py --sample-ids 10 811 1024 --model-source $snapshot --output-json data\raw\deam\verification\stage_b1_mert_representation_inspection.json
```

Repository checks:

```powershell
git diff --check
git status --short
```

## 10. Conclusion: whether Stage B1 passed

**Stage B1 passed.** The current MERT/Transformers implementation returned exactly 13 correctly mapped representations, each with hidden dimension 768. Real DEAM samples followed the locked preprocessing protocol, all temporal positions used by pooling were valid, each layer pooled from `[T, 768]` to `[768]`, and the stacked item representation was `[13, 768]`. The model was frozen, in evaluation and inference mode, and all inspected tensors were finite.

This conclusion is limited to representation inspection and implementation sanity checking. No full-dataset extraction, caching, dataset construction beyond these inspection samples, or probing was performed.
