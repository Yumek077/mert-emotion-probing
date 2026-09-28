# Module A — Stage 5: Real DEAM → MERT End-to-End Sanity Check

## Purpose

This stage verifies the real-audio implementation path from a small, fixed set of DEAM MP3 excerpts to item-level MERT representations:

`DEAM MP3 → decoded waveform → channel handling → 24 kHz resampling → frozen MERT-v1-95M → 13 hidden representations → temporal mean pooling → [13, 768]`

This is an implementation sanity check, not dataset-wide representation extraction or an emotion-probing experiment. No model parameters were trained, no embedding cache was created, and no conclusion about emotion information in any layer is supported by these tests.

## Selected Samples

The four excerpts were selected deterministically from the Stage 4 inventory. No 2015 full song was used.

| Song ID | Selection reason | Source rate | Channels |
|---:|---|---:|---:|
| 10 | Ordinary 44.1 kHz stereo case | 44,100 Hz | 2 |
| 1198 | Non-44.1-kHz stereo case | 48,000 Hz | 2 |
| 811 | Confirmed mono case | 44,100 Hz | 1 |
| 1024 | Special low-rate stereo case | 16,000 Hz | 2 |

**Implementation choice:** Fixed IDs make the check repeatable and deliberately exercise each preprocessing branch.

**Interpretation:** These samples establish coverage of the intended loading, mono-conversion, and resampling paths; they are not a statistically representative subset of DEAM.

## Audio Loading

Audio was decoded with `soundfile` into `torch.float32`. The loader always returns a two-dimensional tensor shaped `[channels, source_samples]`; axis 0 is the channel axis and axis 1 is the decoded sample axis.

| Song ID | Raw waveform shape `[C, N]` | Dtype | Decoded duration |
|---:|---:|---|---:|
| 10 | `[2, 1,987,190]` | `torch.float32` | 45.060998 s |
| 1198 | `[2, 2,163,456]` | `torch.float32` | 45.072000 s |
| 811 | `[1, 1,984,896]` | `torch.float32` | 45.008980 s |
| 1024 | `[2, 720,001]` | `torch.float32` | 45.000063 s |

**Observed result:** All four MP3 files decoded successfully and retained their actual decoded lengths. No excerpt was cropped or padded to exactly 45 seconds.

## Channel Handling

**Implementation choice:** For `[C, N]` waveforms with `C > 1`, mono is the arithmetic mean over dimension 0. For `C == 1`, the singleton channel dimension is removed without averaging.

| Song ID | Before | Operation | After |
|---:|---:|---|---:|
| 10 | `[2, 1,987,190]` | Arithmetic channel mean | `[1,987,190]` |
| 1198 | `[2, 2,163,456]` | Arithmetic channel mean | `[2,163,456]` |
| 811 | `[1, 1,984,896]` | Remove singleton channel dimension | `[1,984,896]` |
| 1024 | `[2, 720,001]` | Arithmetic channel mean | `[720,001]` |

**Observed result:** For mono ID 811, the resulting one-dimensional waveform was exactly equal to the original channel values. The mono branch therefore did not accidentally rescale or otherwise change the samples.

## Resampling

Each mono waveform was resampled on the fly to 24,000 Hz with `torchaudio.functional.resample`.

| Song ID | Source rate | Source samples | Target samples | Before | After | Duration delta |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 44,100 Hz | 1,987,190 | 1,081,464 | 45.060998 s | 45.061000 s | +0.0023 ms |
| 1198 | 48,000 Hz | 2,163,456 | 1,081,728 | 45.072000 s | 45.072000 s | 0.0000 ms |
| 811 | 44,100 Hz | 1,984,896 | 1,080,216 | 45.008980 s | 45.009000 s | +0.0204 ms |
| 1024 | 16,000 Hz | 720,001 | 1,080,002 | 45.000063 s | 45.000083 s | +0.0208 ms |

**Observed result:** Duration was preserved to normal resampling-rounding precision for every tested rate, including 16 kHz → 24 kHz and 48 kHz → 24 kHz.

## Label Lookup

Labels were read from the previously verified unified static-annotation table and joined by integer `song_id`.

| Song ID | `valence_mean` | `arousal_mean` |
|---:|---:|---:|
| 10 | 4.0 | 4.7 |
| 1198 | 3.5 | 1.8 |
| 811 | 4.1 | 4.9 |
| 1024 | 3.2 | 2.9 |

**Observed result:** Each selected audio ID had exactly one matching pair of static averaged targets. The targets were only checked for correspondence; they were not used for fitting or inference.

## MERT Input

MERT-v1-95M was loaded with the Stage 3 compatible stack (`transformers==4.24.0`, `nnAudio==0.3.4`), placed on the NVIDIA RTX 4060 Laptop GPU, set to evaluation mode, and fully frozen. Inference ran with gradients disabled.

The official MERT feature extractor was called with `sampling_rate=24000`. Its loaded configuration reports `do_normalize=True`, so the processor's configured normalization was retained; no separate normalization was added in project code.

For each item, the processor produced `input_values` shaped `[B, N]`, where `B=1` is the item batch dimension and `N` is the number of 24 kHz waveform samples. All inputs were `torch.float32` on `cuda:0`:

| Song ID | MERT input shape |
|---:|---:|
| 10 | `[1, 1,081,464]` |
| 1198 | `[1, 1,081,728]` |
| 811 | `[1, 1,080,216]` |
| 1024 | `[1, 1,080,002]` |

## MERT Representations

Every forward pass returned 13 hidden-state tensors with hidden dimension 768. The indexing used throughout the implementation is:

- representation 0: pre-Transformer representation (the encoder input representation returned in `hidden_states[0]`)
- representations 1–12: outputs of Transformer Layers 1–12

| Song ID | Representation count | Shape of every representation | `T` | `D` |
|---:|---:|---:|---:|---:|
| 10 | 13 | `[1, 3379, 768]` | 3379 | 768 |
| 1198 | 13 | `[1, 3380, 768]` | 3380 | 768 |
| 811 | 13 | `[1, 3375, 768]` | 3375 | 768 |
| 1024 | 13 | `[1, 3374, 768]` | 3374 | 768 |

**Observed result:** Temporal length varied with the actual decoded waveform length, as expected; it was not assumed to be identical across approximately 45-second excerpts.

## Tensor Shape Trace

The following is the actual trace for representative sample ID 10:

```text
Raw waveform:                         [2, 1987190]
  axis 0 = channels; axis 1 = source samples at 44,100 Hz
After arithmetic-mean mono:          [1987190]
After resampling to 24,000 Hz:        [1081464]
Before MERT (batch dimension added):  [1, 1081464]

Representation 0  (pre-Transformer): [1, 3379, 768]
Representation 1  (Layer 1):         [1, 3379, 768]
Representation 2  (Layer 2):         [1, 3379, 768]
Representation 3  (Layer 3):         [1, 3379, 768]
Representation 4  (Layer 4):         [1, 3379, 768]
Representation 5  (Layer 5):         [1, 3379, 768]
Representation 6  (Layer 6):         [1, 3379, 768]
Representation 7  (Layer 7):         [1, 3379, 768]
Representation 8  (Layer 8):         [1, 3379, 768]
Representation 9  (Layer 9):         [1, 3379, 768]
Representation 10 (Layer 10):        [1, 3379, 768]
Representation 11 (Layer 11):        [1, 3379, 768]
Representation 12 (Layer 12):        [1, 3379, 768]

Mean-pooled representation 0:        [1, 768]
Mean-pooled representation 1:        [1, 768]
Mean-pooled representation 2:        [1, 768]
Mean-pooled representation 3:        [1, 768]
Mean-pooled representation 4:        [1, 768]
Mean-pooled representation 5:        [1, 768]
Mean-pooled representation 6:        [1, 768]
Mean-pooled representation 7:        [1, 768]
Mean-pooled representation 8:        [1, 768]
Mean-pooled representation 9:        [1, 768]
Mean-pooled representation 10:       [1, 768]
Mean-pooled representation 11:       [1, 768]
Mean-pooled representation 12:       [1, 768]

Stacked batch representation:        [1, 13, 768]
Final item representation:           [13, 768]
```

## Temporal Mean Pooling

**Implementation choice:** Each `[B, T, 768]` hidden state was averaged over temporal dimension 1, producing `[B, 768]`. The 13 pooled representations were stacked along a representation dimension, producing `[B, 13, 768]`; removing the sanity-check batch dimension yielded `[13, 768]` per audio item.

**Observed result:** All four items produced the expected `[13, 768]` item matrix. No item-level embeddings were written to disk.

## Numerical Sanity Checks

For every sample and every one of the 13 pooled vectors, the runner recorded finiteness, mean, population standard deviation, and L2 norm.

**Observed result:** All 52 pooled vectors were finite. None contained NaN or infinity, and none was an all-zero vector. Means, standard deviations, and L2 norms were non-degenerate and are retained in the ignored machine-readable verification JSON.

**Interpretation boundary:** These statistics indicate that the numerical pipeline is not obviously broken. They do not show that MERT encodes Valence or Arousal, nor do they support comparing layers for emotion prediction.

## Cross-Sample Check

The four `[13, 768]` matrices were flattened only for a lightweight pairwise implementation check.

| Song IDs | L2 distance | Cosine similarity | Exactly equal |
|---|---:|---:|---|
| 10 / 1198 | 10.4628 | 0.8129 | No |
| 10 / 811 | 9.4830 | 0.8408 | No |
| 10 / 1024 | 11.2632 | 0.7879 | No |
| 1198 / 811 | 13.5584 | 0.7043 | No |
| 1198 / 1024 | 12.1788 | 0.7710 | No |
| 811 / 1024 | 11.3730 | 0.7960 | No |

**Observed result:** Different real audio inputs did not produce identical pooled representations.

**Interpretation boundary:** This is not a layer ranking, similarity analysis, or emotion result. It only confirms input-sensitive output at the end of the implemented path.

## GPU / Runtime Observation

All measurements used batch size 1, FP32 model/input tensors, evaluation mode, gradients disabled, and `output_hidden_states=True` on the NVIDIA RTX 4060 Laptop GPU.

| Song ID | Total time | Forward + pooling | Peak allocated VRAM | Peak reserved VRAM |
|---:|---:|---:|---:|---:|
| 10 | 0.701 s | 0.688 s | 1617.65 MiB | 2316 MiB |
| 1198 | 0.399 s | 0.388 s | 1618.12 MiB | 2720 MiB |
| 811 | 0.406 s | 0.397 s | 1615.94 MiB | 2712 MiB |
| 1024 | 0.530 s | 0.520 s | 1615.86 MiB | 2712 MiB |

The first item includes normal first-use effects, and these single runs are implementation observations rather than a controlled benchmark. All four approximately 45-second single-pass forwards completed without CUDA out-of-memory errors.

## Implementation Added

- `src/mert_emotion_probing/audio.py`: MP3 loading, explicit channel handling, and on-the-fly mono resampling.
- `src/mert_emotion_probing/mert.py`: frozen MERT/feature-extractor loading, input preparation, 13-state extraction, and temporal mean pooling.
- `src/mert_emotion_probing/__init__.py`: minimal package marker.
- `scripts/sanity_check_deam_mert.py`: deterministic four-item end-to-end runner with label lookup, tensor traces, numerical checks, cross-sample checks, and runtime/VRAM observations.

Core functions contain no dataset-specific absolute paths. The runner's paths are repository-relative and can be overridden by command-line arguments. It writes only summaries—not embeddings—to the ignored verification area:

`data/raw/deam/verification/deam_mert_e2e_sanity.json`

## Problems Encountered

- For ID 1198, Stage 4's MP3 container-level `sf.info` inventory reported 45.224063 seconds, while full decoding in this stage yielded 2,163,456 samples at 48 kHz, or 45.072 seconds. Stage 5 uses the actual decoded sample count for preprocessing and duration preservation. This metadata-versus-decoding discrepancy did not prevent loading, resampling, or MERT inference and should remain documented for future duration audits.
- Loading emitted non-fatal library warnings about Hugging Face download resumption, pinning a remote-code revision, and PyTorch weight-norm deprecation. There were no missing weights, dependency failures, CUDA errors, or failed samples.

No MERT source was modified and no workaround or monkey patch was used.

## Stage 5 Conclusion

**Observed result:** The complete real-DEAM preprocessing and frozen-representation path was operational for 44.1 kHz stereo, 48 kHz stereo, 44.1 kHz mono, and 16 kHz stereo inputs. Every sample produced 13 valid hidden representations and a finite `[13, 768]` mean-pooled item representation within the available GPU memory.

**Interpretation:** This result verifies implementation compatibility and tensor flow only. It does not establish that any MERT layer predicts musical emotion or that one layer is preferable for Valence or Arousal.

## Readiness for Module B

The preprocessing and representation-extraction pipeline is operational and provides reusable functions for later work. Subject to researcher review and the separate Module A closure step, it is technically ready to support Module B planning.

Dataset-wide extraction, embedding-cache design, batching policy, data splits, probe training, baselines, and formal ablations have not begun.
