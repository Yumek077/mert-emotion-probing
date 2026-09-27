# Module A — Data and Representation Protocol

## 1. Module Goal

Module A establishes the data population, audio preprocessing, software environment, representation indexing, pooling rule, and hardware feasibility required before formal emotion probing. It turns a broad idea—probing musical emotion in pretrained MERT representations—into a reproducible input-to-representation protocol grounded in the downloaded DEAM release and real model execution.

No regression probe was trained in this module. The observations below concern dataset integrity and implementation behavior, not emotion-prediction performance.

## 2. Research Question Context

The project asks whether continuous static Valence and Arousal targets can be decoded from a frozen pretrained music model, and how that decodability varies across representation depth. MERT-v1-95M is therefore used as a representation extractor rather than fine-tuned as an emotion model. Later, a simple regression probe will be fitted separately to each representation.

A controlled protocol is necessary because apparent layer differences could otherwise reflect inconsistent audio duration, source format, preprocessing, pooling, checkpoint loading, or dataset composition. Module A fixes those choices before any target-aware modeling.

## 3. DEAM Dataset Investigation

The [official DEAM release](https://cvml.unige.ch/databases/DEAM/) contains audio, static whole-item annotations, dynamic time-varying annotations, and metadata. The [official manual](https://cvml.unige.ch/databases/DEAM/manual.pdf) describes Valence as the pleasantness/positivity dimension and Arousal as the activation/energy dimension of emotion.

Static annotations provide one averaged Valence value and one averaged Arousal value per audio item. Dynamic annotations describe changing emotion over time. The primary research question is item-level continuous regression, so the static averaged labels align directly with the desired supervised unit. Dynamic prediction remains a possible future extension but is not part of the primary experiment.

Three sample definitions were considered conceptually:

1. one audio item with one pair of static labels;
2. fixed-length clips inheriting the parent item's static labels;
3. time windows paired with dynamic annotations.

The final primary definition is **one DEAM audio item = one supervised sample**. This avoids duplicating a static label across pseudo-independent clips and keeps the target definition aligned with the annotation unit.

## 4. Dataset Verification

The official audio, annotation, and metadata archives were downloaded locally and audited with `scripts/verify_deam.py`. Raw files and generated verification tables remain outside version control.

### Audio and label integrity

- 1,802 MP3 files were present and all 1,802 were readable.
- There were 1,802 unique numeric audio IDs and 1,802 unique static-label IDs.
- All 1,802 audio IDs matched static labels.
- No audio item lacked a static Valence or Arousal mean.
- No duplicate audio or static-label IDs were found.
- No malformed numeric audio filenames were found.

### Duration and subset structure

- 1,744 items associated with the `metadata_2013.csv` and `metadata_2014.csv` files are approximately 45-second excerpts.
- Their observed durations range from 44.643 to 45.605 seconds (mean 45.052 seconds, SD 0.098 seconds).
- 58 items associated with `metadata_2015.csv` are metadata-defined full songs, ranging from 49.563 to 628.616 seconds.
- “Full song” is defined by the official metadata subset, not an arbitrary duration threshold; two metadata-defined full songs are shorter than 60 seconds.

The metadata filenames provide subset provenance. They should not automatically be interpreted as independent recording years, experimental splits, or identical annotation procedures.

### Source-format heterogeneity

The real archive is not uniformly 44.1 kHz stereo:

| Source property | Verified count |
|---|---:|
| 44,100 Hz | 1,778 |
| 48,000 Hz | 20 |
| 22,050 Hz | 3 |
| 16,000 Hz | 1 |
| Stereo | 1,789 |
| Mono | 13 |

This audit is the reason preprocessing reads the actual sample rate and channel count rather than relying on a dataset-wide assumption.

### Static-label observations

The official static rating scale is 1–9. The observed averaged labels occupy a narrower range: Valence 1.6–8.4 and Arousal 1.6–8.1. No label rescaling or normalization was applied during verification. Across all 1,802 items, the descriptive Pearson correlation between the two static means was positive (`r = 0.570`). This is a distributional observation, not a predictive result or causal relationship.

## 5. Final Dataset Scope

The primary study population is the **1,744 approximately 45-second DEAM excerpts**. The 58 `metadata_2015` full-song items are excluded from the primary experiment.

This decision is based on experimental control rather than an audio-quality judgment:

- the 1,744 excerpts have highly homogeneous duration;
- full-song status is completely tied to the `metadata_2015` subset;
- the full songs require substantially different duration handling and were collected under a different annotation design;
- including them would combine duration, subset, and annotation-design differences;
- they add only 58 items, approximately 3.2% of the complete release.

The trade-off is explicit: the primary results will describe the DEAM excerpt collection, not the complete 1,802-item DEAM release. The full songs are not described as low quality.

The primary duration strategy is a single pass over each actual approximately 45-second excerpt. Excerpts will not be cropped or padded merely to force exactly 45.000 seconds. A later 5-second chunk-and-aggregate ablation is planned to test sensitivity to context length and representation-extraction strategy; it is not part of Module A execution.

## 6. Audio Preprocessing Protocol

The executable order is:

```text
decode the source MP3 and inspect its real sample rate/channel count
→ represent the waveform as [channels, source_samples]
→ if multichannel, take the arithmetic mean over channels
→ if mono, remove only the singleton channel dimension
→ resample the one-dimensional waveform on the fly to 24,000 Hz
→ pass it to the official MERT feature extractor
```

The actual decoded sample count is authoritative. Resampling preserves the decoded duration apart from normal integer-sample rounding. No project-level peak, loudness, or other additional normalization is applied. The official feature extractor's configured preprocessing is retained (`do_normalize=True`).

At present, extraction uses batch size 1 and therefore has no padding contribution to temporal pooling. If variable-length batching is introduced later, valid-frame masking must be handled explicitly.

## 7. MERT Compatibility Investigation

MERT-v1-95M relies on checkpoint-specific remote modeling code. Three Transformers versions were tested rather than assuming that either the newest release or a merely executable configuration was scientifically acceptable.

| Transformers version | Observed behavior | Decision |
|---|---|---|
| 5.17.0 | The feature extractor and model loaded and forward computation ran, but `outputs.hidden_states` was `None` even with `output_hidden_states=True`. | Rejected because layer-wise probing requires the hidden-state sequence. |
| 4.44.0 | Thirteen hidden states were returned, but loading reported unused legacy positional-convolution weights and newly initialized parametrized positional-convolution weights. | Rejected because the checkpoint was not loaded intact. |
| 4.24.0 | This matches the version recorded in the checkpoint-era configuration. After installing `nnAudio==0.3.4`, the model loaded without missing, unused, unexpected, or newly initialized checkpoint-weight warnings. | Selected and verified. |

The selected environment uses `transformers==4.24.0`, `nnAudio==0.3.4`, frozen FP32 MERT-v1-95M, evaluation mode, disabled gradients, and an NVIDIA RTX 4060 Laptop GPU. No MERT source modification or monkey patch was used.

The reproducibility lesson is specific but important: **“the code runs” is insufficient evidence that a pretrained representation model loaded correctly**. The requested outputs and checkpoint-loading diagnostics must also be checked. This observation is about the tested MERT checkpoint/version combinations, not a claim that all other Transformers versions are generally unreliable.

## 8. Representation Definition

For an item and representation index `l`, let the hidden sequence be:

`H_l ∈ R^(T × 768)`

where `T` is the input-dependent temporal frame count after MERT's convolutional frontend and 768 is the hidden dimension.

The project uses 13 representations with the following fixed convention:

| Representation index | Name |
|---:|---|
| 0 | Pre-Transformer representation |
| 1 | Transformer Layer 1 |
| 2 | Transformer Layer 2 |
| … | … |
| 12 | Transformer Layer 12 |

For batched model output, every state has shape `[B, T, 768]`. Primary item-level pooling is the temporal mean:

`z_l = (1 / T) Σ_t H_(l,t)`

Thus each representation changes from `[B, T, 768]` to `[B, 768]`. Stacking the 13 independently pooled vectors gives `[B, 13, 768]`, or `[13, 768]` after removing the single-item batch dimension.

The layers will **not** be concatenated for the primary probe. Layer-wise probing requires a separate 768-dimensional design matrix for each representation so that decodability can be compared across depth without changing probe input dimensionality.

## 9. Real-Audio End-to-End Verification

A deterministic real-audio sanity check exercised four verified excerpts:

| Song ID | Test case | Resampled input shape | Hidden-state shape (each of 13) | Item output |
|---:|---|---:|---:|---:|
| 10 | 44.1 kHz stereo | `[1, 1081464]` | `[1, 3379, 768]` | `[13, 768]` |
| 1198 | 48 kHz stereo | `[1, 1081728]` | `[1, 3380, 768]` | `[13, 768]` |
| 811 | 44.1 kHz mono | `[1, 1080216]` | `[1, 3375, 768]` | `[13, 768]` |
| 1024 | 16 kHz stereo | `[1, 1080002]` | `[1, 3374, 768]` | `[13, 768]` |

All labels joined successfully by `song_id`. Mono handling left the original mono values unchanged. Resampling preserved duration to rounding precision. All four CUDA forwards succeeded in FP32, every sample returned 13 states with hidden size 768, and temporal length varied with actual decoded duration.

All 52 pooled vectors were finite, contained no NaN or infinity, and were not all-zero. Pairwise checks confirmed that different inputs did not produce identical `[13, 768]` matrices. These are implementation checks only; they do not rank layers or demonstrate emotion information.

Single-item approximately 45-second inference used roughly 1.578–1.580 GiB peak allocated VRAM and 2.262–2.656 GiB peak reserved VRAM in the real-audio runs. It therefore fit comfortably on the tested RTX 4060 Laptop GPU without mixed precision or chunking.

## 10. Final Protocol

| Component | Locked primary protocol |
|---|---|
| Dataset | DEAM |
| Primary subset | 1,744 approximately 45-second excerpts |
| Targets | Static `valence_mean` and `arousal_mean`, modeled separately |
| Sample unit | One audio item |
| Input duration | Actual decoded excerpt; no forced exact-45-second crop/pad |
| Channels | Arithmetic mean when multichannel; mono values unchanged |
| Model sampling rate | 24 kHz, resampled on the fly |
| Processor | Official MERT feature extractor; no extra project normalization |
| Model | `m-a-p/MERT-v1-95M`, frozen, evaluation mode, gradients disabled |
| Precision/hardware verified | FP32 on NVIDIA RTX 4060 Laptop GPU |
| Representations | Pre-Transformer + Transformer Layers 1–12 |
| Hidden-state shape | `[B, T, 768]` |
| Pooling | Temporal mean over valid `T` |
| Per-item output | `[13, 768]` |
| Primary context | Full approximately 45-second excerpt, single pass |
| Planned context ablation | 5-second chunk-and-aggregate |

## 11. Known Caveats

1. DEAM's metadata subsets differ in audio duration and annotation design; metadata filenames are provenance labels, not automatic independent experimental years or splits.
2. Source sample rates and channel counts are heterogeneous, so every file must be inspected and processed from its actual decoded properties.
3. Excluding the 58 full songs narrows the study population to the excerpt collection and removes the entire `metadata_2015` subset, not a random sample.
4. A 45-second forward pass is longer than MERT-v1-95M's documented 5-second pretraining context. Hardware feasibility does not guarantee invariance to context-length distribution shift.
5. Temporal mean pooling removes temporal ordering and can smooth brief events.
6. Stage 5 verifies implementation and numerical behavior, not Valence/Arousal decodability.
7. For ID 1198, container/header inspection and complete waveform decoding yielded different duration estimates. Full decoding produced 2,163,456 samples at 48 kHz (45.072 seconds), and the preprocessing pipeline treats the decoded sample count as authoritative. This isolated discrepancy is documented without treating it as evidence of dataset corruption.

## 12. What Module A Establishes

Module A establishes a reproducible DEAM input population and a verified path from real MP3 files to 13 fixed-dimensional frozen MERT representations. It fixes the software compatibility stack, audio preprocessing, representation naming, pooling operation, and primary duration protocol.

Module A does **not** establish:

- emotion-prediction performance;
- the best MERT layer for Valence or Arousal;
- a causal interpretation of any representation;
- superiority of MERT over traditional features or another model;
- success of any regression probe.

## 13. Next Module

Module B will address dataset-wide representation extraction, representation storage/cache design, supervised dataset construction, and preparation for layer-wise probing. None of those tasks is implemented in this closure step.
