# Module B — Representation Extraction and Dataset Construction

## 1. Module Objective

Module B turned the representation protocol fixed in Module A into a validated dataset-wide implementation. Its goals were to verify the meaning and shape of the returned MERT states, determine a representation-consistent extraction path, establish a reproducible sample split and identity mapping, extract all primary samples, and create a probing-ready representation dataset.

Module B did **not** test whether musical emotion is decodable from MERT. It prepared and verified the inputs required for that later experiment.

## 2. Inherited Locked Protocol

Module B inherited the following decisions from Module A:

- Dataset: the 1,744 primary DEAM approximately 45-second excerpts; the 58 full-length songs are excluded.
- Sample unit: one audio item is one sample.
- Duration: preserve the actually decoded excerpt; do not force an exact 45-second crop or pad.
- Channels: arithmetic mean for multichannel-to-mono conversion; mono values remain unchanged.
- Sampling rate: resample on the fly to 24 kHz.
- Model: `m-a-p/MERT-v1-95M`, frozen, FP32, evaluation mode, with gradients disabled.
- Representations: Pre-Transformer plus Transformer Layers 1–12.
- Hidden dimension: 768.
- Pooling: temporal mean over valid time positions.
- Expected per-sample output: `[13, 768]`.

## 3. Stage B1 — Representation Inspection

Stage B1 inspected three real DEAM excerpts covering ordinary stereo, mono, and 16 kHz resampling branches. Each item returned 13 hidden states, with observed per-item shapes `[T, 768]`; `T` varied with the actual decoded duration. Temporal mean pooling mapped each level from `[T, 768]` to `[768]`, and stacking all levels produced `[13, 768]`.

The layer mapping was verified rather than inferred from tensor count alone. The loaded encoder implementation appends its initial encoder state before entering the 12-layer Transformer loop. Forward hooks then showed exact equality between `hidden_states[0]` and the input to Transformer Layer 1, and between `hidden_states[i]` and the output of Transformer Layer `i` for `i = 1..12`. Therefore:

- `hidden_states[0]` is the Pre-Transformer representation after positional convolution, layer normalization, and dropout, immediately before Transformer Layer 1;
- `hidden_states[1]` through `hidden_states[12]` are Transformer Layers 1 through 12.

The model was frozen, in evaluation/inference mode, and FP32 on CUDA. Hidden states and pooled outputs were finite, contained no NaN or Inf, and were not all zero. For these unpadded single-item runs, explicit valid-frame masked means agreed with ordinary temporal means within `rtol=1e-6, atol=1e-7`; the largest reduction-order difference was `2.384185791015625e-07`.

## 4. Stage B2 — Batched Validation and Unexpected Result

### Initial design

The initial Stage B2 design tested whether several variable-length excerpts could be processed together using dynamic raw-waveform padding, the feature extractor's input attention mask, the model's own feature-frame mask conversion, and valid-frame masked temporal mean pooling. The intended result was that each padded-batch item would be numerically equivalent to its independent single-item representation.

Eight deliberately selected real excerpts were processed in two batches of four. The comparison tolerance was declared before observing the results: `rtol=1e-4, atol=1e-5`.

### Failed equivalence test

All 8 single-item versus padded-batch comparisons failed the predeclared elementwise tolerance. The largest maximum absolute difference was `0.0164652765` for ID 272, the most heavily padded item. ID 272 received 10,664 padded waveform samples, corresponding to 34 padded feature frames. Its mean absolute difference was `0.0004675511`, also the largest of the eight comparisons.

The valid-frame mask and masked pooling were implemented correctly: padding-derived output frames were excluded from the final mean. Investigation nevertheless showed that MERT-v1-95M uses GroupNorm in its convolutional feature extractor. Raw-waveform padding enters this frontend before the later feature-frame attention mask is applied, and GroupNorm includes the temporal dimension in its normalization statistics. Masked pooling can exclude invalid output frames, but it cannot undo padding-induced changes that have already affected valid convolutional features.

The correct conclusion is not that MERT cannot use batching. Rather:

> Padded multi-item raw-waveform inference was empirically found not to be representation-equivalent to independent single-item inference under this extraction protocol.

Batch size 4 also used approximately 7.8–7.9 GiB on the tested 8 GiB GPU and was operationally slower than single-item inference. This was a secondary engineering observation, not the primary research reason for the protocol refinement.

### Researcher-reviewed refinement

After review of the failed validation, the primary extraction protocol was formally refined to **single-item inference (`batch size = 1`)**. This choice prioritizes representation consistency, interpretability, and reproducibility. The checkpoint, preprocessing, decoded-duration policy, representation levels, pooling rule, and hidden dimension were not changed.

The failed comparison and its explanation remain part of the research record; Stage B2 passed only after this explicit refinement and subsequent validation.

## 5. Dataset Split

A fixed sample-level 70%/15%/15% split was created once for the complete 1,744-sample primary population using seed 42:

| Split | Count |
|---|---:|
| Train | 1,221 |
| Validation | 262 |
| Test | 261 |
| **Total** | **1,744** |

The artifact is `data/metadata/deam_primary_split_seed42.csv`. It contains only `sample_id,split` and must be reused by downstream probing and layer comparisons. The implementation sorts the verified IDs, applies one `numpy.random.default_rng(42).permutation`, assigns deterministic split counts, and sorts the saved CSV by sample ID for readability. Seed 42 has no special scientific meaning; it fixes one reproducible random assignment.

No Valence/Arousal binning, target stratification, genre stratification, or metadata-subset substitution was used. Integrity checks confirmed 1,744 unique IDs, mutually exclusive assignments, complete population coverage, no missing or extra IDs, and exact deterministic regeneration. Descriptive Valence and Arousal means and standard deviations were similar across the three splits, with no obvious anomaly that justified altering the approved seed.

## 6. Identity and Alignment

**Sample ID is the authoritative identity key.** A row position describes where an item happens to appear in a particular table or tensor; it is not the item's identity and may change after sorting, filtering, or serialization.

Stage B2 therefore verified explicit ID-based correspondence:

```text
Representation
↔ Sample ID
↔ Valence
↔ Arousal
↔ Train / Validation / Test split
```

No alignment step assumed that representation row `i` automatically matched metadata row `i`. Validation samples covering all three splits had complete representations, targets, and split assignments with no duplicate or mismatched ID.

## 7. Stage B3 — Full Extraction

Stage B3 extracted all 1,744 primary excerpts using frozen single-item MERT inference. All 1,744 samples succeeded and no dummy, zero, or replacement representation was used.

The canonical cache is:

`outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt`

It contains:

- `representations`: FP32 tensor `[1744, 13, 768]`;
- `sample_ids`: int64 tensor `[1744]`, ordered by ascending ID and explicitly aligned with the first tensor dimension;
- `metadata`: compact dataset, model, snapshot, preprocessing, representation, split, version, ordering, and creation information.

The cache size is 69,666,277 bytes (approximately 66.44 MiB). The extraction loop took 739.648 seconds, or approximately 0.4241 seconds per sample; the complete invocation took 746.121 seconds on the tested RTX 4060 Laptop GPU. Runtime is an engineering observation, not a research outcome.

Extraction was resume-safe: each successful sample was atomically saved as an ID-bearing part and validated before reuse, and the final cache was constructed only after all parts were present and valid. The successful run had no interruption, so real interruption recovery was not exercised; only the mechanism and its structural validation are claimed.

## 8. Cache Integrity and Reproducibility

Pre- and post-serialization checks confirmed:

- exact representation shape `[1744, 13, 768]` and sample-ID shape `[1744]`;
- 1,744 unique IDs in deterministic ascending order;
- no missing, extra, or duplicate primary IDs;
- FP32 representations and int64 IDs;
- no NaN or Inf values;
- no all-zero sample representation;
- frozen/evaluation/inference model state and no cross-sample waveform padding.

Fresh independent single-item extraction was repeated for ID 10 (train), ID 811 (validation), and ID 1640 (test). Each fresh `[13, 768]` matrix matched the cached matrix with maximum absolute difference `0.0` under the predeclared `rtol=1e-6, atol=1e-7`. Explicit ID lookup also recovered the expected Valence, Arousal, and split for all three samples.

The canonical cache and intermediate parts are derived artifacts. They remain local and Git-ignored rather than being uploaded to GitHub. Their construction is reproducible from the tracked source, fixed split, documented protocol, and local DEAM/model inputs.

## 9. Final Module B Output

Module B produces the probing-ready representation tensor:

```text
X = [1744, 13, 768]
```

For every first-dimension row, `sample_ids[i]` provides the authoritative key for retrieving the corresponding static Valence, static Arousal, and fixed train/validation/test assignment. Downstream work can select one 768-dimensional representation level at a time without rerunning MERT.

## 10. What Module B Establishes

Module B establishes that:

- the frozen MERT extraction implementation matches the locked representation protocol;
- the Pre-Transformer and Transformer-layer mapping is verified;
- each real excerpt produces a valid `[13, 768]` item representation;
- padded multi-item extraction is not representation-equivalent to independent extraction under the tested protocol, motivating the reviewed single-item path;
- the dataset split is fixed, reproducible, and ID-addressable;
- sample identity, targets, split, and representations remain traceable;
- the full canonical cache is complete, finite, reproducible in the tested environment, and ready for downstream probing;
- downstream probing does not need to rerun MERT for every experiment.

## 11. What Module B Does Not Establish

Module B does **not** establish that:

- MERT understands musical emotion in a human-like sense;
- Valence or Arousal is decodable from these representations;
- one MERT layer is more predictive than another;
- MERT representations outperform traditional audio features;
- Valence or Arousal prediction generalizes well to held-out data;
- a future probe is free from all possible confounds.

These require later controlled probing and comparison. Successful representation extraction is not evidence of emotion-prediction performance.

## 12. Final Frozen Decisions

All downstream Modules must inherit the following unless a concrete problem is documented and the researcher explicitly approves a revision:

- primary extraction uses independent single-item MERT inference with `batch size = 1`;
- no cross-sample raw-waveform padding is used for the canonical representations;
- one sample representation is `[13, 768]`: Pre-Transformer plus Transformer Layers 1–12;
- representations are FP32 temporal means over valid feature frames;
- the complete canonical tensor is `[1744, 13, 768]` with an explicit `[1744]` sample-ID vector;
- Sample ID is the authoritative identity key;
- the fixed split is `data/metadata/deam_primary_split_seed42.csv`;
- split counts are 1,221 train, 262 validation, and 261 test, generated once with seed 42;
- targets and split membership are joined explicitly by Sample ID rather than row order;
- the canonical cache is `outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt`;
- the canonical cache and intermediate representation artifacts remain derived, local, and excluded from Git.

With these decisions and artifacts frozen, Module B is complete.
