# Module B — Stage 2: Batched Representation & Alignment Validation

Date: 2026-09-28

Status: **Passed after researcher-reviewed protocol refinement and iteration**

## 1. Stage objective

This Stage tested two small-scale implementation questions using eight real DEAM excerpts:

1. whether dynamically padded batches of four can produce masked-pooled MERT representations consistent with independently extracted single-item representations; and
2. whether every representation can be joined explicitly by sample ID to its static Valence, static Arousal, and locked train/validation/test split.

No full-dataset extraction, formal cache, probing, split creation, or research-protocol change was performed.

## 2. Files created or modified

- Modified `src/mert_emotion_probing/mert.py` with reusable dynamic-batch input preparation, model-derived feature-frame masking, and valid-frame masked temporal mean pooling.
- Created `src/mert_emotion_probing/splits.py` with the approved fixed split definition and deterministic generation method.
- Created `scripts/validate_batched_alignment.py`, a fixed eight-sample validation runner.
- Created `scripts/create_deam_primary_split.py`, which creates the approved split once and thereafter verifies rather than overwrites it.
- Created `scripts/validate_stage2_iteration.py`, which verifies split integrity, target distributions, the primary single-item extraction path, and ID-based alignment.
- Created the Git-trackable split artifact `data/metadata/deam_primary_split_seed42.csv`.
- Created this report: `docs/codex_reports/module_b_stage2_batched_alignment_validation.md`.
- Created the small Git-ignored record `data/raw/deam/verification/module_b_stage2_batched_alignment_validation.json`. It contains shapes, metadata, checks, and comparison summaries only; no embeddings were saved.
- Created the small Git-ignored iteration record `data/raw/deam/verification/module_b_stage2_iteration_validation.json`. It contains verification summaries only; no embeddings were saved.

The existing audio decoding, mono conversion, resampling, frozen-model loading, and hidden-state extraction functions were reused.

## Part I — Initial validation

## 3. Selected DEAM samples and why

The eight IDs were selected deterministically from the verified primary excerpt population. Selection covered decoded-length variation, mono and stereo, 16/22.05/44.1/48 kHz source rates, and both 2013 and 2014 metadata subsets. Metadata subset is provenance and was not treated as a train/validation/test split.

| ID | Selection reason | Decoded `[C, N]` | Source rate | Decoded duration | Resampled length | Valence | Arousal |
|---:|---|---:|---:|---:|---:|---:|---:|
| 272 | Shortest primary excerpt in the Module A inventory | `[2, 1,968,758]` | 44,100 | 44.643039 s | 1,071,433 | 5.9 | 6.4 |
| 1708 | Largest container-level duration in the Module A inventory; actual decoded length retained | `[1, 1,988,352]` | 44,100 | 45.087347 s | 1,082,097 | 4.2 | 3.6 |
| 10 | Ordinary stereo reference used previously | `[2, 1,987,190]` | 44,100 | 45.060998 s | 1,081,464 | 4.0 | 4.7 |
| 1198 | 48 kHz stereo and known container/decoded-duration discrepancy | `[2, 2,163,456]` | 48,000 | 45.072000 s | 1,081,728 | 3.5 | 1.8 |
| 811 | Mono reference used previously | `[1, 1,984,896]` | 44,100 | 45.008980 s | 1,080,216 | 4.1 | 4.9 |
| 1024 | Only verified 16 kHz primary excerpt | `[2, 720,001]` | 16,000 | 45.000063 s | 1,080,002 | 3.2 | 2.9 |
| 1640 | 22.05 kHz stereo resampling branch | `[2, 992,251]` | 22,050 | 45.000045 s | 1,080,002 | 6.6 | 6.1 |
| 1303 | 48 kHz stereo branch | `[2, 2,160,001]` | 48,000 | 45.000021 s | 1,080,001 | 6.1 | 6.5 |

Every preprocessing decision used the actually decoded waveform rather than a container-level duration estimate.

## 4. Batch construction and padding

The runner used batch size 4 and padded only to the longest resampled waveform inside each batch. The official MERT feature extractor normalized each waveform and returned `input_values` plus an input attention mask. Mask value 1 denotes a real waveform sample and 0 denotes batch-local right padding.

| Batch | IDs in explicit row order | Padded input | Input mask | Valid input positions by row |
|---:|---|---:|---:|---|
| 0 | 272, 1708, 10, 1198 | `[4, 1,082,097]` | `[4, 1,082,097]` | 1,071,433; 1,082,097; 1,081,464; 1,081,728 |
| 1 | 811, 1024, 1640, 1303 | `[4, 1,080,216]` | `[4, 1,080,216]` | 1,080,216; 1,080,002; 1,080,002; 1,080,001 |

No waveform was permanently padded to 45 seconds, and no sample was cropped.

## 5. Valid-frame mask and pooling implementation

The input attention mask was converted to a feature-frame mask using the loaded model's existing `_get_feature_vector_attention_mask(feature_sequence_length, input_attention_mask)` method. This uses the model's own convolutional output-length calculation rather than an independently invented approximation.

For every hidden state `H` and binary feature mask `M`, pooling used:

```text
sum_t(H[b,t,:] * M[b,t]) / sum_t(M[b,t])
```

This maps `[B, T_max, 768]` to `[B, 768]` per representation. Stacking Pre-Transformer plus Transformer Layers 1–12 produces `[B, 13, 768]`. Padding-derived feature frames were excluded from the mean.

An important limitation was exposed: masking invalid output frames does not guarantee that earlier valid features are identical to single-item features when raw-waveform padding changes convolutional normalization. The loaded checkpoint has `feat_extract_norm="group"`; Transformers 4.24.0 implements its first HuBERT feature-extractor layer with `GroupNorm` after convolution, before the Transformer attention mask is applied.

## 6. Observed tensor shapes

| Batch | Each of 13 hidden states | Feature mask | Valid feature frames by row | Pooled batch |
|---:|---:|---:|---|---:|
| 0 | `[4, 3381, 768]` | `[4, 3381]` | 3347; 3381; 3379; 3380 | `[4, 13, 768]` |
| 1 | `[4, 3375, 768]` | `[4, 3375]` | 3375; 3374; 3374; 3374 | `[4, 13, 768]` |

Every independently extracted sample and every batch row produced an item matrix `[13, 768]`.

## 7. Single-item vs batched representation comparison

The tolerance was fixed before comparison at `rtol=1e-4, atol=1e-5`; it was not changed after observing the results.

| ID | Padded input samples | Padded feature frames | Maximum absolute difference | Mean absolute difference | Allclose |
|---:|---:|---:|---:|---:|:---:|
| 272 | 10,664 | 34 | 0.0164652765 | 0.0004675511 | No |
| 1708 | 0 | 0 | 0.0001042187 | 0.0000080394 | No |
| 10 | 633 | 2 | 0.0003828406 | 0.0000211260 | No |
| 1198 | 369 | 1 | 0.0002559945 | 0.0000174483 | No |
| 811 | 0 | 0 | 0.0001433492 | 0.0000080887 | No |
| 1024 | 214 | 1 | 0.0002671480 | 0.0000152427 | No |
| 1640 | 214 | 1 | 0.0003900193 | 0.0000140337 | No |
| 1303 | 215 | 1 | 0.0002074838 | 0.0000121478 | No |

All eight comparisons failed the predeclared elementwise tolerance. Small nonzero differences for unpadded longest rows are consistent with normal batched-versus-single FP32 kernel/reduction differences, but ID 272 showed a clearly larger difference and also had by far the most padding.

The most likely implementation-level cause is padding sensitivity in the checkpoint's group-normalized convolutional feature extractor: zero padding exists before the model derives and applies the feature-frame attention mask, and GroupNorm includes the time dimension in its statistics. The output mask correctly prevents padded frames from entering the final temporal mean, but it cannot undo changes already introduced into valid convolutional features. No tolerance was loosened and no preprocessing or model behavior was changed to conceal this result.

## 8. Sample ID / Valence / Arousal / split alignment verification

Representations were stored in the validation process under explicit integer sample-ID keys. The alignment view was then constructed by looking up each selected ID in the verified Module A mapping table, not by assuming matching row positions.

| Sample ID | Representation | Valence | Arousal | Split |
|---:|---:|---:|---:|---|
| 272 | `[13, 768]` | 5.9 | 6.4 | Unresolved |
| 1708 | `[13, 768]` | 4.2 | 3.6 | Unresolved |
| 10 | `[13, 768]` | 4.0 | 4.7 | Unresolved |
| 1198 | `[13, 768]` | 3.5 | 1.8 | Unresolved |
| 811 | `[13, 768]` | 4.1 | 4.9 | Unresolved |
| 1024 | `[13, 768]` | 3.2 | 2.9 | Unresolved |
| 1640 | `[13, 768]` | 6.6 | 6.1 | Unresolved |
| 1303 | `[13, 768]` | 6.1 | 6.5 | Unresolved |

Checks passed for unique selected IDs, one representation per ID, complete Valence, complete Arousal, and no unexpected representation ID. The split check did not pass: all eight split values are missing because no locked split can be recovered from the repository.

Repository evidence is explicit:

- Module A Stage 4 states, “No DEAM split was created.”
- Module A Stage 5 states that data splits had not begun.
- `configs/` contains no split configuration.
- Verified Module A CSV/JSON artifacts contain no train/validation/test assignment.
- Repository filename/content search and Git-history search found no split artifact or definition.

No new split was created, and metadata subset labels (`2013`/`2014`) were not misused as experimental splits.

## 9. Model-state and numerical checks

- Model: `m-a-p/MERT-v1-95M`, cached snapshot `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`
- Transformers: 4.24.0
- PyTorch: 2.14.0+cu130
- Device: `cuda:0`, NVIDIA GeForce RTX 4060 Laptop GPU
- Dtype: `torch.float32`
- Evaluation mode: yes
- All model parameters frozen: yes
- Inference mode: enforced by the reused `extract_hidden_states` function
- Returned pooled tensors requiring gradients: none
- NaN or Inf: none
- Unexpectedly all-zero pooled outputs: none
- Single-item shape: `[13, 768]` for every ID
- Batch shape: `[4, 13, 768]` for both batches

## 10. Warnings / unexpected findings

1. **No recoverable locked split.** This blocks the required split alignment and prevents Stage 2 from passing.
2. **Single-vs-batch mismatch.** All eight rows failed the predeclared allclose tolerance; maximum absolute difference was `0.0164652765` for the most heavily padded sample.
3. **Likely padding-sensitive GroupNorm.** The checkpoint's raw-audio convolutional frontend uses group normalization before attention masking, so valid features can change when raw waveforms are padded.
4. **Hardware pressure.** Batch size 4 at approximately 45 seconds in FP32 ran near the tested 8 GiB GPU's memory limit (approximately 7.8–7.9 GiB observed during the run) and was much slower than single-item inference. This is an operational observation, not a batch-size optimization decision.
5. **Wrapper encoding failure after output creation.** The Python child completed and wrote the full JSON, but `conda run` failed while re-printing captured stdout because the Windows GBK console could not encode a replacement character from the local environment. The saved JSON parsed successfully and passed structural checks.

## 11. Protocol consistency

The run retained actual decoded durations, arithmetic-mean mono conversion, 24 kHz resampling, the frozen FP32 MERT-v1-95M checkpoint, evaluation/inference mode, disabled gradients, 13 representation levels, hidden dimension 768, and valid-frame temporal mean pooling. It did not perform full extraction, caching, probing, split creation, label redefinition, or any research-protocol change.

The Stage cannot satisfy the locked-split requirement because Module A did not define a split. The batched path also did not meet the required single-item consistency criterion. Both conflicts are reported rather than silently repaired.

## 12. Exact commands used

Repository and split-source audit:

```powershell
rg -n -i --hidden -g '!**/.git/**' -g '!data/raw/**' "\b(train|training|validation|valid|test|split)\b" README.md docs src scripts configs environment.yml
rg --files -uu -g '!**/.git/**' | rg -i "split|train|valid|test"
git log --all --oneline --decorate --grep='split' -i
```

Syntax and diff checks:

```powershell
& 'C:\Users\Rinshinozaki\miniconda3\Scripts\conda.exe' run -n mert-emotion python -m py_compile src\mert_emotion_probing\mert.py scripts\validate_batched_alignment.py
git diff --check
```

Validation run:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_MODULES_CACHE=Join-Path $env:TEMP 'mert-b2-hf-modules'
$snapshot='C:\Users\Rinshinozaki\.cache\huggingface\hub\models--m-a-p--MERT-v1-95M\snapshots\12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
& 'C:\Users\Rinshinozaki\miniconda3\Scripts\conda.exe' run -n mert-emotion python scripts\validate_batched_alignment.py --model-source $snapshot --output-json data\raw\deam\verification\module_b_stage2_batched_alignment_validation.json | Out-Null
```

GroupNorm implementation inspection:

```powershell
rg -n -C 8 "class HubertGroupNormConvLayer|GroupNorm" C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\Lib\site-packages\transformers\models\hubert\modeling_hubert.py
```

## Initial validation conclusion

**Stage 2 did not pass.** Dynamic padding, model-derived feature-frame masks, valid-frame pooling, explicit sample-ID representation mapping, label lookup, expected shapes, and numerical/model-state checks were implemented and exercised successfully. However:

- the existing locked train/validation/test split cannot be recovered because Module A did not create one; and
- batched representations did not agree with single-item representations at the predeclared FP32 tolerance, with a maximum absolute difference of `0.0164652765`.

No workaround was introduced. Researcher review is required before deciding how and when the split should be defined and whether a batching strategy that changes frontend normalization behavior is scientifically acceptable. No Stage 3 work should begin.

## Part II — Researcher-reviewed protocol refinement

After reviewing the initial evidence, the researcher approved two explicit decisions. These are refinements based on the documented failure, not retroactive claims that the first design succeeded.

### Decision B2.1 — Primary representation extraction

Primary MERT representation extraction will use **single-item inference (`batch size = 1`)**.

The reason is precise: padded batched inference was empirically found not to be representation-equivalent to independent single-item inference under the tested extraction protocol. Primary extraction therefore prioritizes representation consistency, research interpretability, and reproducibility over throughput.

This does not mean that MERT cannot perform batched inference. It means that raw-waveform padding combined with this checkpoint's frontend normalization changed representations enough to fail the predeclared equivalence criterion. The checkpoint, preprocessing, decoded-duration policy, representation levels, pooling operation, and hidden dimension remain unchanged.

The locked primary path is now:

```text
one actual decoded DEAM excerpt
→ arithmetic-mean mono conversion when multichannel
→ resample to 24 kHz
→ frozen FP32 MERT-v1-95M in evaluation/inference mode
→ Pre-Transformer plus Transformer Layers 1–12
→ temporal mean over all valid frames
→ [13, 768]
```

### Decision B2.2 — Fixed dataset split

The researcher approved a single sample-level **70% train / 15% validation / 15% test** split for the 1,744 primary excerpts using random seed 42. Sample ID is the authoritative identity. No target binning, target stratification, genre stratification, or metadata-subset substitution is used.

Integer counts use a deterministic rule: round the train and validation targets half-up, then assign the exact remaining samples to test. For 1,744 samples this gives:

| Split | Target ratio | Exact count |
|---|---:|---:|
| Train | 70% | 1,221 |
| Validation | 15% | 262 |
| Test | 15% | 261 |
| **Total** | **100%** | **1,744** |

The generation method sorts all verified primary sample IDs, applies one `numpy.random.default_rng(42).permutation`, assigns contiguous split segments using these counts, and sorts the saved artifact back by `sample_id` for readability. The artifact is:

`data/metadata/deam_primary_split_seed42.csv`

It contains only `sample_id,split`, is Git-trackable, and must be reused by later probing and layer-comparison experiments. The creation script refuses to overwrite an existing artifact if it differs from the deterministic approved result.

## Part III — Iteration validation

### Split integrity

All required integrity checks passed:

- total rows: 1,744;
- unique sample IDs: 1,744;
- duplicate IDs: none;
- allowed values only: `train`, `validation`, `test`;
- missing primary IDs: none;
- extra IDs: none;
- splits mutually exclusive: yes;
- union equals the complete verified primary population: yes;
- every sample assigned exactly once: yes;
- exact counts: 1,221 / 262 / 261;
- in-memory regeneration with seed 42 reproduced the saved artifact exactly.

Running the split creation command a second time returned `verified_existing`; it did not generate a new assignment or overwrite the artifact.

### Valence / Arousal distribution sanity check

No stratification was performed, and seed 42 was used once as approved. The observed descriptive statistics were:

| Split | Target | Count | Mean | SD | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| Train | Valence | 1,221 | 4.8947 | 1.1750 | 1.6 | 7.9 |
| Train | Arousal | 1,221 | 4.8170 | 1.2958 | 1.6 | 8.1 |
| Validation | Valence | 262 | 4.8869 | 1.1492 | 1.9 | 8.4 |
| Validation | Arousal | 262 | 4.8118 | 1.2151 | 2.2 | 7.8 |
| Test | Valence | 261 | 4.9613 | 1.1960 | 1.9 | 7.5 |
| Test | Arousal | 261 | 4.7931 | 1.3355 | 1.9 | 7.5 |

The means and standard deviations are descriptively similar across splits, and no obvious distribution anomaly requiring researcher review was observed. Differences in extrema are plausible for an unstratified random split and were not used to alter the seed or regenerate the split.

### Primary single-item extraction validation

Three real samples were selected to cover all three split labels while retaining useful audio branches: ID 10 (`train`, 44.1 kHz stereo), ID 811 (`validation`, 44.1 kHz mono), and ID 1640 (`test`, 22.05 kHz stereo).

| ID | Split | Model input | Each hidden state | Valid frames | Pooled batch | Item output |
|---:|---|---:|---:|---:|---:|---:|
| 10 | Train | `[1, 1081464]` | `[1, 3379, 768]` | 3379 | `[1, 13, 768]` | `[13, 768]` |
| 811 | Validation | `[1, 1080216]` | `[1, 3375, 768]` | 3375 | `[1, 13, 768]` | `[13, 768]` |
| 1640 | Test | `[1, 1080002]` | `[1, 3374, 768]` | 3374 | `[1, 13, 768]` | `[13, 768]` |

Every input attention mask contained only valid positions; no cross-sample raw-waveform padding occurred. The model remained frozen, in evaluation mode, FP32 on `cuda:0`, and hidden-state extraction ran under inference mode. All outputs were finite, contained no NaN or Inf, were not all zero, and did not require gradients.

### Final ID / representation / label / split alignment

Representations were stored and retrieved under explicit sample-ID keys. Labels were looked up from the verified Module A mapping by `song_id`, and split membership was independently looked up from the fixed artifact by `sample_id`. No row-order correspondence was assumed.

| Sample ID | Representation Shape | Valence | Arousal | Split |
|---:|---:|---:|---:|---|
| 10 | `[13, 768]` | 4.0 | 4.7 | Train |
| 811 | `[13, 768]` | 4.1 | 4.9 | Validation |
| 1640 | `[13, 768]` | 6.6 | 6.1 | Test |

All alignment checks passed: selected IDs were unique, representation IDs matched exactly, and no Valence, Arousal, or split value was missing.

### Iteration commands

Syntax check:

```powershell
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' -m py_compile src\mert_emotion_probing\splits.py scripts\create_deam_primary_split.py scripts\validate_stage2_iteration.py
```

Split creation and deterministic second-run verification:

```powershell
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts\create_deam_primary_split.py
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts\create_deam_primary_split.py
```

Single-item extraction and alignment validation:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_MODULES_CACHE=Join-Path $env:TEMP 'mert-b2-iteration-hf-modules'
$snapshot='C:\Users\Rinshinozaki\.cache\huggingface\hub\models--m-a-p--MERT-v1-95M\snapshots\12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts\validate_stage2_iteration.py --model-source $snapshot --output-json data\raw\deam\verification\module_b_stage2_iteration_validation.json
```

## Final Stage 2 conclusion

**Stage 2 passed after iteration and is ready for researcher review and freeze.**

Stage 2 did not demonstrate that padded batching works identically. Instead, it identified that padded batching is not representation-equivalent for this MERT extraction setup, preserved and explained that negative result, refined the primary extraction protocol to single-item inference, established one reproducible sample-level dataset split, and verified complete sample identity/label/split alignment.

The final locked outcomes are:

- primary MERT extraction uses single-item inference (`batch size = 1`);
- the fixed split artifact is `data/metadata/deam_primary_split_seed42.csv`;
- exact split counts are Train 1,221, Validation 262, Test 261;
- all future probing and layer comparisons must reuse this same split;
- no full representation extraction or probing was performed in Stage 2.
