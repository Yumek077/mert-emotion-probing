# Module B, Stage 3 — Full Representation Extraction and Caching

## 1. Stage objective

Extract frozen MERT representations for all 1,744 primary DEAM excerpts with the locked single-item protocol, consolidate them into a resume-safe canonical cache, and verify numerical integrity, identity, serialization, fresh reproducibility, and label/split traceability. No probing or prediction was performed.

## 2. Locked protocol used

- Dataset: the 1,744 primary DEAM approximately 45-second excerpts; the 58 full-length songs were excluded.
- Sample definition: one decoded audio item is one sample; the actually decoded duration was preserved.
- Audio: arithmetic channel mean for multichannel-to-mono conversion, followed by on-the-fly resampling to 24 kHz.
- Model: `m-a-p/MERT-v1-95M`, snapshot `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`, frozen, evaluation mode, inference mode, FP32.
- Extraction: one sample per forward pass (`batch_size = 1`), with no cross-sample waveform padding.
- Representation: pre-Transformer plus Transformer Layers 1–12; 768 hidden dimensions; temporal mean over valid feature frames; `[13, 768]` per sample.
- Split: `data/metadata/deam_primary_split_seed42.csv` (1,221 train, 262 validation, 261 test). The split was read and verified, not regenerated.

## 3. Files created or modified

Created:

- `scripts/extract_deam_mert_representations.py`
- `docs/codex_reports/module_b_stage3_full_representation_extraction.md`

Generated but Git-ignored:

- `outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt`
- `outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.manifest.json`
- `outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.parts/`

No pre-existing research protocol, split, label definition, preprocessing rule, or personal note was modified in this Stage.

## 4. Preflight checks

All preflight checks passed before full extraction:

- 1,744 unique primary sample IDs recovered.
- 1,744 corresponding audio files indexed explicitly by sample ID.
- Valence and Arousal were complete for the primary population.
- Fixed split contained exactly 1,744 IDs with the locked counts.
- The local MERT snapshot loaded successfully.
- Model state was FP32, frozen, and evaluation mode.
- Feature extractor sampling rate was 24,000 Hz.
- CUDA was available on an NVIDIA GeForce RTX 4060 Laptop GPU.
- Output directory was writable.
- Available disk space was 173,877,878,784 bytes; the final representation tensor was expected to require 69,648,384 bytes.
- The canonical cache, manifest, per-sample parts, and `personal_notes/` were confirmed Git-ignored.

## 5. Full extraction implementation

The script deterministically indexed primary audio by sample ID and processed IDs in ascending order. Each item followed the locked path:

`decoded audio -> arithmetic channel mean -> 24 kHz resampling -> single-item MERT inference -> 13 hidden states -> valid-frame masked temporal mean -> [13, 768]`

For a single unpadded item, the input attention mask and model-derived feature-frame mask were required to contain only valid positions. Every representation was moved to CPU as contiguous FP32 and checked for the expected shape, finite values, and nonzero content before checkpointing. Any sample failure would stop final-cache construction rather than insert a dummy value or silently omit the sample.

## 6. Resume/checkpoint mechanism

Each successful sample was atomically saved as a small part containing its explicit `sample_id` and `[13, 768]` representation. A JSON state file recorded completion, elapsed extraction time, and failures. On restart, the script validates existing parts and skips valid completed IDs. Unexpected, corrupt, mismatched, or out-of-population parts cause an error. The canonical cache is built only when all 1,744 validated parts exist.

This run completed without interruption, so resume was not exercised operationally: `resumed = false`, and all 1,744 parts were newly extracted in one invocation.

## 7. Exact sample count

- Successfully extracted: 1,744 / 1,744
- Failed samples: 0
- Missing primary IDs: 0
- Extra IDs: 0
- Duplicate IDs: 0

## 8. Extraction runtime and operational observations

- Extraction loop: 739.648 seconds (12 minutes 19.648 seconds)
- Mean extraction-loop time: 0.4241 seconds per sample
- Total full-script invocation, including preflight, model loading, consolidation, serialization checks, and fresh verification: 746.121 seconds (12 minutes 26.121 seconds)
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Interruption/resume: none

Runtime is an engineering observation, not a research result.

## 9. Canonical cache structure

The canonical PyTorch cache is:

`outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt`

It contains:

- `representations`: FP32 tensor ordered by ascending sample ID
- `sample_ids`: int64 tensor providing explicit identity for the first dimension
- `metadata`: compact protocol, model, snapshot, split, dependency, ordering, and creation information

The cache is self-describing and does not duplicate Valence, Arousal, or split values. Those remain recoverable through the authoritative sample ID.

## 10. Final tensor shapes

- `representations.shape`: `[1744, 13, 768]`
- `sample_ids.shape`: `[1744]`
- Per-sample representation: `[13, 768]`
- Representation dtype: `torch.float32`
- Sample-ID dtype: `torch.int64`

## 11. Identity integrity checks

All identity checks passed both before serialization and after loading the saved cache:

- IDs were unique and in documented ascending order.
- Cached ID set exactly equalled the verified 1,744-primary-excerpt population.
- No primary ID was missing.
- No extra ID was present.
- `representations[i]` is explicitly associated with `sample_ids[i]`; no filesystem or metadata row ordering is assumed.

## 12. Numerical integrity checks

All numerical checks passed before and after serialization:

- FP32 dtype: passed
- NaN: none
- Inf: none
- Unexpected all-zero sample representations: none
- Expected tensor shape: passed

The cached metadata also records frozen parameters, evaluation mode, inference mode, single-item extraction, no cross-sample padding, 24 kHz, 13 levels, 768 hidden dimensions, and valid-frame temporal mean pooling.

## 13. Fresh re-extraction verification

The tolerance was fixed before comparison at `rtol = 1e-6`, `atol = 1e-7`.

| Sample ID | Split | Cache shape | Fresh shape | Maximum absolute difference | Allclose |
|---:|---|---|---|---:|---|
| 10 | train | `[13, 768]` | `[13, 768]` | 0.0 | Passed |
| 811 | validation | `[13, 768]` | `[13, 768]` | 0.0 | Passed |
| 1640 | test | `[13, 768]` | `[13, 768]` | 0.0 | Passed |

All three fresh independent single-item extractions matched their cached representations exactly in this environment.

## 14. Label/split traceability verification

Lookup was performed explicitly through sample ID, not through common row positions.

| Sample ID | Cache index | Representation shape | Valence | Arousal | Split |
|---:|---:|---|---:|---:|---|
| 10 | 6 | `[13, 768]` | 4.0 | 4.7 | train |
| 811 | 624 | `[13, 768]` | 4.1 | 4.9 | validation |
| 1640 | 1383 | `[13, 768]` | 6.6 | 6.1 | test |

The three-way train/validation/test traceability check passed.

## 15. Cache size and Git policy

- Canonical cache: 69,666,277 bytes (approximately 66.44 MiB)
- Intermediate parts and state: 72,457,264 bytes (approximately 69.10 MiB)

The cache, manifest, and intermediate parts match the existing `outputs/embeddings/*` ignore rule and are not Git-trackable. The raw dataset, model cache, and `personal_notes/` also remain excluded. Only extraction code, lightweight split/protocol artifacts, and reports are intended for Git.

## 16. Warnings and unexpected findings

No sample-level, numerical, identity, serialization, or traceability failure occurred. The Transformers loader emitted a recommendation to pass an explicit custom-code revision even though this run loaded an immutable local snapshot path. PyTorch also emitted the upstream deprecation warning for `torch.nn.utils.weight_norm`. Neither warning changed the extracted representations or Stage outcome.

The resume mechanism was validated structurally by strict part loading and skip logic, but no real interruption occurred during this successful run.

## 17. Exact commands used

Static and Git-ignore checks:

```powershell
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' -m py_compile 'scripts\extract_deam_mert_representations.py'
git check-ignore -v 'outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt'
git check-ignore -v 'outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.parts/10.pt'
git check-ignore -v 'personal_notes/module_b_what_i_should_understand.md'
git diff --check
```

Preflight only:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_MODULES_CACHE=(Join-Path $env:TEMP 'mert-b3-hf-modules')
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' 'scripts\extract_deam_mert_representations.py' --model-source 'C:\Users\Rinshinozaki\.cache\huggingface\hub\models--m-a-p--MERT-v1-95M\snapshots\12af15fef9d0ac838c3f475bfbbf26d2060dd4f5' --preflight-only
```

Full extraction and verification:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_MODULES_CACHE=(Join-Path $env:TEMP 'mert-b3-hf-modules')
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' 'scripts\extract_deam_mert_representations.py' --model-source 'C:\Users\Rinshinozaki\.cache\huggingface\hub\models--m-a-p--MERT-v1-95M\snapshots\12af15fef9d0ac838c3f475bfbbf26d2060dd4f5' --progress-every 25
```

Independent cache and repository review:

```powershell
git check-ignore -v 'outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt'
git check-ignore -v 'outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.manifest.json'
git check-ignore -v 'outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.parts/10.pt'
git check-ignore -v 'personal_notes/module_b_what_i_should_understand.md'
git status --short
git diff
git diff --check
```

## 18. Protocol consistency

The implementation matched every locked Stage 3 decision. It did not change the DEAM population, sample definition, decoded-duration policy, mono conversion, target sample rate, checkpoint, frozen/evaluation/inference state, representation levels, hidden dimension, pooling, target definitions, or fixed split. It did not use padded multi-item extraction and did not perform probing, layer comparison, prediction, metric evaluation, baseline construction, or confound analysis.

## 19. Final conclusion

**Stage 3 passed.** All 1,744 primary DEAM excerpts were successfully extracted with the frozen single-item MERT protocol and consolidated into a complete `[1744, 13, 768]` canonical FP32 cache. Identity, numerical, serialization, fresh re-extraction, and label/split traceability checks all passed. No failed sample, interruption, or protocol change occurred. Stage 4 was not started.
