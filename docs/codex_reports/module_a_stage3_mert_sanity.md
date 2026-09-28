# Module A — Stage 3: MERT Compatibility & Hardware Sanity Test

Date: 2026-09-27

Status: temporary Codex report; not a formal Module A research log

Model: `m-a-p/MERT-v1-95M`

## Purpose

This sanity test had three limited goals:

1. determine whether the frozen MERT-v1-95M checkpoint can be loaded and run correctly in the local environment;
2. verify the number and shapes of the returned hidden states and verify temporal mean pooling;
3. measure single-item FP32 inference memory for synthetic 5-, 15-, and 45-second mono waveforms on the local RTX 4060 Laptop GPU.

No DEAM data were downloaded or used. No probe was trained, no formal embeddings were created, and no research protocol was finalized.

## Environment

| Component | Verified value |
|---|---|
| OS platform string | `Windows-10-10.0.26200-SP0` |
| Python | `3.10.21` |
| PyTorch | `2.14.0+cu130` |
| PyTorch CUDA runtime | `13.0` |
| GPU | `NVIDIA GeForce RTX 4060 Laptop GPU` |
| NVIDIA driver | `617.14` |
| GPU memory reported by `nvidia-smi` | `8188 MiB` |
| Final Transformers | `4.24.0` |
| Final tokenizers | `0.13.3` |
| huggingface-hub | `0.36.2` |
| nnAudio | `0.3.4` |
| Inference precision | model-default `torch.float32` |
| Batch size | `1` |
| Input sampling rate | `24,000 Hz` |

The final `pip check` result was: `No broken requirements found.`

## Transformers Compatibility

### Attempt 1 — Transformers 5.17.0

The official loading sequence succeeded:

- `Wav2Vec2FeatureExtractor.from_pretrained(..., trust_remote_code=True)` succeeded;
- `AutoModel.from_pretrained(..., trust_remote_code=True)` succeeded;
- `model.eval()` succeeded;
- `model.to("cuda")` succeeded;
- the model parameters were FP32 on `cuda:0`.

However, the version was not usable for layer-wise probing. Forward computation completed when called with `output_hidden_states=True`, but `outputs.hidden_states` was `None`. The failure was identical at 5, 15, and 45 seconds. The complete harness traceback was:

```text
Traceback (most recent call last):
  File "C:\Users\Rinshinozaki\Documents\ChatGPT\AI music mini proj\mert-emotion-probing\docs\codex_reports\_stage3_sanity_runner.py", line 68, in run_duration
    hidden_shapes = [list(state.shape) for state in outputs.hidden_states]
TypeError: 'NoneType' object is not iterable
```

This is consistent in kind with the previously identified MERT/custom-code versus newer-Transformers API compatibility risk, although it is not the exact `conv_pos_batch_norm` exception reported by another user. No monkey patch or MERT source modification was used.

### Attempt 2 — Transformers 4.44.0

Version 4.44.0 was selected because the official model repository discussion identifies it as a compatibility boundary/workable older release. It passed `pip check`, loaded, returned all 13 hidden states, and completed all forward tests. However, checkpoint loading printed the following scientifically unacceptable warning:

```text
Some weights of the model checkpoint at m-a-p/MERT-v1-95M were not used when initializing MERTModel:
['encoder.pos_conv_embed.conv.weight_g', 'encoder.pos_conv_embed.conv.weight_v']

Some weights of MERTModel were not initialized from the model checkpoint at m-a-p/MERT-v1-95M and are newly initialized:
['encoder.pos_conv_embed.conv.parametrizations.weight.original0',
 'encoder.pos_conv_embed.conv.parametrizations.weight.original1']
```

This appears to be a weight-normalization parameter-name mismatch. Although its tensor shapes and memory measurements were usable as an architectural check, the loaded network was not accepted as the intact pretrained checkpoint because part of the positional convolution was newly initialized.

### Attempt 3 — Transformers 4.24.0 (final)

Version 4.24.0 was selected because the checkpoint's own `config.json` records `transformers_version: 4.24.0`. This was the final version tried.

On the first 4.24.0 load, its older remote-code import checker treated the optional `nnAudio` import in `modeling_MERT.py` as mandatory and stopped before loading weights:

```text
Traceback (most recent call last):
  File "C:\Users\Rinshinozaki\Documents\ChatGPT\AI music mini proj\mert-emotion-probing\docs\codex_reports\_stage3_sanity_runner.py", line 113, in main
    extractor, model = load_model()
  File "C:\Users\Rinshinozaki\Documents\ChatGPT\AI music mini proj\mert-emotion-probing\docs\codex_reports\_stage3_sanity_runner.py", line 37, in load_model
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
  File "C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\lib\site-packages\transformers\models\auto\auto_factory.py", line 455, in from_pretrained
    model_class = get_class_from_dynamic_module(
  File "C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\lib\site-packages\transformers\dynamic_module_utils.py", line 363, in get_class_from_dynamic_module
    final_module = get_cached_module_file(
  File "C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\lib\site-packages\transformers\dynamic_module_utils.py", line 237, in get_cached_module_file
    modules_needed = check_imports(resolved_module_file)
  File "C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\lib\site-packages\transformers\dynamic_module_utils.py", line 134, in check_imports
    raise ImportError(
ImportError: This modeling file requires the following packages that were not found in your environment: nnAudio. Run `pip install nnAudio`
```

Installing only the explicitly required `nnAudio==0.3.4` resolved this import check. The final 4.24.0 load produced no missing, unused, newly initialized, or unexpected checkpoint-weight warning. It printed only a PyTorch deprecation notice for the old weight-normalization API, which does not indicate missing weights.

## Model Loading

Final verified loading behavior:

| Check | Result |
|---|---|
| Feature extractor loaded | Yes |
| Model loaded with `trust_remote_code=True` | Yes |
| Checkpoint weights loaded without mismatch warning | Yes |
| `model.eval()` | Yes |
| Model device | `cuda:0` |
| Model parameter dtype | `torch.float32` |
| Feature extractor sampling rate | `24000` |
| Feature extractor normalization | `do_normalize=True` |
| CUDA model baseline allocated memory | `360.25 MiB` |
| CUDA model baseline reserved memory | `400.00 MiB` |

The model weights and custom code were downloaded to the Hugging Face cache. No DEAM files were downloaded.

## 5-second Forward Test

A synthetic 440 Hz mono sine wave was generated in memory:

- duration: 5 seconds;
- sampling rate: 24,000 Hz;
- waveform shape before batching: `[120000]`;
- feature-extractor output shape: `[1, 120000]`;
- batch size: 1;
- inference mode: `torch.inference_mode()`;
- `output_hidden_states=True`.

The forward pass succeeded on CUDA. It returned 13 hidden states. Every hidden state had shape `[1, 374, 768]`.

## Hidden State Verification

Final verified values under Transformers 4.24.0:

| Item | Result |
|---|---|
| Number of hidden-state tensors | `13` |
| `hidden_states[0]` at 5 s | `[1, 374, 768]` |
| `hidden_states[12]` at 5 s | `[1, 374, 768]` |
| `last_hidden_state` at 5 s | `[1, 374, 768]` |
| Hidden dimension | `768` |
| Final hidden-state shape equals `last_hidden_state` shape | Yes |

The 13 tensors correspond structurally to the pre-Transformer representation plus Transformer layers 1–12. This test verified count and shape, not the semantic content of individual layers.

## Mean Pooling Verification

Temporal mean pooling was applied independently to every returned hidden state with `state.mean(dim=1)`:

```text
[B, T', 768] -> [B, 768]
```

For the 5-second input, each of the 13 tensors changed from `[1, 374, 768]` to `[1, 768]`. The same output shape `[1, 768]` was verified for all 13 layers at 15 and 45 seconds. No probe or trainable pooling operation was used.

## GPU Memory Tests

All final measurements below used Transformers 4.24.0, FP32, batch size 1, `model.eval()`, `torch.inference_mode()`, and `output_hidden_states=True`. Before each forward pass, the CUDA cache was cleared and peak-memory statistics were reset. Peak values include the model's resident memory.

| Duration | Success | T' | Hidden-state shape | Peak Allocated VRAM | Peak Reserved VRAM | Inference Time |
|---:|:---:|---:|---|---:|---:|---:|
| 5 s | Yes | 374 | `[1, 374, 768]` × 13 | 454.92 MiB (0.444 GiB) | 550.00 MiB (0.537 GiB) | 0.361 s |
| 15 s | Yes | 1,124 | `[1, 1124, 768]` × 13 | 671.63 MiB (0.656 GiB) | 734.00 MiB (0.717 GiB) | 0.127 s |
| 45 s | Yes | 3,374 | `[1, 3374, 768]` × 13 | 1,616.29 MiB (1.578 GiB) | 2,336.00 MiB (2.281 GiB) | 0.432 s |

The first 5-second measurement included first-forward CUDA warm-up effects, so the single timing observations are sanity measurements rather than a reliable speed benchmark. The non-monotonic 5/15-second timings should not be interpreted as comparative throughput. No OOM occurred, and no mixed/half-precision fallback was needed or tested.

## Problems Encountered

1. **Transformers 5.17.0:** model loading and computation succeeded, but `outputs.hidden_states` remained `None` despite `output_hidden_states=True`. This blocks the planned layer-wise interface.
2. **Transformers 4.44.0:** hidden states worked, but positional-convolution weight names did not map cleanly and two pretrained parameters were replaced by newly initialized parameters. This version was rejected for scientific use.
3. **Transformers 4.24.0 before nnAudio installation:** the old dynamic-module import checker required `nnAudio`, even though this checkpoint has `feature_extractor_cqt=false`.
4. **Final combination:** Transformers 4.24.0 plus nnAudio 0.3.4 loaded the checkpoint cleanly and passed all tests. `pip check` reported no broken requirements.
5. **Hugging Face cache warning:** Windows symlink support was unavailable, so the cache may use additional disk space. This did not affect model correctness.

No MERT source file was changed, and no monkey patch was applied.

## Implication for Chunk Strategy

The actual hardware results show that 5-second FP32 chunks are a conservative and low-memory option on this RTX 4060 Laptop GPU: the peak was approximately 455 MiB allocated and 550 MiB reserved, including the resident model. This duration also matches the 5-second pretraining context reported by the MERT model card.

However, the hardware test also shows that a complete 45-second item can run successfully at batch size 1 in FP32 while returning all 13 hidden states. Its peak was approximately 1.58 GiB allocated and 2.28 GiB reserved, well below the available 8188 MiB. Therefore, chunking is **not required purely to prevent OOM for a 45-second item** under this tested configuration.

Five-second non-overlapping chunking nevertheless looks technically safe and operationally reasonable, particularly for full songs and for bounding memory consistently. Its research consequences remain undecided: chunk aggregation discards cross-chunk Transformer context and introduces aggregation choices, whereas a single 45-second pass preserves attention across the excerpt but operates far beyond the model's reported 5-second pretraining context. This test does not select between those alternatives and does not finalize the audio-duration protocol.
