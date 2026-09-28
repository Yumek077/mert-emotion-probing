# Module A — Stage 2: Understand MERT-v1-95M

> Temporary Codex research report; not a final Module A research log.
>
> Date: 2026-09-27
>
> Scope: documentation/configuration review only. No model weights were downloaded and no inference was run.

## Fixed project decision (not reconsidered here)

- One DEAM audio item is one sample.
- Targets are the static averaged Valence and Arousal values.
- The main task is continuous regression.
- Dynamic emotion prediction is a future extension, not part of the main experiment.

## 1. What MERT-v1-95M is

### 【Officially confirmed facts】

MERT (Music undERstanding model with large-scale self-supervised Training) is a self-supervised acoustic music representation model. Its pretraining follows a masked-language-model-style idea over audio: parts of the latent audio sequence are masked and the network learns to predict discrete pseudo-labels supplied by teacher representations. The paper identifies a combination of an acoustic teacher based on RVQ-VAE and a musical teacher based on CQT; the v1 model card further states that MERT-v1 uses eight EnCodec codebooks for pseudo-labels and in-batch noise mixture. The aim is to encode both acoustic/timbral and music-specific tonal/pitch information rather than train directly for emotion prediction. Sources: [MERT paper (arXiv)](https://arxiv.org/abs/2306.00107), [official MERT-v1-95M model card](https://huggingface.co/m-a-p/MERT-v1-95M), [official GitHub repository](https://github.com/yizhilll/MERT).

The `95M` checkpoint is the base-size model, approximately 95 million parameters. The official model card reports 20,000 hours of pretraining data, 5-second pretraining context, 12 Transformer layers, hidden dimension 768, feature rate 75 Hz, and training sample rate 24 kHz. The paper reports evaluation across 14 music-understanding tasks. The model is exposed as a bare representation model, without an emotion-regression head. Source: [official model card](https://huggingface.co/m-a-p/MERT-v1-95M).

### 【Inference from those facts】

MERT is suitable as a music representation extractor because its frame-level hidden states are pretrained from large amounts of music and are explicitly intended for downstream use at different layers. In frozen probing, MERT parameters remain unchanged; only a small regression probe is fitted on pooled hidden representations. The experiment therefore asks how readily Valence/Arousal can be decoded from information already present at each representation depth—not how well MERT can learn emotion after fine-tuning. A successful probe is evidence of decodability, not proof that MERT has an explicit or causal “emotion module”; pooling, probe capacity, data split, and acoustic confounds still affect the result.

## 2. Audio input requirements

### 【Officially confirmed facts】

- **Sampling rate:** 24,000 Hz. The checkpoint configuration has `sample_rate: 24000`, and the model card labels the feature rate as 75 Hz. The official example explicitly resamples audio to `processor.sampling_rate` before calling the feature extractor. Sources: [config.json](https://huggingface.co/m-a-p/MERT-v1-95M/raw/main/config.json), [model card usage example](https://huggingface.co/m-a-p/MERT-v1-95M#model-usage).
- **Input representation:** decoded raw waveform values, passed as `input_values`. The underlying HuBERT API expects a float tensor of shape `(batch_size, sequence_length)`. Sources: [MERT custom model forward](https://huggingface.co/m-a-p/MERT-v1-95M/blob/main/modeling_MERT.py), [Transformers HuBERT documentation](https://huggingface.co/docs/transformers/model_doc/hubert).
- **Feature extractor:** the repository supplies a `Wav2Vec2FeatureExtractor` configuration with `feature_size: 1`, right padding with zero, `return_attention_mask: true`, `sampling_rate: 24000`, and `do_normalize: true`. `do_normalize` means zero-mean/unit-variance waveform normalization. It also batches, pads, creates tensors and (for this checkpoint) returns an attention mask. It does **not** replace decoding or resampling; the official example resamples separately. Sources: [preprocessor_config.json](https://huggingface.co/m-a-p/MERT-v1-95M/raw/main/preprocessor_config.json), [Wav2Vec2FeatureExtractor documentation](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2).
- **Compressed audio:** an MP3 is a file encoding, not the model input. It must first be decoded to waveform samples, then resampled to 24 kHz as needed, and then passed to the feature extractor. The official demo likewise obtains a decoded audio array before preprocessing. Source: [official inference script](https://github.com/yizhilll/MERT/blob/main/scripts/MERT_demo_inference.py).
- **Length controls in the feature extractor:** `max_length` and `truncation` are optional call arguments; `truncation` defaults to `False` and `max_length` defaults to `None`. Thus the feature extractor does not silently truncate by default. Source: [Wav2Vec2FeatureExtractor documentation](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2).

### 【Inference from those facts】

- A single decoded mono item is normally a one-dimensional float array `[T]`; after batching it becomes `[B, T]`. A decoded stereo file is normally `[channels, T]` and must be reduced to one channel before being treated as one MERT example. The unchanged checkpoint does not define a stereo-aware input convention. Consistent downmixing (for example, channel mean) is therefore a preprocessing decision; selecting only one channel is another possible but information-losing convention.
- Do not add an unrelated peak-normalization rule merely because the model accepts floats. The checkpoint’s supplied feature extractor already specifies zero-mean/unit-variance normalization. Any extra loudness or peak normalization would change the signal and should be a separate, justified experimental choice.
- MERT’s convolutional/positional-convolution encoder has no documented fixed `max_position_embeddings` ceiling, and its forward method accepts arbitrary waveform sequence length. It is therefore variable-length in the architectural/API sense. However, the published pretraining context was only 5 seconds, so substantially longer input is outside the documented pretraining context distribution even when it is accepted by the code.
- If padding multiple items into a batch, temporal pooling must exclude padded frames using the downsampled attention mask. Otherwise short items receive artificial zero-padding contributions. Processing one item at a time avoids padding but not length-related compute cost.

## 3. Architecture and hidden states

### 【Officially confirmed facts】

The checkpoint is HuBERT-style and contains:

- a seven-layer convolutional waveform feature encoder (`conv_dim = [512 × 7]`);
- convolution kernels `[10, 3, 3, 3, 3, 2, 2]`;
- convolution strides `[5, 2, 2, 2, 2, 2, 2]`;
- a feature projection from the convolutional representation to hidden size 768;
- a positional convolution inside the HuBERT encoder;
- 12 Transformer encoder layers;
- hidden size `D = 768`;
- 12 attention heads;
- feed-forward intermediate size 3072.

The checkpoint configuration sets `feature_extractor_cqt: false`: CQT is relevant to the pretraining teacher/objective, but the released inference path for this checkpoint uses the convolutional waveform frontend, not a CQT frontend concatenated at inference. Sources: [config.json](https://huggingface.co/m-a-p/MERT-v1-95M/raw/main/config.json), [MERT modeling code](https://huggingface.co/m-a-p/MERT-v1-95M/blob/main/modeling_MERT.py).

With `output_hidden_states=True`, the official example reports 13 representations of shape `[13, time steps, 768]` after removing the batch dimension. Hugging Face defines `hidden_states` as the initial encoder/embedding representation plus the output of every layer. Sources: [official model card](https://huggingface.co/m-a-p/MERT-v1-95M#model-usage), [Transformers model-output documentation](https://huggingface.co/docs/transformers/main_classes/output).

### 【Inference from those facts】

For the 12-layer model:

- `hidden_states[0]`: the encoder input after convolutional extraction, projection to 768 dimensions, and encoder positional preprocessing; it is a contextualization baseline, not the raw waveform and not simply the unprojected 512-channel convolution output.
- `hidden_states[1]`: output of Transformer layer 1.
- …
- `hidden_states[12]`: output of Transformer layer 12; this should coincide with `last_hidden_state` in ordinary evaluation mode.

The `L + 1` count therefore arises because the initial projected/position-aware representation is retained in addition to all `L` layer outputs. A layer-wise study must explicitly state whether “layer 0” is included and name it as the pre-Transformer encoder representation rather than Transformer layer 0.

## 4. Tensor shapes and temporal downsampling

Let the decoded waveform tensor be:

`input_values.shape = [B, T]`

where:

- `B` = batch size / number of audio items processed together;
- `T` = waveform sample count after resampling (for 45 seconds at 24 kHz, `T = 1,080,000`).

The model returns:

`last_hidden_state.shape = [B, T', D]`

and, for every `l = 0…12`:

`hidden_states[l].shape = [B, T', D]`

where:

- `T'` = number of downsampled audio frames after the convolutional frontend;
- `D = 768`.

### 【Officially confirmed facts】

The seven convolution strides multiply to `5 × 2^6 = 320`, and the model card reports an approximately 75 Hz output feature rate at 24 kHz. Source: [config.json](https://huggingface.co/m-a-p/MERT-v1-95M/raw/main/config.json), [official model card](https://huggingface.co/m-a-p/MERT-v1-95M).

### 【Inference/calculation from the official configuration】

Each output step advances about 320 waveform samples: `320 / 24000 = 13.33 ms`, or about 75 frames/second. Applying the kernel/stride output-length formula through all seven convolutions gives approximately 3,374 frames for exactly 45 seconds (the simple 75 Hz estimate is 3,375; edge effects explain the one-frame difference). The convolutional receptive field is approximately 400 waveform samples, about 16.7 ms, while the step between adjacent output frames is about 13.3 ms.

Thus `T'` is much smaller than `T` because each convolution stage summarizes overlapping local waveform regions and strides forward by more than one sample. The Transformer attends over these downsampled frames, not over 1.08 million individual waveform samples.

## 5. From a frame sequence to one item vector: pooling options

For a layer representation `H ∈ R^(T'×D)`, the project needs a fixed-size item vector before fitting an item-level Valence/Arousal probe.

| Method | Result | Meaning for static item-level emotion | Advantages | Risks / limitations |
|---|---:|---|---|---|
| **A. Temporal mean**: `z = mean_t(H_t)` | `D = 768` | Average activation over the item | Simple, deterministic, length-invariant; explicitly demonstrated by the official MERT example for utterance-level tasks; low-dimensional and easy to compare layer-wise | Smooths away brief affective peaks and temporal order; must mask padding; a long quiet/neutral region can dominate |
| **B. Temporal max**: `z_d = max_t H_(t,d)` | `D = 768` | Strongest activation seen in each feature dimension | Preserves rare/high-intensity evidence that mean pooling can suppress | Different dimensions may select different moments, so the vector is not a coherent frame; sensitive to transients, decoding artifacts and outliers; ignores prevalence/duration |
| **C. Special token** | potentially `D` | Global summary token, if one were pretrained for that purpose | Convenient in architectures with a trained classification token | The released bare MERT/HuBERT interface documents frame outputs, not a trained `[CLS]`/global token or pooler. Taking the first time frame would merely select an ordinary audio frame and is not a justified global representation |
| **D. Mean + standard deviation** | `2D = 1536` | Average content plus temporal variability | Retains dispersion/dynamics while remaining order-agnostic; potentially useful robustness extension | Doubles probe input size and parameter count; scale/regularization becomes more important; still loses temporal order and may reward noisy variability |

### 【Officially confirmed fact】

The MERT model card explicitly demonstrates temporal mean reduction from `[13, time, 768]` to `[13, 768]`. It does not prescribe it as the uniquely correct method. Source: [official model card usage](https://huggingface.co/m-a-p/MERT-v1-95M#model-usage).

### 【Inference】

There is no suitable pretrained special token in the documented MERT-v1-95M interface. Mean, max, and mean+std are therefore genuine aggregation choices; a “first token” strategy would not be equivalent to BERT-style CLS pooling. Pooling must be fixed consistently across layers to make layer-wise comparisons interpretable, unless pooling itself is an explicit experimental factor.

## 6. The practical 45-second and full-song problem

### 【Officially confirmed facts】

- MERT-v1-95M was pretrained with 5-second contexts and produces about 75 frames/second. Source: [official model card](https://huggingface.co/m-a-p/MERT-v1-95M).
- The model’s attention implementation forms attention scores whose sequence axes are `target_length × source_length`; for self-attention these are both `T'`. Source: [MERT modeling code](https://huggingface.co/m-a-p/MERT-v1-95M/blob/main/modeling_MERT.py).

### 【Inference/calculation】

Self-attention time and its main score/probability memory scale approximately as `O(T'^2)`. A 45-second input has about 3,374 frames. One 12-head attention-score tensor at that length contains about 136.6 million values—roughly 0.51 GiB in FP32 or 0.25 GiB in FP16—before accounting for model weights, Q/K/V tensors, hidden states, temporary copies and CUDA allocator overhead. In inference mode only one layer’s working tensors need dominate at a time, but requesting all 13 hidden states also retains roughly 129 MiB of FP32 frame features for batch size 1.

Consequently, a complete 45-second excerpt at batch size 1 is **plausible but not guaranteed** on an RTX 4060 Laptop GPU, especially with `eval()`, inference/no-grad mode, and an appropriate floating-point precision. It should be verified with a small dedicated memory test after the software-compatibility issue below is fixed. Large batches are unlikely to be attractive. Chunking is a safe fallback, but it is not yet proven mandatory for 45 seconds.

Full songs are qualitatively harder. For illustration, four minutes at 75 Hz is about 18,000 frames; a single FP32 12-head square attention tensor would already be roughly 14.5 GiB, before all other allocations. Feeding full multi-minute songs in one pass is therefore not realistic on this GPU class. The precise cutoff depends on song duration, precision, implementation and available VRAM.

If chunking is chosen while retaining **one audio item = one sample**, chunks should remain internal representation-extraction units, not become independent supervised samples carrying duplicated static labels. A defensible aggregation path is:

1. split an item into fixed-duration chunks (optionally with documented overlap);
2. extract the same hidden layer(s) for each chunk;
3. pool valid frames within each chunk;
4. combine chunk vectors into one item vector, ideally weighting by valid frame count/duration when the last chunk is shorter.

For non-overlapping chunks, concatenating all valid frame representations and then taking a global mean is equivalent to a valid-frame-count-weighted mean of chunk means. With overlap, naïve averaging overweights overlapped audio and must be documented.

### Duration/full-song options (analysis only)

| Option | What it means | Advantages | Risks |
|---|---|---|---|
| **A. All items use a strict uniform 45 seconds** | Apply one common 45-second extraction rule to every item | Uniform input length, compute, and number of MERT frames; clean comparison | Requires a documented segment position for long songs and possibly trim/pad rules around nominal excerpts; discards music; static item labels may not describe the selected segment equally well |
| **B. Keep 45-second excerpts intact; crop only the 58 full songs to a fixed 45 seconds** | Treat the two DEAM audio groups differently only where necessary | Preserves the standard excerpts exactly and limits compute; changes fewer files | The crop position for full songs can introduce bias; full-song static annotations may not match one selected excerpt; remaining group difference may act as a confound |
| **C. Chunk every item and aggregate to one vector** | Use multiple bounded MERT passes and combine them before probing | Scales to long audio and can use all available content; respects one-item/one-sample if aggregation precedes the probe | More compute/storage and more design choices (chunk length, hop, overlap, weighting); model saw 5-second pretraining contexts; unequal item duration can change representation reliability |
| **D. Exclude the 58 full songs** | Restrict analysis to the approximately 45-second collection | Most homogeneous duration and simplest extraction | Reduces sample size and coverage; may introduce selection bias; conclusions no longer cover the full annotated collection |

No option is selected in this report.

## 7. Layer-wise probing interface

### 【Officially confirmed fact】

The checkpoint’s custom `forward` accepts `output_hidden_states`, and the official example reads `outputs.hidden_states`. Loading requires `trust_remote_code=True`. Source: [MERT custom model](https://huggingface.co/m-a-p/MERT-v1-95M/blob/main/modeling_MERT.py), [official inference script](https://github.com/yizhilll/MERT/blob/main/scripts/MERT_demo_inference.py).

Illustrative only (not project code):

```python
from transformers import AutoModel, Wav2Vec2FeatureExtractor

processor = Wav2Vec2FeatureExtractor.from_pretrained(
    "m-a-p/MERT-v1-95M", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "m-a-p/MERT-v1-95M", trust_remote_code=True
)
inputs = processor(waveform_24k_mono, sampling_rate=24_000, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)
all_hidden_states = outputs.hidden_states  # 13 tensors: [B, T', 768]
```

For eventual reproducibility, the Hub model revision should be pinned after a compatible stack is verified; `trust_remote_code=True` otherwise allows the repository’s current custom code to define behavior.

## 8. Compatibility and hardware findings requiring attention

### 【Officially confirmed facts】

- The model’s `config.json` records `transformers_version: 4.24.0` and its custom code imports internal HuBERT implementation classes. Source: [config.json](https://huggingface.co/m-a-p/MERT-v1-95M/raw/main/config.json), [modeling_MERT.py](https://huggingface.co/m-a-p/MERT-v1-95M/blob/main/modeling_MERT.py).
- A Hugging Face model discussion reports that Transformers versions newer than 4.44.0 can fail with `AttributeError: 'MERTConfig' object has no attribute 'conv_pos_batch_norm'`; the reported workarounds are to use 4.44.0 or earlier or explicitly add that config field. This is a community report on the official model repository, not a maintainer release guarantee. Source: [official model-repository discussion #4](https://huggingface.co/m-a-p/MERT-v1-95M/discussions/4).
- This repository’s current `environment.yml` pins `transformers==5.17.0`.

### 【Inference / immediate implication】

There is a **high-priority compatibility risk before first model loading**: the project’s Transformers 5.17.0 is newer than the reported breaking threshold and much newer than the checkpoint’s recorded 4.24.0. This stage did not download or load the checkpoint, so failure has not been reproduced locally. Before any MERT experiment, the project should perform a controlled compatibility decision/test—e.g. select and pin a known-compatible Transformers version, or deliberately patch/test the missing configuration field. The environment must not be changed casually because downgrading Transformers can affect other dependencies.

`nnAudio` is not required by the configured inference path because `feature_extractor_cqt` is `false`; the custom file may print a warning when it cannot import `nnAudio`, but it only asserts that dependency when CQT frontend extraction is enabled. No extra package should be added solely on the basis of that warning without an actual requirement.

There is no evidence of an inherent CUDA-model incompatibility with an RTX 4060 Laptop GPU. The hardware issue is capacity: 45 seconds needs batch-size/precision verification, while multi-minute full-song single-pass attention is not practical.

## 9. Decisions for the researcher

### Decision 1 — Sampling rate

| Option | Trade-off |
|---|---|
| Resample every decoded item to **24 kHz on the fly** | Matches the checkpoint; saves duplicate audio files; adds repeated CPU work and makes the resampler implementation part of reproducibility |
| Precompute/cache **24 kHz** decoded audio | Matches the checkpoint and speeds repeated extraction; requires extra storage and a clearly versioned preprocessing artifact |
| Keep native/another rate | Avoids resampling but violates the checkpoint’s documented input rate and changes time/frequency interpretation; not a comparable use of pretrained MERT-v1-95M |

The model-compatible rate is fixed at 24 kHz; the real open choice is when/how to resample and whether to cache it.

### Decision 2 — Mono/stereo handling

| Option | Trade-off |
|---|---|
| Arithmetic mean of left/right channels | Deterministic and uses both channels; phase cancellation can occur |
| Library-standard mono downmix | Convenient and usually sensible; exact coefficients/behavior must be documented and version-stable |
| Select one channel | Simple, no cancellation; discards the other channel and can bias unusual mixes |
| Process channels separately then aggregate | Preserves stereo information longer; doubles compute and is not the checkpoint’s standard input path |

### Decision 3 — Audio duration / full-song handling

Choose among strict uniform 45 seconds, preserve excerpts and crop only full songs, chunk-and-aggregate every item, or exclude the 58 full songs. The main trade-off is standardization/simplicity versus retaining content and dataset coverage; any chunking must still yield one final vector per DEAM item.

### Decision 4 — Pooling strategy

Choose temporal mean, max, or mean+std (possibly as a later robustness variant). Mean is the official example and the simplest `768-D` baseline; max emphasizes rare events but is outlier-sensitive; mean+std adds temporal variability but doubles dimensionality. A CLS/special-token strategy is not supported by the documented checkpoint.

### Decision 5 — Hidden representations in layer-wise probing

| Option | Trade-off |
|---|---|
| All 13 (`hidden_states[0]` plus Transformer layers 1–12) | Complete depth profile and a useful pre-Transformer baseline; increases multiple comparisons and requires clear naming |
| Transformer outputs only (1–12) | “Layer-wise” terminology is simpler; loses the projected/frontend baseline |
| Selected layers (e.g. early/middle/late/final) | Lower compute/storage and fewer tests; can miss a non-monotonic peak and makes selection rationale important |
| Final layer only | Cheapest conventional baseline; does not answer the main layer-wise research question by itself |

## 10. Compact pipeline interpretation

```text
DEAM audio file
→ decode MP3 to floating-point waveform
→ consistent mono conversion
→ resample to 24 kHz
→ Wav2Vec2FeatureExtractor: normalize, batch/pad, attention mask
→ [B,T] waveform
→ 7-layer convolutional frontend (≈75 frames/s)
→ projection + positional preprocessing
→ hidden_states[0]: [B,T',768]
→ 12 frozen Transformer encoder layers
→ hidden_states[1…12]: each [B,T',768]
→ chosen valid-frame pooling / optional chunk aggregation
→ one fixed-dimensional vector per DEAM item
→ later: simple continuous Valence/Arousal regression probe
```

## Primary official sources

1. [MERT official GitHub repository](https://github.com/yizhilll/MERT)
2. [MERT-v1-95M official Hugging Face model card](https://huggingface.co/m-a-p/MERT-v1-95M)
3. [MERT-v1-95M config.json](https://huggingface.co/m-a-p/MERT-v1-95M/raw/main/config.json)
4. [MERT-v1-95M preprocessor_config.json](https://huggingface.co/m-a-p/MERT-v1-95M/raw/main/preprocessor_config.json)
5. [MERT-v1-95M custom modeling code](https://huggingface.co/m-a-p/MERT-v1-95M/blob/main/modeling_MERT.py)
6. [MERT paper, arXiv:2306.00107](https://arxiv.org/abs/2306.00107)
7. [Hugging Face Transformers HuBERT documentation](https://huggingface.co/docs/transformers/model_doc/hubert)
8. [Hugging Face Wav2Vec2FeatureExtractor documentation](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2)
9. [Hugging Face model-output documentation](https://huggingface.co/docs/transformers/main_classes/output)

Supplementary compatibility evidence: [MERT-v1-95M discussion #4](https://huggingface.co/m-a-p/MERT-v1-95M/discussions/4). This is clearly treated above as a community report, not as peer-reviewed or maintainer-confirmed documentation.
