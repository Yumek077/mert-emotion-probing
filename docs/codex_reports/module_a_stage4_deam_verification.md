# Module A — Stage 4: DEAM Acquisition & Dataset Verification

Date: 2026-09-27

Status: temporary research report; not a final Module A research log

## Purpose

This stage acquired the official DEAM audio, annotations, and metadata and verified the actual local files before any modeling. It inventories every MP3, validates static averaged Valence/Arousal labels, checks audio–label identity, describes the official metadata subsets, and inspects label distributions.

No DEAM split was created. MERT was not run, no representation was extracted, and no probe was trained.

Evidence labels used below:

- **Official documentation:** a statement made by the DEAM official website or manual.
- **Direct observation:** computed from the downloaded official files in this local copy.
- **Interpretation:** a limited implication or trade-off inferred from those facts; it is not treated as an observed fact.

## Downloaded Official Files

**Official documentation.** The [University of Geneva DEAM page](https://cvml.unige.ch/databases/DEAM/) describes 1,802 excerpts and full songs and directly links the audio, annotation, and metadata archives. It also offers precomputed openSMILE features, but those were not required for this project and were not downloaded. The [official manual](https://cvml.unige.ch/databases/DEAM/manual.pdf) was used to interpret the subsets and annotation files.

**Direct observation.** Three archives were downloaded from the official host:

| File | Official URL | Downloaded size | SHA-256 | Extraction location |
|---|---|---:|---|---|
| `DEAM_audio.zip` | `https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip` | 1,343,203,527 bytes (1.251 GiB) | `E2814325F573ADE8CAEBC697F0EFCFDF7207142B737D72558A416117CA591353` | `data/raw/deam/audio/` |
| `DEAM_Annotations.zip` | `https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip` | 4,735,283 bytes (4.516 MiB) | `809B0FEB4BA6196C1EDA9B2B7B33B892B11FA71457CFCEB0D47E832ED1A5F15F` | `data/raw/deam/annotations/` |
| `metadata.zip` | `https://cvml.unige.ch/databases/DEAM/metadata.zip` | 344,760 bytes (0.329 MiB) | `3D1D5AB42E852803A770F2CFBD685C6B464E1EBE76F26B0E10954597902B90F8` | `data/raw/deam/metadata/` |

All three archives opened successfully before extraction. The official internal directory names were retained instead of flattening or renaming the files.

## Local Data Structure

**Direct observation.** The relevant local layout is:

```text
data/raw/deam/
├── audio/
│   └── MEMD_audio/                    # 1,802 MP3 files
├── annotations/
│   └── annotations/
│       ├── annotations averaged per song/
│       │   ├── dynamic (per second annotations)/
│       │   └── song_level/            # two static averaged CSVs
│       └── annotations per each rater/
├── metadata/
│   └── metadata/
│       ├── metadata_2013.csv
│       ├── metadata_2014.csv
│       └── metadata_2015.csv
├── downloads/                         # the three original ZIP files
└── verification/                      # temporary generated CSV/JSON files
```

The full raw tree is covered by the existing `data/raw/*` Git ignore rule. The verification script writes temporary tables to `data/raw/deam/verification/`, so they are also ignored.

Temporary generated outputs:

- `data/raw/deam/verification/deam_audio_inventory.csv`
- `data/raw/deam/verification/deam_static_annotations_unified.csv`
- `data/raw/deam/verification/deam_item_mapping.csv`
- `data/raw/deam/verification/deam_verification_summary.json`

## Audio Inventory

The inventory used `soundfile.info()` on every actual MP3; it did not rely only on the manual.

### Counts and duration

**Direct observation.**

| Measure | Result |
|---|---:|
| MP3 files | 1,802 |
| Readable MP3 files | 1,802 |
| Unreadable or malformed MP3 files | 0 |
| Duplicate numeric audio IDs | 0 |
| Approximately 45-second items (`44.0 ≤ duration ≤ 46.0`) | 1,744 |
| Items longer than 60 seconds | 56 |
| Metadata-defined 2015 full songs | 58 |
| Minimum duration | 44.643 s (ID 272) |
| Maximum duration | 628.616 s (ID 2011) |
| Median duration, all items | 45.035 s |

All 1,744 non-2015 items fall within 44.643–45.605 seconds (mean 45.052 s, SD 0.098 s). The 58 items associated with `metadata_2015.csv` range from 49.563 to 628.616 seconds (median 228.780 s; mean 234.579 s).

Two metadata-defined full songs are not longer than the descriptive 60-second threshold:

- ID 2016: 56.921 seconds;
- ID 2017: 49.563 seconds.

**Interpretation.** “Full song” must be determined from the official subset/metadata, not from an arbitrary duration threshold. The observed counts exactly support 1,744 approximately 45-second excerpts plus 58 full-song items, even though only 56 exceed 60 seconds.

### Sampling rate

**Direct observation.**

| Sample rate | Count |
|---:|---:|
| 44,100 Hz | 1,778 |
| 48,000 Hz | 20 |
| 22,050 Hz | 3 |
| 16,000 Hz | 1 |

All 58 metadata-defined full songs are 44.1 kHz. All 24 non-44.1-kHz items occur in the 2014 metadata subset and are approximately 45 seconds long. IDs are preserved in the generated JSON summary and inventory; the 16 kHz item is ID 1024.

**Official documentation.** The manual states that the 45-second excerpts were re-encoded to 44.1 kHz.

**Interpretation / anomaly.** The actual archive does not fully match that general statement: 24 of the 1,744 excerpts report another sampling rate. This does not block the project because on-the-fly resampling to MERT's required 24 kHz is already a fixed decision, but source-rate heterogeneity must not be assumed away.

### Channels

**Direct observation.**

| Channels | Count |
|---:|---:|
| Stereo (2) | 1,789 |
| Mono (1) | 13 |

All 58 full songs are stereo. The 13 mono files are excerpts: IDs 811, 990, 1203, 1218, 1230, 1248, 1286, 1373, 1708, 1825, 1866, 1867, and 1869.

**Interpretation.** The fixed arithmetic-channel-mean rule should be applied conditionally: average channels when `channels > 1`, and leave a mono waveform unchanged. Treating every file as necessarily stereo would be incorrect.

## Static Annotation Schema

**Direct observation.** The averaged song-level labels are split across two official files:

1. `static_annotations_averaged_songs_1_2000.csv`
   - `song_id`
   - `valence_mean`, `valence_std`
   - `arousal_mean`, `arousal_std`
2. `static_annotations_averaged_songs_2000_2058.csv`
   - the same core fields;
   - additional Valence/Arousal minimum and maximum summary fields;
   - a minor original header inconsistency: `valence_ max_mean` contains an extra space.

The verification script strips surrounding column whitespace and combines the files without rescaling labels. The project-relevant unified fields are:

```text
song_id
valence_mean
valence_std
arousal_mean
arousal_std
```

The additional 2015 min/max fields are retained in the temporary unified CSV but are not substituted for the main targets.

### Static label integrity and descriptives

**Direct observation.**

| Measure | Valence | Arousal |
|---|---:|---:|
| Non-missing items | 1,802 | 1,802 |
| Missing items | 0 | 0 |
| Minimum | 1.6 | 1.6 |
| Maximum | 8.4 | 8.1 |
| Mean | 4.904 | 4.814 |
| Standard deviation | 1.174 | 1.282 |
| Median | 4.9 | 4.9 |

The annotator-disagreement fields are present for all items:

- `valence_std`: 0.30–2.90, mean 1.502;
- `arousal_std`: 0.37–2.59, mean 1.463.

There are 1,802 rows, 1,802 unique IDs, no duplicate IDs, and no missing Valence or Arousal means.

**Official documentation.** The manual describes static whole-song ratings on a 1–9 scale. The observed means occupy a narrower empirical range inside that scale. No normalization or rescaling was applied.

## Audio–Label Integrity

This was the central integrity check.

**Direct observation.**

| Check | Result |
|---|---:|
| Audio IDs | 1,802 |
| Static-label IDs | 1,802 |
| Matched IDs | 1,802 |
| Audio without static label | 0 |
| Static label without audio | 0 |
| Duplicate audio IDs | 0 |
| Duplicate static-label IDs | 0 |
| Malformed audio filenames | 0 |
| Unreadable MP3 files | 0 |

No sample was silently dropped or repaired.

## Year / Subset Structure

**Direct observation.** Mapping IDs by the official metadata filenames gives:

| Official metadata file | IDs represented | Audio type observed |
|---|---:|---|
| `metadata_2013.csv` | 744 | approximately 45-second excerpts |
| `metadata_2014.csv` | 1,000 | approximately 45-second excerpts |
| `metadata_2015.csv` | 58 | full songs |

All 1,802 audio IDs map to exactly one metadata file; there are no cross-file duplicate IDs and no unknown mappings.

**Official documentation.** The manual describes a 744-song development set, a 1,000-song evaluation set, and a 58-song 2015 evaluation set, and states that 2013/2014 data use 45-second excerpts while 2015 uses full songs. It also describes reuse of subsets across MediaEval years.

**Interpretation / caution.** The report's `year_or_subset` field is derived directly from the metadata filename. Because the manual discusses campaign development/evaluation reuse, this filename-derived value should not automatically be treated as an independent recording year, annotation year, or experimental split. For this local verification it is a provenance/subset indicator only.

The resulting minimal mapping is stored in `deam_item_mapping.csv` with `song_id`, filename, duration, `year_or_subset`, `is_full_song`, audio properties, and static labels.

## Label Distribution

Inspection figures:

- `docs/codex_reports/figures/deam_valence_mean_histogram.png`
- `docs/codex_reports/figures/deam_arousal_mean_histogram.png`
- `docs/codex_reports/figures/deam_valence_arousal_scatter.png`

**Direct observation.** Across all 1,802 items, Pearson's descriptive correlation between static Valence and Arousal means is `r = 0.570`. The scatter is visibly concentrated in the middle of the 1–9 rating scale, with relatively few values near the observed extremes. Labels lie on a visible discrete grid because the provided means are rounded/quantized values.

Descriptives by audio/subset type:

| Group | n | Valence mean ± SD | Valence range | Arousal mean ± SD | Arousal range | Valence–Arousal r |
|---|---:|---:|---:|---:|---:|---:|
| 45-second excerpts | 1,744 | 4.903 ± 1.174 | 1.6–8.4 | 4.813 ± 1.289 | 1.6–8.1 | 0.588 |
| 2015 full songs | 58 | 4.924 ± 1.194 | 3.0–7.2 | 4.857 ± 1.052 | 3.0–7.0 | -0.081 |

**Interpretation.** The group means are similar, while the 58-song subset has a visibly narrower observed Arousal range and a different descriptive within-group correlation. These observations must not be interpreted as a hypothesis test or causal effect: the full-song sample is small, and full-song status is perfectly tied to the 2015 metadata subset and its annotation protocol.

## 45-second Excerpts vs Full Songs

The two candidate dataset protocols are:

### Option A — Use the 1,744 approximately 45-second excerpts

- **Direct observation:** retains `1744 / 1802 = 96.7814%` of all items.
- **Advantage:** duration is extremely consistent (44.643–45.605 s); a uniform single-pass or uniform chunking rule is easy to reproduce.
- **Advantage:** avoids mixing excerpt labels with whole-song labels and avoids a separate full-song duration policy.
- **Risk:** removes `58 / 1802 = 3.2186%` of data and removes the entire `metadata_2015.csv` subset rather than a random 3.2%.
- **Risk:** conclusions would explicitly apply to the excerpt collection, not the complete DEAM release.
- **Potential bias:** if 2015 songs differ in music source, annotation protocol, or label distribution, excluding them changes population coverage.

### Option B — Use all 1,802 items

- **Direct observation:** static labels and readable audio are complete for all items.
- **Advantage:** retains the complete official collection and its 2015 coverage.
- **Risk:** duration changes from a near-constant 45 seconds to a 49.6–628.6-second full-song range for the additional 58 items.
- **Risk:** full-song status, metadata subset, and annotation design are confounded: every full song is in the 2015 file and every non-2015 item is an excerpt.
- **Risk:** requires additional choices about cropping versus chunking, remainder handling, chunk aggregation, and compute normalization.
- **Reproducibility cost:** the protocol must describe those choices precisely; “use all audio” is not itself an executable duration policy.

**Official documentation.** The manual states that static labels should be suitable for all songs, while also noting that 2014 and 2015 label sets were collected differently. It reports stronger certainty for 2015 dynamic annotations; that statement should not be generalized into a claim that the 2015 static means are categorically superior.

**Interpretation.** The numerical sample-size gain from Option B is modest, but sample percentage alone is not decisive. The main trade-off is homogeneous input/annotation conditions versus complete subset coverage. No option is selected here.

## Audio Duration Strategy Considerations

### A. 45-second single pass for excerpts

- **Direct observation:** all 1,744 excerpts can be represented by one approximately 45-second input.
- **Prior verified hardware fact:** the Stage 3 FP32 test successfully processed 45 seconds with all 13 hidden states on this RTX 4060 Laptop GPU, peaking at approximately 1.58 GiB allocated / 2.28 GiB reserved VRAM.
- **Advantage:** preserves cross-time self-attention over the whole excerpt and requires only one item-level temporal mean.
- **Risk:** MERT's official model card reports a 5-second pretraining context, so a 45-second inference context is longer than the documented training context.
- **Full-song issue:** this rule still requires cropping, exclusion, or a separate aggregation policy for the 58 full songs.

### B. 5-second non-overlapping chunks plus item aggregation

- **Prior verified hardware fact:** a 5-second FP32 forward is very safe on the tested GPU (approximately 455 MiB allocated / 550 MiB reserved including the model).
- **Advantage:** aligns chunk length with the documented MERT pretraining context and bounds memory for full songs.
- **Advantage:** can cover an entire item while preserving the fixed decision that one DEAM item—not one chunk—is one supervised sample.
- **Risk:** removes Transformer attention across chunk boundaries and requires an explicit valid-duration weighting/remainder rule.
- **Risk:** the approximately 45-second files are not all exactly 45.000 seconds; a nine-chunk policy needs a stated treatment for the final fractional remainder.
- **Full-song issue:** full songs yield very different chunk counts. Chunk vectors should be aggregated before probing; treating chunks as independent samples would violate the fixed sample definition and inflate long songs.

### C. 45-second primary analysis plus 5-second chunking as a later ablation

- **Advantage:** gives the homogeneous 1,744-excerpt main task a simple primary representation while explicitly testing sensitivity to context length later.
- **Risk:** roughly doubles representation-extraction workflows and adds an analysis branch; it does not itself solve how the 58 full songs enter the primary dataset.

### If full songs are retained

Additional choices would be unavoidable:

1. crop each full song to a fixed 45-second segment (with a documented segment-selection rule), risking mismatch between a full-song static label and the chosen segment;
2. chunk the complete full song and combine chunk vectors using a documented duration/frame weighting rule;
3. define overlap and final-partial-chunk handling;
4. control compute and caching for songs up to 628.6 seconds;
5. acknowledge that full-song handling is inseparable from the 2015 subset in this dataset.

No duration strategy is selected in this report.

## Data Quality / Risks

1. **No audio–label integrity failure was found:** all 1,802 IDs match and all MP3s are readable.
2. **Source sample rates are heterogeneous:** 24 excerpts differ from 44.1 kHz despite the manual's general 44.1-kHz statement. The fixed 24-kHz on-the-fly resampling step must inspect each file's real source rate.
3. **Channel counts are heterogeneous:** 13 excerpts are mono. Arithmetic stereo averaging must not assume two channels unconditionally.
4. **Full-song status is not a pure duration threshold:** two official 2015 full songs are shorter than 60 seconds.
5. **Subset and duration are confounded:** 2013/2014 metadata files contain excerpts; the 2015 file contains full songs.
6. **Static annotation schema differs:** the 2015 label file contains additional min/max fields and a minor header inconsistency.
7. **Label distributions differ descriptively:** the small full-song group has narrower observed Arousal coverage and a different within-group Valence–Arousal correlation. No significance or causal claim is made.
8. **Metadata schemas differ strongly across years:** the minimal ID/file provenance mapping is reliable, but fields such as titles, genres, and tags should not be combined by assuming identical columns.
9. **Licensing/repository hygiene:** raw DEAM audio and annotations remain under Git-ignored `data/raw/` and must not be redistributed through this repository.

## Decisions Still Required

1. **Dataset scope:** use the homogeneous 1,744 excerpt items or all 1,802 items including the 58-item 2015 full-song subset.
2. **Primary duration protocol:** 45-second single-pass, 5-second chunk-and-aggregate, or 45-second primary with a later 5-second ablation.
3. **If full songs are retained:** decide crop versus full-song chunks, crop position, chunk remainder/overlap policy, and duration-weighted aggregation.
4. **Protocol wording:** decide whether filename-derived `2013/2014/2015` should be called “metadata subsets” rather than experimental years/splits in the formal log.

## Reproducible Verification Tool

The retained script is:

```text
scripts/verify_deam.py
```

It uses repository-relative/configurable paths, reads every MP3, regenerates the temporary inventory and integrity summary, unifies static annotations, builds the minimal metadata mapping, and regenerates the three figures. It contains no MERT inference or training code.
