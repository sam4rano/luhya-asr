# Luhya ASR — Cross-Model Comparison Protocol

**Technical report: aligning CTC (Wav2Vec2-BERT) and Seq2Seq (Whisper-small) training pipelines so their results are directly comparable**

Date: 18 Aug 2026
Repo: `luhya-asr`

---

## 1. Context and motivation

Luhya is a low-resource Bantu language spoken in Kenya. This project fine-tunes two
pretrained ASR model families on ~40 hours of Luhya speech to compare them under
**identical experimental conditions**:

| | Whisper-small | Wav2Vec2-BERT 2.0 |
|---|---|---|
| Family | Seq2Seq (encoder–decoder) | CTC (encoder + CTC head) |
| Size | ~244M params | ~580M params |
| Pretraining | 680k h weak supervision | 4.5M h unlabeled (Seamless) |
| Audio features | 80-dim log-mel (fixed 30 s window) | 80-dim mel (SeamlessM4T) |
| Decoding | Greedy generation (beam=1) | CTC argmax (no LM) |

The repository previously contained **two independent, incompatible training
pipelines** with different configuration schemas. They selected different data,
evaluated on different sets, and normalized text differently — so their WER/CER
numbers were **not comparable** (the Whisper model card explicitly flagged this).

### Problems identified in the original configs

| Aspect | Whisper config | Wav2Vec2 config | Impact |
|---|---|---|---|
| Dataset source | `Digital-Divide-Data/...` | `DDD-Kenya/...` (deprecated) | two repo IDs for the same data |
| Split policy | 80/10/10 speaker-disjoint + test | 80/20 random rows, **no test set** | different eval sets → not comparable |
| Audio window | min 0.2 s, max 30 s | max 30 s only | different clips in/out |
| Hour cap | 40 h after filtering, at-or-under | 40 h before filtering, may overshoot | different 40 h selection |
| Learning rate | 1e-5 | 5e-5 (5×) | different optimization schedule |
| Warmup | 5% (fraction) | 10% ratio, or 500 steps (zeroed by a bug) | different LR schedule |
| Weight decay / grad clip | 0.01 / 1.0 | not set | different regularization |
| Optimizer | adamw_torch_fused | adamw_torch (betas 0.9/0.98) | different optimizer state |
| Precision | fp16 (T4 x2) | bf16 or fp16 | fine (hardware-specific) |
| Text normalization | NFC + lowercase + apostrophes | char-set stripping + accents | **references differ → WER not comparable** |
| Eval metric | WER/CER on val + test | WER/CER on 20% val | different denominators |

---

## 2. Objective

Make the two pipelines run on **the same conditions** so a head-to-head comparison
is scientifically valid:

1. Same data (same repo, same 40 h train selection, same 2–30 s window).
2. Same eval protocol (same deterministic speaker-disjoint 80/10/10 splits;
   identical reference strings).
3. Same hyperparameters (lr, warmup, weight decay, grad clip, batch, epochs),
   with precision chosen per hardware.
4. Same text normalization for labels and scoring.

---

## 3. Approach

### 3.1 Single shared data-preparation module

All split / filter / hour-cap logic was extracted from the Whisper script into a
shared module **`src/data/splits.py`**, and both pipelines now call it. This
guarantees the *exact same* clips enter both trainers.

- **Deterministic speaker-disjoint 80/10/10 split** (seed 42). Pre-existing
  splits are kept only if speaker-disjoint; otherwise a deterministic rebuild runs.
- **Safe fallbacks (edge cases):**
  - too few speakers (< 3 for 3-way, < 2 for 2-way) → deterministic row-level split;
  - `test_ratio: 0` → clean two-way 80/20 split (for a pipeline that prefers no test set);
  - ratio validation (`validation_ratio > 0`, `test_ratio ≥ 0`, `val + test < 1`);
  - audio-window validation (`0 ≤ min ≤ max`, `max > 0`).
- **Common filtering:** audio decodes once; corrupt/empty audio, blank transcripts,
  and clips outside **2–30 s** are removed before training.
- **Deterministic 40-hour train cap** (seed 42, at-or-under). Validation/test are
  not hour-capped.
- **Shared canonical text normalization** (`canonical_text`), applied to training
  labels **and** evaluation scoring in both pipelines:
  - NFC normalization, curly quotes → straight apostrophes, lowercase;
  - diacritics folded to ASCII base letters (`é→e`, `ñ→n`);
  - keep only letters, digits, space, apostrophe (word-internal in Luhya);
  - all other punctuation dropped; whitespace collapsed.

### 3.2 Aligned hyperparameters (identical across pipelines)

| Hyperparameter | Value |
|---|---|
| Effective batch | 32 (4 × 8, or 4 × 4 × 2 GPU) |
| Epochs | 5 |
| Learning rate | 1e-5 |
| Warmup | 500 absolute steps |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Gradient checkpointing | on |
| Seed | 42 |
| Audio window | 2–30 s |
| Train-hour cap | 40 h |

Precision is the only training knob that varies, by hardware:
- **fp16** on NVIDIA T4 (Kaggle Whisper run; Colab W2V2 run);
- **bf16** on a bf16-capable GPU (local W2V2 run).

### 3.3 Equivalent evaluation

Both pipelines now evaluate on the **same** validation and test splits, and both
compute corpus-level WER/CER with the **same normalized references**:

- Whisper: greedy generation, then `jiwer` on canonical references/predictions.
- CTC: CTC argmax decoding, then WER/CER on the same canonical references
  (the CTC vocabulary exactly equals the canonical character set).

---

## 4. Bugs found and fixed during alignment

1. **Warmup silently zeroed (real bug).** The CTC trainer passed
   `warmup_steps=500, warmup_ratio=0.0`; in Transformers 5.x, a non-`None`
   `warmup_ratio` **overwrites** `warmup_steps`, so the intended 500-step warmup
   became 0. Fixed by passing `warmup_ratio=None`.
2. **`freeze_feature_encoder: true` was a silent no-op** for `facebook/w2v-bert-2.0`
   (the class has no such method). `src/models/factory.py` now logs an explicit
   warning instead of silently ignoring the flag.
3. **No test set for the CTC pipeline.** `train_model.py` now creates and evaluates
   on a held-out test split (skipped gracefully when `test_ratio: 0`).
4. **Punctuation/normalization mismatch.** Replaced per-pipeline normalization with
   one shared `canonical_text` policy used for labels and scoring in both pipelines.
5. **Dataset repo ID.** All configs now point to the currently listed Hub ID,
   `Digital-Divide-Data/Luhya-ASR-Data-subset-50h`; the notebook resolves and pins its
   immutable commit SHA before either run starts.

---

## 5. Verification performed

- **17 unit tests pass** (split policy, hour cap, canonical normalization,
  distributed-eval padding, CTC token-ID decoding, report mismatch gates,
  speaker-overlap rebuild, 80/20 fallback, and ratio guards).
- **End-to-end synthetic check** of the shared module: speaker-disjoint 80/10/10
  with no speaker overlap, 2 s minimum enforced, deterministic, hour cap respected.
- **API verification** against the installed Transformers 5.14.1 source (the
  pinned Kaggle version) and the official Trainer/Hub documentation:
  - `Seq2SeqTrainer(processing_class=...)` (the 5.x replacement for `tokenizer=`);
  - `WhisperForConditionalGeneration.from_pretrained(..., attn_implementation="sdpa")`;
  - `WhisperProcessor.from_pretrained(model, language=..., task=...)`;
  - `Wav2Vec2BertProcessor` = `SeamlessM4TFeatureExtractor` + `Wav2Vec2CTCTokenizer`;
  - `Wav2Vec2BertForCTC` with `add_adapter`, `ctc_zero_infinity`, `ctc_loss_reduction`;
  - `TrainingArguments` fields (`eval_strategy`, `train_sampling_strategy`,
    `length_column_name`, `warmup_steps` ≥ 1 = absolute steps, `save_strategy="best"`, ...).

---

## 6. Expected / likely results

> These are **predictions**, not measured results — the aligned runs have not been
> executed yet.

### 6.1 What the numbers should show

- **Comparability is now guaranteed**: any residual WER/CER difference is a real
  model/difference signal, not an artifact of data handling.
- **Whisper WER should drop versus its published card** (34.6 % test WER). That
  number was scored *with punctuation kept*; under the new canonical
  punctuation-free scoring, punctuation errors no longer count, typically lowering
  WER by several points.
- **Wav2Vec2-BERT should improve versus its old 43.5 % validation WER**, because
  it now trains on 40 h (previously 20 h) and on the same splits.

### 6.2 Which model is likely to win, and why

No strong prior — both are competitive at 40 h on a low-resource language:

- **Whisper-small advantages:** seq2seq with a language-conditioning token
  (`swahili` used as a documented Bantu proxy, since no `luy` token exists) and a
  stronger decoder prior from 680k h of supervised pretraining. Often more robust
  on short, noisy clips.
- **Wav2Vec2-BERT advantages:** 4.5M h of self-supervised features and more
  parameters; CTC is simple and stable. Often comparable or better CER.
- Expected pattern: **WER may favor Whisper** (lexical prior, better word
  boundaries), while **CER may be close or favor Wav2Vec2-BERT** (phonetic CTC
  alignment). Report both metrics.

### 6.3 Interpretation guidance for the write-up

- Report **validation and test WER/CER** side by side; test is the honest
  held-out number.
- Watch for the **validation/test gap** (Whisper's test was ~10 WER points better
  than validation) — a sign of uneven split difficulty; do not over-interpret the
  absolute numbers.
- Present **per-dialect and per-speaker** breakdowns before drawing conclusions.
- Note that decoding differs by model family (greedy seq2seq vs CTC argmax); this
  is a property of the model, not a confound we can remove.

---

## 7. Limitations and threats to validity

1. **Single seed (42)** — no variance estimate. At minimum, state this explicitly;
   ideally repeat the best model with 2–3 seeds.
2. **No significance tests / confidence intervals** on the reported numbers.
3. **Single dataset subset** (~40 h, one community's recordings). No claim of
   coverage of all Luhya varieties, ages, devices, or acoustic conditions.
4. **Swahili conditioning proxy** for Whisper may bias its decoding slightly
   toward Swahili-like orthography/lexicon.
5. **Punctuation stripped from labels and scoring** — the comparison says nothing
   about punctuation/casing quality.
6. **Different feature extractors and decoders** are inherent to the model
   families; a fully matched decoder (e.g., an external LM for CTC) would be a
   separate experiment.
7. **Hardware differs** (2×T4 vs 1×T4) — training *time* is not comparable, only
   the resulting quality.
8. **No external test benchmark** — results describe this split only.

---

## 8. Reproducibility

**Whisper (Kaggle, 2×T4, fp16):**
```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 \
  --num_cpu_threads_per_process 2 \
  scripts/train_whisper.py \
  --config config_files/ASR_train_config_whisper_small_kaggle.yaml \
  --resume_from_checkpoint latest
```

**Wav2Vec2-BERT (CTC):**
```bash
python3 scripts/train_model.py --config config_files/ASR_train_config_luhya.yaml
```

Config files (identical conditions):
- `config_files/ASR_train_config_whisper_small_kaggle.yaml`
- `config_files/ASR_train_config_luhya.yaml`
- `config_files/ASR_train_config_luhya_colab.yaml`

Both runs write a **split manifest** (counts, hours, speaker counts, fingerprints,
overlap audit) so you can prove the two runs used identical clips.

The root Kaggle notebook also resolves the dataset's immutable Hub SHA before
either model starts, generates `Luhya_ASR_Experiment_Report.md`, creates current
model cards, and keeps Hub uploads private by default pending license review.

---

## 9. Next steps / recommendations

1. Run both aligned pipelines; verify the split manifests match exactly.
2. Review the automatically generated per-dialect and per-speaker WER/CER from
   the paired `test_predictions.csv` artifacts.
3. Repeat the leading model with 2–3 seeds and report mean ± std.
4. Optionally add a CTC decoder (KenLM / beam search) to isolate the encoder
   quality from the decoding algorithm.
5. Consider a dedicated Luhya language token for Whisper once the dataset is
   published with consent and licensing documentation.

---

## Appendix A — Files changed

| File | Change |
|---|---|
| `src/data/splits.py` | **new** — shared split/filter/hour-cap/canonical-text logic |
| `scripts/train_whisper.py` | uses shared module; canonical eval & prediction export |
| `scripts/train_model.py` | raw-audio CTC batching; smoke/resume/recovery; standardized artifacts |
| `scripts/build_comparison_report.py` | strict manifest/pairing gate, report, subgroup metrics, model cards |
| `src/data/dataset.py` | `load_datasets` uses shared 80/10/10 splits; returns test split |
| `src/utils/config.py` | new fields: `min_audio_length`, `max_train_hours`, ratios, `speaker_column`, `weight_decay`, `max_grad_norm`, ... |
| `src/training/trainer.py` | `weight_decay`, `max_grad_norm`, warmup-overwrite fix |
| `src/models/factory.py` | explicit warning when `freeze_feature_encoder` unsupported |
| `config_files/*.yaml` | aligned dataset, splits, window, lr, warmup, precision |
| `luhya-asr-whisper-small.ipynb` | orchestrates both Kaggle runs and bounded HF/GitHub publishing |
| `tests/*.py` | 17 tests covering splits, raw collation, decoding, DDP trimming, and report gates |

## Appendix B — Aligned configuration summary

| Parameter | All aligned configs |
|---|---|
| dataset_path | `Digital-Divide-Data/Luhya-ASR-Data-subset-50h` |
| validation_ratio / test_ratio | 0.10 / 0.10 |
| min / max audio | 2.0 / 30.0 s |
| max_train_hours | 40.0 |
| learning_rate | 1e-5 |
| warmup_steps | 500 |
| weight_decay | 0.01 |
| max_grad_norm | 1.0 |
| epochs | 5 |
| effective batch | 32 |
| text policy | `canonical_text` (letters, digits, space, apostrophe only) |
