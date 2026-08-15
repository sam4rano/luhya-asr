---
language:
- luy
license: apache-2.0
tags:
- asr
- automatic-speech-recognition
- luhya
- w2v-bert
metrics:
- wer
- cer
model-index:
- name: luhya-asr-w2v-BERT
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    metrics:
    - name: Validation WER
      type: wer
      value: 43.45
    - name: Validation CER
      type: cer
      value: 11.36
---

# luhya-asr-w2v-BERT

Fine-tuned Facebook Wav2Vec2-BERT 2.0 (580M parameters) for Automatic Speech Recognition in Luhya, a Bantu language spoken in Kenya.

Model on HuggingFace Hub: [Sam4rano/luhya-asr-w2v-BERT](https://huggingface.co/Sam4rano/luhya-asr-w2v-BERT)

## Dataset

| | Train | Validation | Test |
|---|---|---|---|
| **Source** | `DDD-Kenya/Luhya-ASR-Data-subset-50h` (HF Hub) | same | N/A |
| **Samples** | 6,947 | 400 | — |
| **Duration** | 20.00 hours | ~1.1 hours | — |
| **Notes** | Randomly subsampled from 23,198 samples (66.52 h) with 20-hour cap; 1 corrupt sample removed | Used as-is; no hour cap | No independent test split |

## Model

| Detail | Value |
|---|---|
| **Base model** | `facebook/w2v-bert-2.0` |
| **Parameters** | 580M (24 Conformer layers, 1024 hidden size, 16 attention heads) |
| **Feature extractor** | SeamlessM4T (80-dim mel spectrograms at 16 kHz) |
| **Decoding** | CTC (Connectionist Temporal Classification) |
| **Adapter** | TDNN final layer (kernel=3, stride=2) |
| **Feature encoder** | Frozen (pretrained weights preserved) |
| **Vocabulary** | 48 tokens (a-z, 0-9, basic punctuation, word delimiter) |

## Training Setup

| Detail | Value |
|---|---|
| **Batch size** | 4 × gradient accumulation 8 = effective batch 32 |
| **Learning rate** | 5e-5 with 10% warmup |
| **Epochs** | 2.0 (early stop at 18,400 steps; 25 planned) |
| **Precision** | bf16 mixed precision |
| **Gradient checkpointing** | Enabled |
| **Hardware** | NVIDIA T4 (16 GB VRAM), Google Colab |
| **Training time** | ~2.5 hours wall-clock |

## Text Preprocessing

- 16 kHz mono audio
- NFC Unicode normalization + accent mapping
- Lowercasing, whitespace normalisation
- Character set: `a-z`, digits `0-9`, punctuation `!'*,-.:;?`

## Results (Validation Set)

| Metric | Value |
|---|---|
| **Word Error Rate (WER)** | 43.45% |
| **Character Error Rate (CER)** | 11.36% |
| **Eval Loss (CTC)** | 0.4551 |
| **Composite Score** | 72.60/100 |

*Composite Score = (1 - 0.5×WER - 0.5×CER) × 100*

## Framework Versions

| Package | Version |
|---|---|
| Transformers | 5.13.1 |
| PyTorch | 2.11.0+cu128 |
| Datasets | 4.0.0 |
| Tokenizers | 0.22.2 |

## Future Work

- Acquire more labelled Luhya speech (50h+ target)
- Add KenLM / n-gram LM for beam-search decoding
- Data augmentation (SpecAugment, speed perturbation, noise injection)
- Hyperparameter sweep
- Cross-validation folds, speaker-independent test sets
- ONNX export / TorchScript for real-time inference
- Multilingual: add related Bantu languages

## Whisper Small on Kaggle T4 x2

The repository also contains a separate Kaggle-native sequence-to-sequence
pipeline for fine-tuning `openai/whisper-small` on
`Digital-Divide-Data/Luhya-ASR-Data-subset-50h`, with the training split
deterministically limited to at most 40 hours:

During the duration pass, every clip is decoded once on CPU. Corrupt or empty
audio rows are removed before training and their counts are recorded in the
split manifest, preventing a bad file from wasting a multi-hour GPU run.

- Notebook: `notebooks/train_luhya_whisper_kaggle.ipynb`
- Trainer: `scripts/train_whisper.py`
- Configuration: `config_files/ASR_train_config_whisper_small_kaggle.yaml`
- Model card: `model_cards/whisper-small-luhya/README.md`
- Kaggle dependencies: `requirements-kaggle.txt`

The Whisper path is intentionally separate from the CTC trainer. It computes
log-Mel features in the data loader instead of storing a large encoded dataset,
uses both T4 GPUs through DDP, retains only one resumable checkpoint, applies
early stopping, and writes validation/test WER and CER without W&B.

The data loader keeps existing validation and test splits when they are both
present and speaker-disjoint. If either split is missing, or `user_id` overlap
is detected, it deterministically rebuilds 80/10/10 speaker-disjoint splits.
The chosen policy, counts, hours, fingerprints, and overlap checks are recorded
in `evaluation_summary.json`.

Whisper does not provide an official Luhya (`luy`) language token. This baseline
uses the Swahili token only as a documented conditioning proxy; training labels
and decoded outputs remain Luhya. Do not describe the resulting model as a
Swahili ASR model.

### Kaggle run

Create a Kaggle notebook with Internet enabled, choose **GPU T4 x2**, and add an
`HF_TOKEN` secret that has access to the dataset. The included notebook first
runs a two-step smoke test, then launches the full run with:

```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 \
  --num_cpu_threads_per_process 2 \
  scripts/train_whisper.py \
  --config config_files/ASR_train_config_whisper_small_kaggle.yaml \
  --resume_from_checkpoint latest
```

The optimizer global batch is `4 per GPU x 2 GPUs x 4 accumulation = 32`.
Final artifacts are written under
`/kaggle/working/luhya-asr-output/whisper-small-luhya/`:

- `evaluation_summary.json`: configuration, split audit, and all metrics
- `metrics_summary.csv`: presentation-ready validation/test WER and CER
- `test_predictions.csv`: reference/prediction pairs for error analysis
- `validation_results.json`, `test_results.json`, `train_results.json`
- `final-model/`: the selected best checkpoint and processor

If training and metric computation finish but artifact export is interrupted,
regenerate validation/test artifacts from the saved model without retraining:

```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 \
  --num_cpu_threads_per_process 2 scripts/train_whisper.py \
  --config config_files/ASR_train_config_whisper_small_kaggle.yaml \
  --evaluation_only
```
