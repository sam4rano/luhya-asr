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
