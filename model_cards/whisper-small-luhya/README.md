---
language:
- luy
base_model: openai/whisper-small
base_model_relation: finetune
library_name: transformers
pipeline_tag: automatic-speech-recognition
datasets:
- Digital-Divide-Data/Luhya-ASR-Data-subset-50h
metrics:
- wer
- cer
tags:
- asr
- automatic-speech-recognition
- whisper
- luhya
- low-resource-asr
- pytorch
model-index:
- name: Luhya Whisper Small (40-hour training cap)
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Luhya ASR 50h, deterministic speaker-disjoint test split (seed 42)
      type: Digital-Divide-Data/Luhya-ASR-Data-subset-50h
    metrics:
    - name: Test WER
      type: wer
      value: 34.6
    - name: Test CER
      type: cer
      value: 7.1
---

# Luhya Whisper Small ASR

This is a research baseline produced by fully fine-tuning
[`openai/whisper-small`](https://huggingface.co/openai/whisper-small) for Luhya
automatic speech recognition. It was trained on at most 40 hours selected from
`Digital-Divide-Data/Luhya-ASR-Data-subset-50h` and evaluated on deterministic
speaker-disjoint validation and test splits.

**Status:** research and prototyping only. The model has not been validated for
production, safety-critical use, all Luhya varieties, or arbitrary recording
conditions.

## Model details

| Field | Value |
|---|---|
| Base model | `openai/whisper-small` |
| Architecture | Whisper encoder-decoder Transformer, approximately 244M parameters |
| Task | Luhya speech-to-text transcription |
| Audio | 16 kHz; training clips filtered to 0.2-30 seconds |
| Fine-tuning | Full-model fine-tuning; encoder not frozen |
| Decoding | Greedy generation (`num_beams=1`), maximum 225 tokens |
| Conditioning | Whisper `swahili` language token used as a proxy |
| Training code | `scripts/train_whisper.py` |
| Configuration | `config_files/ASR_train_config_whisper_small_kaggle.yaml` |

Whisper does not provide an official Luhya (`luy`) language token. The Swahili
token is used only to condition multilingual Whisper decoding. Training targets
and intended outputs are Luhya; this is not a Swahili ASR model.

## Evaluation results

The final evaluation-only recovery run completed on 15 August 2026.

| Split | WER | CER | Seq2seq loss |
|---|---:|---:|---:|
| Validation | 45.24% | 11.15% | 0.6213 |
| Test | 34.63% | 7.07% | 0.5158 |

Lower is better for all three reported measures. WER and CER are corpus-level
metrics computed with `jiwer` after applying the same text normalization to
references and predictions.

### Interpretation

- A test WER of 34.63% means approximately 35 word edits per 100 normalized
  reference words. It does not mean that 65.37% of utterances are perfect.
- The substantially lower CER (7.07%) indicates that many predictions are
  character-wise close even when a word-level insertion, deletion, substitution,
  or boundary error increases WER.
- Test WER is 10.61 percentage points lower than validation WER, and test CER is
  4.08 points lower than validation CER. This is a material distribution gap.
  The test speakers, dialect mix, clip difficulty, or recording quality may be
  easier than validation. Per-dialect and per-speaker analysis is required
  before making broader performance claims.
- No confidence intervals, repeated seeds, external benchmark, or significance
  tests were computed. These figures describe this split and run only.
- Results from the repository's Wav2Vec2-BERT experiment are not directly
  comparable because that experiment used a different training-hour cap and
  evaluation protocol and did not have an independent test split.

### Distributed-evaluation note

Two-GPU evaluation produced 2,688 gathered prediction rows for 2,685 real test
examples. Three distributed-sampler padding rows were removed when writing
`test_predictions.csv`. The structured metadata rounds the reported result to
one decimal place. Before using the score in a paper or leaderboard, recompute
WER/CER from the 2,685-row prediction CSV so the published number is explicitly
based only on unpadded examples.

The warnings about duplicate Whisper suppression processors and ignored BPE
`clean_up_tokenization_spaces=True` did not stop evaluation. Transformers kept
the custom suppression processors and correctly avoided destructive WordPiece
cleanup for the Whisper BPE tokenizer.

## Evaluation methodology

1. Existing dataset splits were checked for `user_id` overlap.
2. Because speaker overlap was detected, all available rows were recombined and
   deterministically split 80/10/10 by speaker using seed 42.
3. Audio was decoded at 16 kHz. Corrupt/empty audio, blank transcripts, and clips
   outside 0.2-30 seconds were filtered before training.
4. The training split was deterministically limited to at most 40 hours. The
   validation and test splits were not included in this hour cap.
5. Checkpoints were evaluated every 500 optimizer steps. The selected model was
   the checkpoint with the lowest validation WER.
6. Final validation and test inference used greedy generation.
7. References and hypotheses were NFC-normalized, curly apostrophes were mapped
   to straight apostrophes, whitespace was collapsed, and text was lowercased.

The run's `evaluation_summary.json` is the authoritative record for exact split
counts, hours, speaker counts, fingerprints, overlap audit, filtering counts,
training metrics, and package-generated results.

## Training data

The configured dataset identifier was
`Digital-Divide-Data/Luhya-ASR-Data-subset-50h`. The current public Hub listing
may appear under `DDD-Kenya/Luhya-ASR-Data-subset-50h`. The dataset includes
audio, transcript, speaker (`user_id`), and dialect fields. The model card does
not claim that every Luhya variety, age group, gender, geography, device, or
acoustic environment is adequately represented.

The dataset's licensing, consent, collection, demographic, and annotation
documentation was not independently audited as part of this training run.
Users must verify the dataset's current terms and suitability before
redistributing weights or deploying the model.

## Training procedure

| Hyperparameter | Value |
|---|---:|
| Epochs | 5 |
| Optimizer steps | 2,210 |
| Per-device batch | 4 |
| GPUs | 2 x NVIDIA T4 |
| Gradient accumulation | 4 |
| Global optimizer batch | 32 |
| Learning rate | 1e-5 |
| Weight decay | 0.01 |
| Warmup | 5% of optimizer steps |
| Gradient clipping | 1.0 |
| Precision | FP16 |
| Gradient checkpointing | Enabled |
| Optimizer | Fused AdamW |
| Observed training time | Approximately 2 h 47 min, excluding preprocessing/final evaluation |
| Seed | 42 |

Training used Hugging Face Transformers/Accelerate DDP on Kaggle T4 x2. W&B was
disabled. Log-Mel features were computed on demand rather than materialized as a
large fixed feature dataset.

## Intended use

Suitable uses include:

- research on low-resource Luhya ASR;
- prototype transcription with human review;
- qualitative error analysis and comparison under the same split protocol;
- initialization for further dialect/domain-specific adaptation.

## Out-of-scope and unsafe use

Do not use this model as the sole basis for:

- medical, legal, financial, emergency, or other high-stakes decisions;
- surveillance, speaker identification, or inferring sensitive attributes;
- grading, employment, eligibility, or law-enforcement decisions;
- claiming coverage of every Luhya dialect or unseen acoustic domain;
- verbatim publication of sensitive recordings without consent and review.

The model transcribes speech; it does not verify factual accuracy, speaker
identity, consent, or the meaning and context of an utterance.

## Limitations and risks

- Only one random seed and one dataset were evaluated.
- No per-dialect, per-speaker, gender, age, device, noise, or code-switching
  breakdown is currently reported.
- The validation-test gap suggests uneven split difficulty or coverage.
- Short-clip training does not establish long-form transcription quality.
- Whisper can omit speech, repeat text, or hallucinate plausible text,
  especially for silence, noise, music, unfamiliar dialects, and domain shift.
- Lowercased normalized scoring does not measure casing or formatting quality.
- Names, rare words, numbers, borrowed words, and code-switched speech may have
  higher error rates than the aggregate metrics suggest.
- The Swahili proxy token may bias decoding toward Swahili-like lexical or
  orthographic patterns.

Human review and confidence-aware product design are required for real-world
use. A deployment evaluation should include silence/noise tests, hallucination
rates, long-form audio, latency, representative dialects, and subgroup error
analysis.

## Usage

Replace the model ID below if the weights are published under a different Hub
repository.

```python
import torch
from transformers import pipeline

model_id = "Sam4rano/Luhya-ASR-Data-subset-50h-whisper-small-finetuned"
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device=device,
    torch_dtype=dtype,
)

result = transcriber(
    "sample.wav",
    generate_kwargs={"language": "swahili", "task": "transcribe"},
)
print(result["text"])
```

Use 16 kHz mono audio where possible. Treat audio longer than 30 seconds as an
unvalidated use case unless it is chunked and evaluated separately.

## Reproducibility artifacts

- `evaluation_summary.json`: split audit, configuration context, and metrics
- `metrics_summary.csv`: validation/test WER, CER, and loss
- `test_predictions.csv`: 2,685 references and hypotheses for error analysis
- `validation_results.json`, `test_results.json`, `train_results.json`
- `final-model/`: selected model and processor

## Environmental impact

The observed full fine-tuning used two NVIDIA T4 GPUs for approximately 2 hours
47 minutes, followed by validation/test generation. Energy use and carbon
emissions were not measured, so no emissions estimate is claimed.

## Licensing

The `openai/whisper-small` base model is distributed under Apache-2.0. This card
does not assign a license to the fine-tuned weights because the training
dataset's current license and redistribution terms were not verified. Confirm
both the base-model and dataset terms before publishing or redistributing the
fine-tuned checkpoint.

## Citation

If you use this model, cite the Whisper paper, the dataset provider, and the
specific model repository/version used. Also report the split seed and whether
metrics were recomputed from the unpadded prediction artifact.

```bibtex
@article{radford2022robust,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and
          McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```
