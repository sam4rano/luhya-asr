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
    - name: Test WER
      type: wer
      value: 43.4481
    - name: Test CER
      type: cer
      value: 11.3584
---

# luhya-asr-w2v-BERT

This model is a fine-tuned version of a w2v-BERT model for Automatic Speech Recognition in Luhya.

## Training results

| Training Loss | Epoch | Step | Validation Loss | Wer | Cer |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [More info] | 2.0 | [More info] | 0.4551 | 43.4481 | 11.3584 |

### Framework versions
- Transformers 5.13.1
- Pytorch 2.11.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2
