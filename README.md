# Luhya ASR -- Fine-tune Wav2Vec2-BERT on Luhya speech

Fine-tunes [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) (Wav2Vec2-BERT 2.0, 580M params) on ~20 hours of the [Luhya ASR dataset](https://huggingface.co/datasets/DDD-Kenya/Luhya-ASR-Data-subset-50h). Follows the official [HuggingFace fine-tuning guide](https://huggingface.co/blog/fine-tune-w2v2-bert).

## Quickstart

### Local / server with GPU

```bash
pip install transformers datasets evaluate wandb jiwer librosa accelerate soundfile
python3 scripts/train_model.py --config config_files/ASR_train_config_luhya.yaml
```

### Google Colab / Lightning AI

Click the badge below or copy-paste into a notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sam4rano/luhya-asr/blob/main/notebooks/train_luhya_asr.ipynb)

**Or in a notebook cell:**

```python
!git clone https://github.com/sam4rano/luhya-asr.git
%cd luhya-asr
!pip install transformers datasets evaluate wandb jiwer librosa accelerate soundfile
!python3 scripts/train_model.py --config config_files/ASR_train_config_luhya.yaml
```

## Dataset

[DDD-Kenya/Luhya-ASR-Data-subset-50h](https://huggingface.co/datasets/DDD-Kenya/Luhya-ASR-Data-subset-50h) -- ~50 hours of Luhya speech (multiple dialects: Wanga, Kabarasi, Kisa, Banyala, Bukusu, etc.). Training restricted to 20 hours via `max_data_hours: 20.0` in the config.

## Configuration

Edit `config_files/ASR_train_config_luhya.yaml` to adjust:

| Setting | Default | Notes |
|---|---|---|
| `pretrained_model` | `facebook/w2v-bert-2.0` | 580M params; also try `facebook/mms-300m` |
| `max_data_hours` | `20.0` | Set to `0` to use all 50h |
| `batch_size` | `4` | Lower to `1` on T4 if OOM |
| `learning_rate` | `5e-5` | Use `5e-4` for MMS models |
| `add_final_layer_adapter` | `true` | Parameter-efficient; only adapter + CTC head trained |

## Environment variables

Create a `.env` file in the project root (optional):

```env
WANDB_API_KEY="your_wandb_key"
HF_API_KEY="your_hub_key"
```

Training runs without these but won't log to W&B or push to HF Hub.
