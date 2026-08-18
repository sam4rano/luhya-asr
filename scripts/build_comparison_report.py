#!/usr/bin/env python3
"""Validate two completed runs and build shareable comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jiwer


MODEL_LABELS = {
    "whisper": "Whisper Small",
    "wav2vec2_bert": "Wav2Vec2-BERT 2.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whisper-dir", type=Path, required=True)
    parser.add_argument("--wav2vec-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_predictions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def exact_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    references = [row["reference"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    return {
        "wer": float(jiwer.wer(references, predictions)),
        "cer": float(jiwer.cer(references, predictions)),
    }


def validate_reported_test_metrics(
    label: str,
    summary: dict[str, Any],
    exact: dict[str, float],
) -> None:
    metrics = summary["test_metrics"]
    for metric in ("wer", "cer"):
        reported = float(metrics[f"test_{metric}"])
        if abs(reported - exact[metric]) > 1e-6:
            raise ValueError(
                f"{label} {metric.upper()} mismatch: summary={reported:.10f}, "
                f"predictions={exact[metric]:.10f}. Regenerate evaluation artifacts."
            )


def validate_manifests(
    whisper: dict[str, Any],
    wav2vec: dict[str, Any],
) -> dict[str, Any]:
    left = whisper["split_manifest"]
    right = wav2vec["split_manifest"]
    if left != right:
        differing = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        raise ValueError(
            "The runs are not directly comparable because their split manifests differ "
            f"in: {', '.join(differing)}"
        )
    if any(int(value) for value in left["speaker_overlap_counts"].values()):
        raise ValueError("Speaker overlap is non-zero; comparison report generation stopped")
    return left


def validate_and_pair(
    whisper_rows: list[dict[str, str]],
    wav2vec_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if len(whisper_rows) != len(wav2vec_rows):
        raise ValueError(
            "Prediction row counts differ: "
            f"Whisper={len(whisper_rows)}, Wav2Vec2-BERT={len(wav2vec_rows)}"
        )

    paired = []
    ignored = {"prediction"}
    metadata = sorted((set(whisper_rows[0]) & set(wav2vec_rows[0])) - ignored)
    for index, (whisper, wav2vec) in enumerate(zip(whisper_rows, wav2vec_rows)):
        for key in ("id", "reference"):
            if whisper[key] != wav2vec[key]:
                raise ValueError(f"Test row {index} differs in {key}; runs are not paired")
        if abs(float(whisper["audio_duration"]) - float(wav2vec["audio_duration"])) > 1e-6:
            raise ValueError(f"Test row {index} differs in audio duration")
        for key in metadata:
            if key not in {"audio_duration", "id", "reference"} and whisper[key] != wav2vec[key]:
                raise ValueError(f"Test row {index} differs in metadata column {key}")

        reference = whisper["reference"]
        row: dict[str, Any] = {
            "id": whisper["id"],
            "reference": reference,
            "whisper_prediction": whisper["prediction"],
            "wav2vec2_bert_prediction": wav2vec["prediction"],
            "audio_duration": whisper["audio_duration"],
            "whisper_wer": jiwer.wer(reference, whisper["prediction"]),
            "whisper_cer": jiwer.cer(reference, whisper["prediction"]),
            "wav2vec2_bert_wer": jiwer.wer(reference, wav2vec["prediction"]),
            "wav2vec2_bert_cer": jiwer.cer(reference, wav2vec["prediction"]),
        }
        for key in metadata:
            if key not in {"audio_duration", "id", "reference"}:
                row[key] = whisper[key]
        paired.append(row)
    return paired


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subgroup_rows(
    paired: list[dict[str, Any]],
    group_column: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[str(row[group_column])].append(row)

    output = []
    for group, rows in sorted(groups.items()):
        references = [str(row["reference"]) for row in rows]
        whisper_predictions = [str(row["whisper_prediction"]) for row in rows]
        wav2vec_predictions = [str(row["wav2vec2_bert_prediction"]) for row in rows]
        output.append(
            {
                group_column: group,
                "samples": len(rows),
                "hours": round(
                    sum(float(row["audio_duration"]) for row in rows) / 3600.0,
                    4,
                ),
                "whisper_wer_percent": 100 * jiwer.wer(references, whisper_predictions),
                "whisper_cer_percent": 100 * jiwer.cer(references, whisper_predictions),
                "wav2vec2_bert_wer_percent": 100
                * jiwer.wer(references, wav2vec_predictions),
                "wav2vec2_bert_cer_percent": 100
                * jiwer.cer(references, wav2vec_predictions),
            }
        )
    return output


def metric_value(summary: dict[str, Any], split: str, metric: str) -> float:
    return 100 * float(summary[f"{split}_metrics"][f"{split}_{metric}"])


def model_card(
    label: str,
    summary: dict[str, Any],
    exact: dict[str, float],
) -> str:
    manifest = summary["split_manifest"]
    config = summary.get("config", {})
    base_model = summary["model_name"]
    is_whisper = label == "whisper"
    inference_class = (
        "WhisperForConditionalGeneration" if is_whisper else "AutoModelForCTC"
    )
    processor_class = "WhisperProcessor" if is_whisper else "AutoProcessor"
    limitation = (
        "Whisper has no official Luhya language token; Swahili is used only as a "
        "documented conditioning proxy."
        if is_whisper
        else "This is Wav2Vec2-BERT 2.0 with greedy CTC decoding and no external language model."
    )
    return f"""---
language:
- luy
library_name: transformers
base_model: {base_model}
datasets:
- {manifest['dataset_path']}
tags:
- automatic-speech-recognition
- luhya
- {'whisper' if is_whisper else 'wav2vec2-bert'}
metrics:
- wer
- cer
---

# {MODEL_LABELS[label]} fine-tuned for Luhya ASR

This model is a research baseline trained and evaluated under the repository's
shared speaker-disjoint comparison protocol. It is not validated for high-stakes
or production transcription.

## Held-out results

| Split | WER | CER |
|---|---:|---:|
| Validation | {metric_value(summary, 'validation', 'wer'):.2f}% | {metric_value(summary, 'validation', 'cer'):.2f}% |
| Test | {100 * exact['wer']:.2f}% | {100 * exact['cer']:.2f}% |

The exact test values above were recomputed from the exported prediction rows.

## Data and protocol

- Dataset: `{manifest['dataset_path']}` at revision `{manifest['dataset_revision']}`
- Code revision: `{summary.get('code_revision') or 'not recorded'}`
- Split policy: `{manifest['split_policy']}`, seed {manifest['seed']}
- Train/validation/test samples: {manifest['splits']['train']['samples']} / {manifest['splits']['validation']['samples']} / {manifest['splits']['test']['samples']}
- Training audio cap: {manifest['max_train_hours']} hours; accepted clip window: {config.get('min_audio_length', 2.0)}–{config.get('max_audio_length', 30.0)} seconds
- Text scoring: lowercase letters, digits, spaces, and apostrophes; punctuation and diacritics are normalized consistently across both models
- Decoding: {'greedy sequence generation (beam 1)' if is_whisper else 'greedy CTC argmax (no language model)'}

## Limitations

- {limitation}
- Results are from one seed and one held-out split; no confidence interval is claimed.
- The dataset may not cover all Luhya varieties, speakers, devices, or acoustic conditions.
- Punctuation and casing quality are not evaluated.
- Confirm dataset and weight redistribution terms before making the repository public.

## Inference

```python
from transformers import {processor_class}, {inference_class}

model_id = "YOUR_NAMESPACE/YOUR_MODEL_REPO"
processor = {processor_class}.from_pretrained(model_id)
model = {inference_class}.from_pretrained(model_id)
```

See `evaluation_summary.json` and `metrics_summary.csv` in this repository for
the full reproducibility record.
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        "whisper": args.whisper_dir,
        "wav2vec2_bert": args.wav2vec_dir,
    }
    summaries = {
        label: load_json(path / "evaluation_summary.json")
        for label, path in run_dirs.items()
    }
    prediction_rows = {
        label: load_predictions(path / "test_predictions.csv")
        for label, path in run_dirs.items()
    }
    exact = {label: exact_metrics(rows) for label, rows in prediction_rows.items()}
    for label in run_dirs:
        validate_reported_test_metrics(label, summaries[label], exact[label])

    manifest = validate_manifests(summaries["whisper"], summaries["wav2vec2_bert"])
    paired = validate_and_pair(
        prediction_rows["whisper"],
        prediction_rows["wav2vec2_bert"],
    )
    write_csv(args.output_dir / "paired_test_predictions.csv", paired)

    summary_rows = []
    for label, summary in summaries.items():
        for split in ("validation", "test"):
            summary_rows.append(
                {
                    "model": MODEL_LABELS[label],
                    "split": split,
                    "wer_percent": metric_value(summary, split, "wer"),
                    "cer_percent": metric_value(summary, split, "cer"),
                    "loss": summary[f"{split}_metrics"].get(f"{split}_loss", ""),
                }
            )
    write_csv(args.output_dir / "comparison_summary.csv", summary_rows)

    dialect_column = next(
        (column for column in ("dialect", "language") if column in paired[0]),
        None,
    )
    if dialect_column:
        write_csv(
            args.output_dir / "dialect_metrics.csv",
            subgroup_rows(paired, dialect_column),
        )
    speaker_column = manifest.get("speaker_column")
    if speaker_column and speaker_column in paired[0]:
        write_csv(
            args.output_dir / "speaker_metrics.csv",
            subgroup_rows(paired, speaker_column),
        )

    test_wers = {label: 100 * values["wer"] for label, values in exact.items()}
    test_cers = {label: 100 * values["cer"] for label, values in exact.items()}
    wer_winner = min(test_wers, key=test_wers.get)
    cer_winner = min(test_cers, key=test_cers.get)
    generated_at = datetime.now(timezone.utc).isoformat()

    metrics_table = "\n".join(
        f"| {row['model']} | {row['split'].title()} | {row['wer_percent']:.2f}% | {row['cer_percent']:.2f}% |"
        for row in summary_rows
    )
    split_table = "\n".join(
        f"| {name.title()} | {values['samples']} | {values['hours']:.4f} | {values['speakers']} | `{values['fingerprint']}` |"
        for name, values in manifest["splits"].items()
    )
    report = f"""# Luhya ASR experiment report

Generated: {generated_at}

## Result

Both completed runs passed the comparison gate: their full split manifests,
test row order, normalized references, durations, and available metadata match.

- Lower held-out test WER: **{MODEL_LABELS[wer_winner]}** ({test_wers[wer_winner]:.2f}%)
- Lower held-out test CER: **{MODEL_LABELS[cer_winner]}** ({test_cers[cer_winner]:.2f}%)

## Metrics

| Model | Split | WER | CER |
|---|---|---:|---:|
{metrics_table}

Exact test metrics were recomputed from each exported prediction CSV and agree
with the trainer summaries.

## Shared evaluation protocol

- Dataset: `{manifest['dataset_path']}`
- Immutable dataset revision: `{manifest['dataset_revision']}`
- Split policy: `{manifest['split_policy']}` with seed {manifest['seed']}
- Speaker overlap counts: `{json.dumps(manifest['speaker_overlap_counts'], sort_keys=True)}`
- Training audio cap: {manifest['max_train_hours']} hours

| Split | Samples | Hours | Speakers | Fingerprint |
|---|---:|---:|---:|---|
{split_table}

## Reproducibility

| Model | Base model | Code revision | Global batch | Transformers |
|---|---|---|---:|---|
| Whisper Small | `{summaries['whisper']['model_name']}` | `{summaries['whisper'].get('code_revision') or 'not recorded'}` | {summaries['whisper']['optimizer_global_batch_size']} | {summaries['whisper'].get('runtime_versions', {}).get('transformers', 'unknown')} |
| Wav2Vec2-BERT 2.0 | `{summaries['wav2vec2_bert']['model_name']}` | `{summaries['wav2vec2_bert'].get('code_revision') or 'not recorded'}` | {summaries['wav2vec2_bert']['optimizer_global_batch_size']} | {summaries['wav2vec2_bert'].get('runtime_versions', {}).get('transformers', 'unknown')} |

## Interpretation boundaries

- “Wav2vec” in the notebook refers specifically to Wav2Vec2-BERT 2.0, not the smaller classic Wav2Vec2 model.
- Data selection and scoring are matched. Equal learning rates and epochs do not prove each architecture was independently optimized; a model-specific tuning study would be a separate experiment.
- This is a single-seed comparison without confidence intervals or an external benchmark.
- Whisper uses a Swahili conditioning token as a proxy because there is no official Luhya token.
- Whisper uses greedy sequence generation; Wav2Vec2-BERT uses greedy CTC argmax without a language model.
- Punctuation/casing are excluded from training labels and scoring.
- Confirm the source dataset's consent, licensing, and redistribution terms before making model weights public.

## Included artifacts

- `comparison_summary.csv`: compact validation/test metrics
- `paired_test_predictions.csv`: aligned utterance-level references and predictions
- `dialect_metrics.csv`: per-dialect results when dialect metadata exists
- `speaker_metrics.csv`: per-speaker results when speaker metadata exists
- `comparison_manifest.json`: machine-readable summaries and exact recomputed metrics
"""
    (args.output_dir / "Luhya_ASR_Experiment_Report.md").write_text(
        report,
        encoding="utf-8",
    )
    with (args.output_dir / "comparison_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": generated_at,
                "comparison_gate_passed": True,
                "shared_split_manifest": manifest,
                "exact_test_metrics": exact,
                "run_summaries": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    for label, run_dir in run_dirs.items():
        final_model_dir = run_dir / "final-model"
        if not final_model_dir.is_dir():
            raise FileNotFoundError(f"Missing final model directory: {final_model_dir}")
        (final_model_dir / "README.md").write_text(
            model_card(label, summaries[label], exact[label]),
            encoding="utf-8",
        )
        for filename in ("evaluation_summary.json", "metrics_summary.csv"):
            shutil.copy2(run_dir / filename, final_model_dir / filename)

    print(f"Comparison gate passed; report written to {args.output_dir}")


if __name__ == "__main__":
    main()
