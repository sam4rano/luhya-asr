#!/usr/bin/env python3
"""Fine-tune multilingual Whisper for Luhya ASR on Kaggle or locally.

The input dataset stays as raw audio. Log-Mel features are computed in the data
loader instead of being materialized to disk, which avoids a very large cached
feature dataset for Whisper's fixed 30-second input window.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import jiwer
import numpy as np
import torch
import yaml
from accelerate import PartialState
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

LOGGER = logging.getLogger("luhya-whisper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint path, or 'latest' to use the newest checkpoint in output_dir",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run two training steps on small deterministic subsets before the full run",
    )
    parser.add_argument(
        "--evaluation_only",
        action="store_true",
        help="Load the saved final model (or latest checkpoint) and regenerate evaluations",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def setup_logging(is_main_process: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if is_main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize_text(text: Any, lowercase: bool = True) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(text))
    normalized = normalized.replace("’", "'").replace("ʼ", "'")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.lower() if lowercase else normalized


def _audio_samples(audio: Any) -> tuple[np.ndarray, int]:
    """Decode both datasets<=4 dictionary audio and datasets>=5 AudioDecoder."""
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().float().numpy()
        array = np.asarray(data, dtype=np.float32)
        sample_rate = int(samples.sample_rate)
    elif isinstance(audio, dict) and "array" in audio:
        array = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio["sampling_rate"])
    else:
        raise TypeError(f"Unsupported decoded audio value: {type(audio)!r}")

    if array.ndim == 2:
        array = array.mean(axis=0)
    return np.ascontiguousarray(array.squeeze()), sample_rate


def _duration_from_audio(audio: Any) -> dict[str, float | bool]:
    """Decode once so corrupt/empty clips are removed before GPU training."""
    try:
        array, sample_rate = _audio_samples(audio)
    except Exception:
        # A single broken file must not abort a multi-hour training job. The
        # validity flag is filtered below before the data reaches the collator.
        return {"audio_duration": 0.0, "audio_is_valid": False}

    is_valid = sample_rate > 0 and array.size > 0
    duration = float(array.size / sample_rate) if is_valid else 0.0
    return {"audio_duration": duration, "audio_is_valid": is_valid}


def _normalize_text_row(text: Any, lowercase: bool) -> dict[str, str]:
    return {"normalized_text": normalize_text(text, lowercase=lowercase)}


def detect_column(dataset: Dataset, requested: str, fallbacks: Iterable[str]) -> str:
    candidates = [requested, *fallbacks]
    for candidate in candidates:
        if candidate and candidate in dataset.column_names:
            return candidate
    raise ValueError(
        f"None of the expected columns {candidates} exist. Found: {dataset.column_names}"
    )


def _speaker_values(dataset: Dataset, speaker_column: str | None) -> set[str]:
    if not speaker_column or speaker_column not in dataset.column_names:
        return set()
    return {
        str(value).strip()
        for value in dataset[speaker_column]
        if value is not None and str(value).strip()
    }


def speaker_overlap_report(
    splits: DatasetDict, speaker_column: str | None
) -> dict[str, int]:
    speakers = {
        name: _speaker_values(dataset, speaker_column)
        for name, dataset in splits.items()
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    return {
        f"{left}_{right}": len(speakers.get(left, set()) & speakers.get(right, set()))
        for left, right in pairs
    }


def _group_split(
    dataset: Dataset,
    speaker_column: str,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetDict:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, raw_speaker in enumerate(dataset[speaker_column]):
        speaker = str(raw_speaker).strip() if raw_speaker is not None else ""
        groups[speaker or f"__missing_speaker_{index}"].append(index)

    if len(groups) < 3:
        raise ValueError(
            f"Speaker-disjoint splitting requires at least 3 speaker groups; found {len(groups)}"
        )

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    total = len(dataset)
    target_test = max(1, round(total * test_ratio))
    target_validation = max(1, round(total * validation_ratio))
    test_indices: list[int] = []
    validation_indices: list[int] = []
    train_indices: list[int] = []

    for _, indices in group_items:
        if len(test_indices) < target_test:
            test_indices.extend(indices)
        elif len(validation_indices) < target_validation:
            validation_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if not train_indices or not validation_indices or not test_indices:
        raise ValueError("Unable to create non-empty speaker-disjoint train/validation/test splits")

    return DatasetDict(
        train=dataset.select(sorted(train_indices)),
        validation=dataset.select(sorted(validation_indices)),
        test=dataset.select(sorted(test_indices)),
    )


def _row_split(
    dataset: Dataset,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetDict:
    holdout_ratio = validation_ratio + test_ratio
    first = dataset.train_test_split(test_size=holdout_ratio, seed=seed, shuffle=True)
    relative_test_ratio = test_ratio / holdout_ratio
    second = first["test"].train_test_split(
        test_size=relative_test_ratio,
        seed=seed + 1,
        shuffle=True,
    )
    return DatasetDict(
        train=first["train"],
        validation=second["train"],
        test=second["test"],
    )


def limit_dataset_hours(dataset: Dataset, max_hours: float, seed: int) -> Dataset:
    """Deterministically sample at most ``max_hours`` without exceeding it."""
    if max_hours <= 0:
        return dataset

    max_seconds = max_hours * 3600.0
    total_seconds = sum(float(value) for value in dataset["audio_duration"])
    if total_seconds <= max_seconds:
        return dataset

    shuffled = dataset.shuffle(seed=seed)
    selected_indices: list[int] = []
    selected_seconds = 0.0
    for index, raw_duration in enumerate(shuffled["audio_duration"]):
        duration = float(raw_duration)
        if selected_seconds + duration <= max_seconds:
            selected_indices.append(index)
            selected_seconds += duration

    limited = shuffled.select(selected_indices)
    LOGGER.info(
        "Limited training data from %.3f to %.3f hours (%d samples)",
        total_seconds / 3600.0,
        selected_seconds / 3600.0,
        len(limited),
    )
    return limited


def ensure_three_splits(
    raw: DatasetDict,
    speaker_column: str | None,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    rebuild_on_speaker_overlap: bool,
) -> tuple[DatasetDict, str, dict[str, int]]:
    required = {"train", "validation", "test"}
    available = set(raw.keys())

    if required.issubset(available):
        selected = DatasetDict({name: raw[name] for name in ("train", "validation", "test")})
        overlaps = speaker_overlap_report(selected, speaker_column)
        if not rebuild_on_speaker_overlap or not any(overlaps.values()):
            return selected, "existing", overlaps
        LOGGER.warning("Existing splits contain speaker overlap: %s; rebuilding splits", overlaps)

    combined = concatenate_datasets([raw[name] for name in sorted(raw.keys())])
    if speaker_column and speaker_column in combined.column_names:
        selected = _group_split(
            combined,
            speaker_column=speaker_column,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        policy = "deterministic_speaker_disjoint"
    else:
        LOGGER.warning("No speaker column found; using deterministic row-level splits")
        selected = _row_split(
            combined,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        policy = "deterministic_row_level"

    return selected, policy, speaker_overlap_report(selected, speaker_column)


def prepare_splits(
    config: dict[str, Any],
    state: PartialState,
    smoke_test: bool = False,
) -> tuple[DatasetDict, dict[str, Any], str, str, str | None]:
    dataset_path = config["dataset_path"]
    token = os.environ.get("HF_TOKEN")
    cache_dir = config.get("cache_dir")

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    with state.main_process_first():
        raw = load_dataset(
            dataset_path,
            revision=config.get("dataset_revision", "main"),
            cache_dir=cache_dir,
            token=token,
        )
        if not isinstance(raw, DatasetDict):
            raw = DatasetDict(train=raw)

        reference_split = raw["train"] if "train" in raw else next(iter(raw.values()))
        audio_column = detect_column(reference_split, config.get("audio_column", "audio"), ())
        text_column = detect_column(
            reference_split,
            config.get("text_column", "transcript"),
            ("transcription", "text", "sentence"),
        )
        requested_speaker = config.get("speaker_column", "user_id")
        speaker_column = (
            requested_speaker if requested_speaker in reference_split.column_names else None
        )

        splits, split_policy, overlaps = ensure_three_splits(
            raw,
            speaker_column=speaker_column,
            validation_ratio=float(config.get("validation_ratio", 0.1)),
            test_ratio=float(config.get("test_ratio", 0.1)),
            seed=int(config.get("seed", 42)),
            rebuild_on_speaker_overlap=bool(
                config.get("rebuild_splits_on_speaker_overlap", True)
            ),
        )

        if smoke_test:
            smoke_limits = {"train": 64, "validation": 24, "test": 24}
            for split_name, limit in smoke_limits.items():
                shuffled = splits[split_name].shuffle(seed=int(config.get("seed", 42)))
                splits[split_name] = shuffled.select(range(min(limit, len(shuffled))))

        sampling_rate = int(config.get("sampling_rate", 16_000))
        min_duration = float(config.get("min_audio_length", 0.2))
        max_duration = float(config.get("max_audio_length", 30.0))
        num_proc = max(1, int(config.get("preprocessing_num_proc", 2)))
        lowercase = bool(config.get("lowercase", True))

        prepared = DatasetDict()
        filtering_stats: dict[str, dict[str, int]] = {}
        for split_name, dataset in splits.items():
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sampling_rate))
            dataset = dataset.map(
                _duration_from_audio,
                input_columns=[audio_column],
                num_proc=num_proc,
                desc=f"Reading {split_name} audio durations",
            )
            dataset = dataset.map(
                _normalize_text_row,
                input_columns=[text_column],
                fn_kwargs={"lowercase": lowercase},
                num_proc=num_proc,
                desc=f"Normalizing {split_name} transcripts",
            )
            rows_before_filter = len(dataset)
            invalid_audio_rows = sum(
                not bool(value) for value in dataset["audio_is_valid"]
            )
            dataset = dataset.filter(
                lambda is_valid, duration, text: (
                    bool(is_valid)
                    and min_duration <= float(duration) <= max_duration
                    and bool(text.strip())
                ),
                input_columns=[
                    "audio_is_valid",
                    "audio_duration",
                    "normalized_text",
                ],
                num_proc=num_proc,
                desc=f"Filtering {split_name} invalid audio/duration/text",
            )
            filtering_stats[split_name] = {
                "invalid_audio_removed": invalid_audio_rows,
                "all_filtered_rows_removed": rows_before_filter - len(dataset),
            }
            if invalid_audio_rows:
                LOGGER.warning(
                    "%s: removed %d corrupt or empty audio row(s)",
                    split_name,
                    invalid_audio_rows,
                )
            dataset = dataset.remove_columns("audio_is_valid")
            if split_name == "train":
                dataset = limit_dataset_hours(
                    dataset,
                    max_hours=float(config.get("max_train_hours", 0.0)),
                    seed=int(config.get("seed", 42)),
                )
            prepared[split_name] = dataset

    manifest = {
        "dataset_path": dataset_path,
        "dataset_revision": config.get("dataset_revision", "main"),
        "split_policy": split_policy,
        "seed": int(config.get("seed", 42)),
        "speaker_column": speaker_column,
        "speaker_overlap_counts": overlaps,
        "max_train_hours": float(config.get("max_train_hours", 0.0)),
        "filtering": filtering_stats,
        "splits": {},
    }
    for name, dataset in prepared.items():
        durations = [float(value) for value in dataset["audio_duration"]]
        manifest["splits"][name] = {
            "samples": len(dataset),
            "hours": round(sum(durations) / 3600.0, 4),
            "speakers": len(_speaker_values(dataset, speaker_column)),
            "fingerprint": dataset._fingerprint,
        }

    return prepared, manifest, audio_column, "normalized_text", speaker_column


@dataclass
class WhisperSpeechCollator:
    processor: WhisperProcessor
    audio_column: str
    text_column: str
    sampling_rate: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        arrays: list[np.ndarray] = []
        texts: list[str] = []
        for feature in features:
            array, sample_rate = _audio_samples(feature[self.audio_column])
            if sample_rate != self.sampling_rate:
                raise ValueError(
                    f"Expected {self.sampling_rate} Hz after dataset casting, got {sample_rate} Hz"
                )
            arrays.append(array)
            texts.append(feature[self.text_column])

        batch = self.processor.feature_extractor(
            arrays,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            return_attention_mask=False,
        )
        label_features = [
            {"input_ids": self.processor.tokenizer(text).input_ids} for text in texts
        ]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        decoder_start = self.processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
        if labels.shape[1] and torch.all(labels[:, 0] == decoder_start):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class WhisperMetrics:
    def __init__(self, processor: WhisperProcessor, expected_rows: int | None = None):
        self.processor = processor
        self.expected_rows = expected_rows

    def set_expected_rows(self, expected_rows: int) -> None:
        self.expected_rows = expected_rows

    def __call__(self, prediction: Any) -> dict[str, float]:
        predicted_ids = prediction.predictions
        if isinstance(predicted_ids, tuple):
            predicted_ids = predicted_ids[0]
        label_ids = np.array(prediction.label_ids, copy=True)
        if self.expected_rows is not None:
            if len(predicted_ids) < self.expected_rows or len(label_ids) < self.expected_rows:
                raise ValueError(
                    "Gathered metric tensors are shorter than the expected split: "
                    f"predictions={len(predicted_ids)}, labels={len(label_ids)}, "
                    f"expected={self.expected_rows}"
                )
            predicted_ids = predicted_ids[: self.expected_rows]
            label_ids = label_ids[: self.expected_rows]
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        predictions = self.processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        references = self.processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        predictions = [normalize_text(text) for text in predictions]
        references = [normalize_text(text) for text in references]
        return {
            "wer": float(jiwer.wer(references, predictions)),
            "cer": float(jiwer.cer(references, predictions)),
        }


def newest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return str(max(checkpoints)[1]) if checkpoints else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_predictions(
    path: Path,
    prediction_output: Any,
    processor: WhisperProcessor,
    test_dataset: Dataset,
    speaker_column: str | None,
) -> None:
    predicted_ids = prediction_output.predictions
    if isinstance(predicted_ids, tuple):
        predicted_ids = predicted_ids[0]
    label_ids = np.array(prediction_output.label_ids, copy=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    predictions = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    durations = [float(value) for value in test_dataset["audio_duration"]]
    speakers = test_dataset[speaker_column] if speaker_column else None

    expected_rows = len(test_dataset)
    if len(predictions) < expected_rows or len(references) < expected_rows:
        raise ValueError(
            "Prediction output is shorter than the test dataset: "
            f"predictions={len(predictions)}, references={len(references)}, "
            f"test_rows={expected_rows}"
        )
    if len(predictions) > expected_rows or len(references) > expected_rows:
        LOGGER.warning(
            "Trimming distributed evaluation padding: predictions=%d, references=%d, "
            "test_rows=%d",
            len(predictions),
            len(references),
            expected_rows,
        )
        predictions = predictions[:expected_rows]
        references = references[:expected_rows]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["id", "reference", "prediction", "audio_duration"]
        if speaker_column:
            fieldnames.append(speaker_column)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (reference, prediction) in enumerate(zip(references, predictions)):
            row = {
                "id": index,
                "reference": normalize_text(reference),
                "prediction": normalize_text(prediction),
                "audio_duration": durations[index],
            }
            if speaker_column:
                row[speaker_column] = speakers[index]
            writer.writerow(row)


def build_training_arguments(
    config: dict[str, Any],
    output_dir: Path,
    smoke_test: bool,
    evaluation_only: bool = False,
) -> Seq2SeqTrainingArguments:
    eval_steps = 1 if smoke_test else int(config.get("eval_steps", 500))
    save_steps = 1 if smoke_test else int(config.get("save_steps", eval_steps))
    max_steps = 2 if smoke_test else int(config.get("max_steps", -1))
    num_workers = 0 if smoke_test else int(config.get("dataloader_num_workers", 2))

    train_batch_size = (
        2 if smoke_test else int(config.get("per_device_train_batch_size", 4))
    )
    eval_batch_size = (
        1 if smoke_test else int(config.get("per_device_eval_batch_size", 2))
    )
    gradient_accumulation = (
        1 if smoke_test else int(config.get("gradient_accumulation_steps", 4))
    )

    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        do_train=not evaluation_only,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=float(config.get("learning_rate", 1e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        # In Transformers 5.x a fractional warmup_steps value is interpreted
        # as a ratio, without using the deprecated warmup_ratio argument.
        warmup_steps=float(config.get("warmup_steps", 0.05)),
        num_train_epochs=float(config.get("num_train_epochs", 5)),
        max_steps=max_steps,
        fp16=bool(config.get("fp16", True)),
        bf16=bool(config.get("bf16", False)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        use_cache=False,
        optim=config.get("optim", "adamw_torch_fused"),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(config.get("save_total_limit", 1)),
        load_best_model_at_end=not evaluation_only,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=1 if smoke_test else int(config.get("logging_steps", 20)),
        logging_first_step=True,
        report_to="none",
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=int(config.get("generation_max_length", 225)),
        generation_num_beams=int(config.get("generation_num_beams", 1)),
        eval_accumulation_steps=1,
        train_sampling_strategy="group_by_length",
        length_column_name="audio_duration",
        remove_unused_columns=False,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=num_workers > 0,
        dataloader_prefetch_factor=2 if num_workers > 0 else None,
        ddp_find_unused_parameters=False,
        save_on_each_node=False,
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
        torch_compile=bool(config.get("torch_compile", False)),
    )


def main() -> None:
    args = parse_args()
    if args.smoke_test and args.evaluation_only:
        raise ValueError("--smoke_test and --evaluation_only cannot be used together")
    config = load_yaml(args.config)
    state = PartialState()
    setup_logging(state.is_main_process)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    base_output_dir = Path(config.get("output_dir", "outputs/whisper-small-luhya"))
    output_dir = base_output_dir / "smoke-test" if args.smoke_test else base_output_dir
    if state.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    state.wait_for_everyone()

    splits, manifest, audio_column, text_column, speaker_column = prepare_splits(
        config, state, smoke_test=args.smoke_test
    )
    if args.smoke_test:
        manifest["smoke_test"] = True

    model_name = config.get("model_name", "openai/whisper-small")
    language = config.get("language", "swahili")
    task = config.get("task", "transcribe")
    cache_dir = config.get("cache_dir")
    token = os.environ.get("HF_TOKEN")

    final_model_dir = output_dir / "final-model"
    evaluation_model_source: str | None = None
    if args.evaluation_only:
        if final_model_dir.is_dir():
            evaluation_model_source = str(final_model_dir)
        else:
            evaluation_model_source = newest_checkpoint(output_dir)
        if evaluation_model_source is None:
            raise FileNotFoundError(
                f"No final model or checkpoint found under {output_dir}; "
                "evaluation-only recovery is not possible"
            )
        LOGGER.info("Evaluation-only mode: loading %s", evaluation_model_source)

    model_source = evaluation_model_source or model_name
    processor_source = (
        str(final_model_dir)
        if args.evaluation_only and final_model_dir.is_dir()
        else model_name
    )

    with state.main_process_first():
        processor = WhisperProcessor.from_pretrained(
            processor_source,
            language=language,
            task=task,
            cache_dir=cache_dir,
            token=token,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            model_source,
            cache_dir=cache_dir,
            token=token,
            attn_implementation=config.get("attn_implementation", "sdpa"),
        )

    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False

    if bool(config.get("freeze_encoder", False)):
        model.freeze_encoder()

    training_args = build_training_arguments(
        config,
        output_dir,
        args.smoke_test,
        evaluation_only=args.evaluation_only,
    )
    collator = WhisperSpeechCollator(
        processor=processor,
        audio_column=audio_column,
        text_column=text_column,
        sampling_rate=int(config.get("sampling_rate", 16_000)),
    )
    callbacks = []
    patience = int(config.get("early_stopping_patience", 3))
    if patience > 0 and not args.smoke_test and not args.evaluation_only:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    metric_computer = WhisperMetrics(
        processor,
        expected_rows=len(splits["validation"]),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        data_collator=collator,
        processing_class=processor,
        compute_metrics=metric_computer,
        callbacks=callbacks,
    )

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = newest_checkpoint(output_dir)
        if resume is None:
            LOGGER.info("No checkpoint found in %s; starting from the base model", output_dir)

    world_size = max(1, state.num_processes)
    global_batch = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * world_size
    )
    if state.is_main_process:
        LOGGER.info("Devices/processes: %d; optimizer global batch: %d", world_size, global_batch)
        LOGGER.info("Split manifest: %s", json.dumps(manifest, indent=2))

    if args.evaluation_only:
        train_metrics: dict[str, Any] = {}
        train_results_path = output_dir / "train_results.json"
        if train_results_path.is_file():
            with open(train_results_path, encoding="utf-8") as handle:
                train_metrics = json.load(handle)
    else:
        train_result = trainer.train(resume_from_checkpoint=resume)
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

    metric_computer.set_expected_rows(len(splits["validation"]))
    validation_metrics = trainer.evaluate(
        eval_dataset=splits["validation"], metric_key_prefix="validation"
    )
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)

    metric_computer.set_expected_rows(len(splits["test"]))
    test_output = trainer.predict(splits["test"], metric_key_prefix="test")
    trainer.log_metrics("test", test_output.metrics)
    trainer.save_metrics("test", test_output.metrics)

    if state.is_main_process:
        trainer.save_model(str(output_dir / "final-model"))
        processor.save_pretrained(str(output_dir / "final-model"))
        write_predictions(
            output_dir / "test_predictions.csv",
            test_output,
            processor,
            splits["test"],
            speaker_column,
        )
        summary = {
            "model_name": model_name,
            "language_conditioning_token": language,
            "language_token_note": config.get("language_token_note"),
            "task": task,
            "world_size": world_size,
            "optimizer_global_batch_size": global_batch,
            "split_manifest": manifest,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_output.metrics,
        }
        with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as handle:
            json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)

        rows = []
        for split_name, metrics in (
            ("validation", validation_metrics),
            ("test", test_output.metrics),
        ):
            rows.append(
                {
                    "split": split_name,
                    "wer_percent": 100 * float(metrics[f"{split_name}_wer"]),
                    "cer_percent": 100 * float(metrics[f"{split_name}_cer"]),
                    "loss": metrics.get(f"{split_name}_loss", ""),
                }
            )
        with open(output_dir / "metrics_summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["split", "wer_percent", "cer_percent", "loss"]
            )
            writer.writeheader()
            writer.writerows(rows)

        LOGGER.info("Evaluation artifacts saved to %s", output_dir)


if __name__ == "__main__":
    main()
