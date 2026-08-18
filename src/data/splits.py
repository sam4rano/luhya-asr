# src/data/splits.py
"""Shared deterministic data preparation for the Luhya ASR pipelines.

Both the CTC/Wav2Vec2 pipeline (``scripts/train_model.py``) and the Whisper
pipeline (``scripts/train_whisper.py``) must train and evaluate on the *same*
clips so that cross-model results are comparable. All split, filtering, and
hour-cap logic lives here; the two training scripts only adapt the output
columns to their own processors.
"""

from __future__ import annotations

import logging
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset

LOGGER = logging.getLogger("luhya-splits")


def normalize_text(text: Any, lowercase: bool = True) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(text))
    normalized = normalized.replace("’", "'").replace("ʼ", "'")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.lower() if lowercase else normalized


# Canonical character set shared by the CTC and Whisper pipelines. Keeping this
# set in sync with the CTC `character_set` in the YAML configs is what makes the
# two pipelines train and get scored on identical strings: letters, digits,
# space, and apostrophe (apostrophes are word-internal in Luhya orthography).
CANONICAL_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789 '")


def canonical_text(text: Any, lowercase: bool = True) -> str:
    """Shared canonical text used for training labels AND evaluation scoring.

    Applied identically in the CTC and Whisper pipelines so punctuation,
    accents, and spacing cannot make cross-model WER/CER differ:
    - NFC normalization, curly quotes -> straight apostrophes, lowercase
    - diacritics folded to ASCII base letters (e.g. "é" -> "e", "ñ" -> "n")
    - only letters, digits, space, and apostrophe are kept (punctuation dropped)
    - whitespace collapsed
    """
    if text is None:
        return ""
    text = normalize_text(text, lowercase=lowercase)
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = unicodedata.normalize("NFC", text)
    text = "".join(char for char in text if char in CANONICAL_CHARS)
    return re.sub(r"\s+", " ", text).strip()


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
    return {"normalized_text": canonical_text(text, lowercase=lowercase)}


def _keep_row(is_valid: Any, duration: Any, text: Any,
              min_duration: float, max_duration: float) -> bool:
    """Filter predicate for valid audio inside the [min, max] duration window."""
    return (
        bool(is_valid)
        and min_duration <= float(duration) <= max_duration
        and bool(text.strip())
    )


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

    min_groups = 3 if test_ratio > 0 else 2
    if len(groups) < min_groups:
        raise ValueError(
            f"Speaker-disjoint splitting requires at least {min_groups} speaker "
            f"groups; found {len(groups)}"
        )

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    total = len(dataset)
    target_validation = max(1, round(total * validation_ratio))
    validation_indices: list[int] = []
    train_indices: list[int] = []

    if test_ratio <= 0:
        for _, indices in group_items:
            if len(validation_indices) < target_validation:
                validation_indices.extend(indices)
            else:
                train_indices.extend(indices)
        if not train_indices or not validation_indices:
            raise ValueError(
                "Unable to create non-empty speaker-disjoint train/validation splits"
            )
        return DatasetDict(
            train=dataset.select(sorted(train_indices)),
            validation=dataset.select(sorted(validation_indices)),
        )

    target_test = max(1, round(total * test_ratio))
    test_indices: list[int] = []
    for _, indices in group_items:
        if len(test_indices) < target_test:
            test_indices.extend(indices)
        elif len(validation_indices) < target_validation:
            validation_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if not train_indices or not validation_indices or not test_indices:
        raise ValueError(
            "Unable to create non-empty speaker-disjoint train/validation/test splits"
        )

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
    if test_ratio <= 0:
        split = dataset.train_test_split(
            test_size=validation_ratio, seed=seed, shuffle=True
        )
        return DatasetDict(
            train=split["train"],
            validation=split["test"],
        )

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
    """Build deterministic train/validation(/test) splits with safe fallbacks.

    - ``test_ratio <= 0`` produces a two-way train/validation split (80/20).
    - Pre-existing speaker-disjoint splits are retained when they match the
      requested layout.
    - If speaker-disjoint splitting is impossible (too few speakers, or it
      produces an empty split), it falls back to a deterministic row-level split
      instead of crashing.
    """
    validation_ratio = float(validation_ratio)
    test_ratio = float(test_ratio)
    if validation_ratio <= 0 or test_ratio < 0:
        raise ValueError(
            f"validation_ratio must be > 0 and test_ratio must be >= 0; got "
            f"validation_ratio={validation_ratio}, test_ratio={test_ratio}"
        )
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"validation_ratio + test_ratio must be < 1; got "
            f"{validation_ratio} + {test_ratio} = {validation_ratio + test_ratio}"
        )

    expected = {"train", "validation"} if test_ratio <= 0 else {"train", "validation", "test"}
    available = set(raw.keys())

    if expected.issubset(available):
        selected = DatasetDict({name: raw[name] for name in sorted(expected)})
        overlaps = speaker_overlap_report(selected, speaker_column)
        if not rebuild_on_speaker_overlap or not any(overlaps.values()):
            return selected, "existing", overlaps
        LOGGER.warning(
            "Existing splits contain speaker overlap: %s; rebuilding splits", overlaps
        )

    combined = concatenate_datasets([raw[name] for name in sorted(raw.keys())])
    policy: str | None = None
    if speaker_column and speaker_column in combined.column_names:
        try:
            selected = _group_split(
                combined,
                speaker_column=speaker_column,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            policy = "deterministic_speaker_disjoint"
        except ValueError as exc:
            LOGGER.warning(
                "Speaker-disjoint splitting not possible (%s); "
                "falling back to deterministic row-level splits",
                exc,
            )
    if policy is None:
        LOGGER.warning("Using deterministic row-level splits")
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
    state: Any = None,
    smoke_test: bool = False,
    token: str | None = None,
) -> tuple[DatasetDict, dict[str, Any], str, str, str | None]:
    """Load, split, filter, and hour-cap the raw dataset.

    Applies the same deterministic speaker-disjoint split, the same audio-duration
    and text filters, and the same training-hour cap for every model, so results
    are comparable across pipelines. The default layout is 80/10/10; setting
    ``test_ratio: 0.0`` (with ``validation_ratio: 0.20``) produces a two-way
    80/20 train/validation split for pipelines that prefer not to hold out a test
    set. If speaker-disjoint splitting is impossible it falls back to row-level.

    Returns ``(prepared, manifest, audio_column, text_column, speaker_column)``
    where ``text_column`` points to the shared ``normalized_text`` column.
    """
    dataset_path = config["dataset_path"]
    cache_dir = config.get("cache_dir")

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if state is not None:
        with state.main_process_first():
            return _prepare_splits_core(
                config, dataset_path, cache_dir, token, smoke_test
            )
    return _prepare_splits_core(config, dataset_path, cache_dir, token, smoke_test)


def _prepare_splits_core(
    config: dict[str, Any],
    dataset_path: str,
    cache_dir: str | None,
    token: str | None,
    smoke_test: bool,
) -> tuple[DatasetDict, dict[str, Any], str, str, str | None]:
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
        for split_name in list(splits.keys()):
            if split_name not in smoke_limits:
                continue
            shuffled = splits[split_name].shuffle(seed=int(config.get("seed", 42)))
            splits[split_name] = shuffled.select(
                range(min(smoke_limits[split_name], len(shuffled)))
            )

    sampling_rate = int(config.get("sampling_rate", 16_000))
    min_duration = float(config.get("min_audio_length", 0.0))
    max_duration = float(config.get("max_audio_length", 30.0))
    if min_duration < 0 or max_duration <= 0 or min_duration > max_duration:
        raise ValueError(
            f"Invalid audio window: min_audio_length={min_duration} must be >= 0 "
            f"and <= max_audio_length={max_duration}"
        )
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
        invalid_audio_rows = sum(not bool(value) for value in dataset["audio_is_valid"])
        dataset = dataset.filter(
            _keep_row,
            input_columns=[
                "audio_is_valid",
                "audio_duration",
                "normalized_text",
            ],
            fn_kwargs={
                "min_duration": min_duration,
                "max_duration": max_duration,
            },
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
