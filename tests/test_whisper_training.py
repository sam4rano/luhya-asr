import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import Dataset, DatasetDict

from scripts.train_whisper import (
    _duration_from_audio,
    canonical_text,
    ensure_three_splits,
    limit_dataset_hours,
    normalize_text,
    WhisperMetrics,
    write_predictions,
)
from src.training.metrics import decode_ctc_predictions
from src.training.collator import DataCollatorCTCRawAudioWithPadding


class BrokenAudio:
    def get_all_samples(self):
        raise RuntimeError("No audio frames were decoded")


class FakeTokenizer:
    pad_token_id = 0

    def batch_decode(self, rows, skip_special_tokens=True):
        del skip_special_tokens
        return [str(int(row[0])) for row in rows]


class FakeProcessor:
    tokenizer = FakeTokenizer()


class FakeCTCTokenizer:
    pad_token_id = 0
    symbols = {0: "", 1: "a", 2: "b", 3: " "}

    def batch_decode(self, rows, group_tokens=True):
        decoded = []
        for row in rows:
            ids = [int(value) for value in row]
            if group_tokens:
                ids = [value for index, value in enumerate(ids) if index == 0 or value != ids[index - 1]]
            decoded.append("".join(self.symbols.get(value, "") for value in ids))
        return decoded


class FakeCTCProcessor:
    tokenizer = FakeCTCTokenizer()

    def batch_decode(self, rows, group_tokens=True):
        return self.tokenizer.batch_decode(rows, group_tokens=group_tokens)


class FakeRawTokenizer:
    def __call__(self, texts, padding, return_tensors):
        self.last_texts = texts
        self.last_padding = padding
        self.last_return_tensors = return_tensors
        return {
            "input_ids": torch.tensor([[1, 2], [2, 0]]),
            "attention_mask": torch.tensor([[1, 1], [1, 0]]),
        }


class FakeRawProcessor:
    def __init__(self):
        self.tokenizer = FakeRawTokenizer()

    def __call__(self, audio, sampling_rate, padding, return_tensors):
        self.last_audio = audio
        self.last_sampling_rate = sampling_rate
        return {"input_features": torch.ones((len(audio), 3, 2))}


class WhisperTrainingUtilitiesTest(unittest.TestCase):
    def setUp(self):
        self.rows = {
            "audio": ["placeholder"] * 30,
            "transcript": [f"utterance {index}" for index in range(30)],
            "user_id": [f"speaker-{index // 3}" for index in range(30)],
        }

    def test_normalize_text(self):
        self.assertEqual(normalize_text("  LUKHAYO’  Mulembe  "), "lukhayo' mulembe")

    def test_canonical_text_is_shared_punctuation_policy(self):
        self.assertEqual(
            canonical_text("  Café, N'da ’Uk'hu?!  "),
            "cafe n'da 'uk'hu",
        )
        self.assertEqual(canonical_text("  LUKHAYO’  Mulembe  "), "lukhayo' mulembe")
        self.assertEqual(canonical_text("Hello, world — 123."), "hello world 123")

    def test_corrupt_audio_is_marked_invalid_instead_of_crashing(self):
        result = _duration_from_audio(BrokenAudio())

        self.assertEqual(result["audio_duration"], 0.0)
        self.assertFalse(result["audio_is_valid"])

    def test_prediction_export_trims_distributed_padding(self):
        dataset = Dataset.from_dict({"audio_duration": [1.0, 2.0]})
        output = SimpleNamespace(
            predictions=np.array([[10], [20], [10], [20]]),
            label_ids=np.array([[11], [21], [11], [21]]),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            write_predictions(path, output, FakeProcessor(), dataset, None)
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["prediction"] for row in rows], ["10", "20"])

    def test_metrics_ignore_distributed_padding(self):
        metric = WhisperMetrics(FakeProcessor(), expected_rows=2)
        output = SimpleNamespace(
            predictions=np.array([[10], [20], [99], [99]]),
            label_ids=np.array([[10], [20], [11], [21]]),
        )

        result = metric(output)

        self.assertEqual(result["wer"], 0.0)
        self.assertEqual(result["cer"], 0.0)

    def test_ctc_decode_accepts_preprocessed_ids_and_trims_distributed_padding(self):
        output = SimpleNamespace(
            predictions=np.array(
                [
                    [1, 1, 3, 2],
                    [2, 2, 3, 1],
                    [2, 2, 2, 2],
                    [1, 1, 1, 1],
                ]
            ),
            label_ids=np.array(
                [
                    [1, 3, 2, -100],
                    [2, 3, 1, -100],
                    [1, 1, 1, -100],
                    [2, 2, 2, -100],
                ]
            ),
        )

        predictions, references = decode_ctc_predictions(
            output,
            FakeCTCProcessor(),
            expected_rows=2,
        )

        self.assertEqual(predictions, ["a b", "b a"])
        self.assertEqual(references, ["a b", "b a"])

    def test_ctc_raw_collator_extracts_features_per_batch(self):
        processor = FakeRawProcessor()
        collator = DataCollatorCTCRawAudioWithPadding(processor)
        features = [
            {
                "audio": {
                    "array": np.array([0.0, 0.1], dtype=np.float32),
                    "sampling_rate": 16_000,
                },
                "clean_transcription": "a b",
            },
            {
                "audio": {
                    "array": np.array([0.2, 0.3], dtype=np.float32),
                    "sampling_rate": 16_000,
                },
                "clean_transcription": "b",
            },
        ]

        batch = collator(features)

        self.assertEqual(tuple(batch["input_features"].shape), (2, 3, 2))
        self.assertEqual(batch["labels"].tolist(), [[1, 2], [2, -100]])
        self.assertEqual(processor.last_sampling_rate, 16_000)

    def test_missing_splits_are_rebuilt_by_speaker(self):
        raw = DatasetDict(train=Dataset.from_dict(self.rows))
        splits, policy, overlaps = ensure_three_splits(
            raw,
            speaker_column="user_id",
            validation_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            rebuild_on_speaker_overlap=True,
        )

        self.assertEqual(policy, "deterministic_speaker_disjoint")
        self.assertEqual(set(splits), {"train", "validation", "test"})
        self.assertEqual(sum(len(split) for split in splits.values()), 30)
        self.assertTrue(all(len(split) > 0 for split in splits.values()))
        self.assertFalse(any(overlaps.values()))

    def test_existing_disjoint_splits_are_retained(self):
        raw = DatasetDict(
            train=Dataset.from_dict({key: values[:18] for key, values in self.rows.items()}),
            validation=Dataset.from_dict(
                {key: values[18:24] for key, values in self.rows.items()}
            ),
            test=Dataset.from_dict({key: values[24:] for key, values in self.rows.items()}),
        )
        splits, policy, overlaps = ensure_three_splits(
            raw,
            speaker_column="user_id",
            validation_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            rebuild_on_speaker_overlap=True,
        )

        self.assertEqual(policy, "existing")
        self.assertEqual(
            [len(splits[name]) for name in ("train", "validation", "test")],
            [18, 6, 6],
        )
        self.assertFalse(any(overlaps.values()))

    def test_training_hour_limit_is_deterministic_and_never_exceeded(self):
        dataset = Dataset.from_dict(
            {
                "id": list(range(20)),
                "audio_duration": [600.0] * 20,
            }
        )
        first = limit_dataset_hours(dataset, max_hours=2.0, seed=42)
        second = limit_dataset_hours(dataset, max_hours=2.0, seed=42)

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(first), 12)
        self.assertLessEqual(sum(first["audio_duration"]), 2.0 * 3600.0)

    def test_test_ratio_zero_produces_two_way_split(self):
        raw = DatasetDict(train=Dataset.from_dict(self.rows))
        splits, policy, overlaps = ensure_three_splits(
            raw,
            speaker_column="user_id",
            validation_ratio=0.2,
            test_ratio=0.0,
            seed=42,
            rebuild_on_speaker_overlap=True,
        )

        self.assertEqual(policy, "deterministic_speaker_disjoint")
        self.assertEqual(set(splits), {"train", "validation"})
        self.assertEqual(sum(len(split) for split in splits.values()), 30)
        self.assertFalse(any(overlaps.values()))

    def test_too_few_speakers_falls_back_to_row_split(self):
        few_speakers = {
            key: values for key, values in self.rows.items()
        }
        few_speakers["user_id"] = [f"speaker-{index % 2}" for index in range(30)]
        raw = DatasetDict(train=Dataset.from_dict(few_speakers))
        splits, policy, overlaps = ensure_three_splits(
            raw,
            speaker_column="user_id",
            validation_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            rebuild_on_speaker_overlap=True,
        )

        self.assertEqual(policy, "deterministic_row_level")
        self.assertEqual(set(splits), {"train", "validation", "test"})
        self.assertEqual(sum(len(split) for split in splits.values()), 30)

    def test_invalid_split_ratios_are_rejected(self):
        raw = DatasetDict(train=Dataset.from_dict(self.rows))
        with self.assertRaises(ValueError):
            ensure_three_splits(
                raw, "user_id", 0.6, 0.6, 42, True
            )
        with self.assertRaises(ValueError):
            ensure_three_splits(
                raw, "user_id", 0.0, 0.1, 42, True
            )


if __name__ == "__main__":
    unittest.main()
