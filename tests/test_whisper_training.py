import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from datasets import Dataset, DatasetDict

from scripts.train_whisper import (
    _duration_from_audio,
    ensure_three_splits,
    limit_dataset_hours,
    normalize_text,
    WhisperMetrics,
    write_predictions,
)


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


class WhisperTrainingUtilitiesTest(unittest.TestCase):
    def setUp(self):
        self.rows = {
            "audio": ["placeholder"] * 30,
            "transcript": [f"utterance {index}" for index in range(30)],
            "user_id": [f"speaker-{index // 3}" for index in range(30)],
        }

    def test_normalize_text(self):
        self.assertEqual(normalize_text("  LUKHAYO’  Mulembe  "), "lukhayo' mulembe")

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


if __name__ == "__main__":
    unittest.main()
