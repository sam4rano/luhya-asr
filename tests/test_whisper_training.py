import unittest

from datasets import Dataset, DatasetDict

from scripts.train_whisper import (
    _duration_from_audio,
    ensure_three_splits,
    limit_dataset_hours,
    normalize_text,
)


class BrokenAudio:
    def get_all_samples(self):
        raise RuntimeError("No audio frames were decoded")


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
