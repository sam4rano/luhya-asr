import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jiwer

from scripts.build_comparison_report import main, validate_and_pair, validate_manifests


class ComparisonReportTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "dataset_path": "org/dataset",
            "dataset_revision": "abc123",
            "split_policy": "deterministic_speaker_disjoint",
            "seed": 42,
            "speaker_column": "user_id",
            "speaker_overlap_counts": {
                "train_validation": 0,
                "train_test": 0,
                "validation_test": 0,
            },
            "max_train_hours": 40.0,
            "filtering": {},
            "splits": {
                "train": {"samples": 8, "hours": 1.0, "speakers": 4, "fingerprint": "train"},
                "validation": {"samples": 1, "hours": 0.1, "speakers": 1, "fingerprint": "val"},
                "test": {"samples": 1, "hours": 0.1, "speakers": 1, "fingerprint": "test"},
            },
        }

    def test_identical_manifests_pass(self):
        result = validate_manifests(
            {"split_manifest": self.manifest},
            {"split_manifest": dict(self.manifest)},
        )
        self.assertEqual(result["dataset_revision"], "abc123")

    def test_different_revisions_stop_comparison(self):
        other = dict(self.manifest)
        other["dataset_revision"] = "different"
        with self.assertRaises(ValueError):
            validate_manifests(
                {"split_manifest": self.manifest},
                {"split_manifest": other},
            )

    def test_pairing_requires_identical_references(self):
        whisper = [{
            "id": "0",
            "reference": "hello",
            "prediction": "hello",
            "audio_duration": "2.0",
            "user_id": "speaker-1",
        }]
        wav2vec = [{**whisper[0], "reference": "different"}]
        with self.assertRaises(ValueError):
            validate_and_pair(whisper, wav2vec)

    def test_end_to_end_report_builds_model_cards_and_comparison(self):
        prediction_sets = {
            "whisper": ["hello world", "good day"],
            "wav2vec": ["hello", "good day"],
        }
        references = ["hello world", "good day"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whisper_dir = root / "whisper"
            wav2vec_dir = root / "wav2vec"
            output_dir = root / "comparison"
            for label, run_dir in (("whisper", whisper_dir), ("wav2vec", wav2vec_dir)):
                (run_dir / "final-model").mkdir(parents=True)
                predictions = prediction_sets[label]
                wer = jiwer.wer(references, predictions)
                cer = jiwer.cer(references, predictions)
                summary = {
                    "model_name": "base/model",
                    "code_revision": "code123",
                    "optimizer_global_batch_size": 32,
                    "split_manifest": self.manifest,
                    "config": {"min_audio_length": 2.0, "max_audio_length": 30.0},
                    "runtime_versions": {"transformers": "5.14.1"},
                    "validation_metrics": {
                        "validation_wer": wer,
                        "validation_cer": cer,
                    },
                    "test_metrics": {"test_wer": wer, "test_cer": cer},
                }
                (run_dir / "evaluation_summary.json").write_text(
                    json.dumps(summary),
                    encoding="utf-8",
                )
                (run_dir / "metrics_summary.csv").write_text(
                    "split,wer_percent,cer_percent,loss\n",
                    encoding="utf-8",
                )
                with (run_dir / "test_predictions.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "id",
                            "reference",
                            "prediction",
                            "audio_duration",
                            "user_id",
                            "dialect",
                        ],
                    )
                    writer.writeheader()
                    for index, (reference, prediction) in enumerate(
                        zip(references, predictions)
                    ):
                        writer.writerow(
                            {
                                "id": index,
                                "reference": reference,
                                "prediction": prediction,
                                "audio_duration": 2.0 + index,
                                "user_id": "speaker-1",
                                "dialect": "Wanga",
                            }
                        )

            argv = [
                "build_comparison_report.py",
                "--whisper-dir",
                str(whisper_dir),
                "--wav2vec-dir",
                str(wav2vec_dir),
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            self.assertTrue((output_dir / "Luhya_ASR_Experiment_Report.md").is_file())
            self.assertTrue((output_dir / "paired_test_predictions.csv").is_file())
            self.assertTrue((output_dir / "speaker_metrics.csv").is_file())
            self.assertTrue((whisper_dir / "final-model" / "README.md").is_file())
            self.assertTrue((wav2vec_dir / "final-model" / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
