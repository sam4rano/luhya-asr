#!/usr/bin/env python3
"""Train/evaluate the CTC model with the shared Luhya comparison protocol."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from accelerate import PartialState
from transformers import AutoModelForCTC, AutoProcessor, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_vocabulary, create_processor, load_datasets
from src.models.factory import create_asr_model
from src.training.collator import DataCollatorCTCRawAudioWithPadding
from src.training.metrics import decode_ctc_predictions
from src.training.trainer import create_asr_trainer
from src.utils.config import load_config

LOGGER = logging.getLogger("luhya-ctc")


def setup_logging(is_main_process: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if is_main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_env_file(env_path: Path) -> None:
    """Load a simple local .env without overriding explicit environment values."""
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value.strip().strip("'").strip('"'))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint path, or 'latest' to use the newest checkpoint in the run directory",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run two optimizer steps on small deterministic subsets",
    )
    parser.add_argument(
        "--evaluation_only",
        action="store_true",
        help="Reload the final model (or latest checkpoint) and regenerate artifacts",
    )
    return parser.parse_args()


def newest_checkpoint(output_dir: Path) -> str | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return str(max(checkpoints)[1]) if checkpoints else None


def runtime_versions() -> dict[str, str]:
    packages = ("transformers", "datasets", "accelerate", "torch", "jiwer")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


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
    processor: Any,
    test_dataset: Any,
    speaker_column: str | None,
) -> None:
    expected_rows = len(test_dataset)
    predictions, references = decode_ctc_predictions(
        prediction_output,
        processor,
        expected_rows=expected_rows,
    )
    durations = [float(value) for value in test_dataset["audio_duration"]]
    metadata_columns = [
        name
        for name in (speaker_column, "dialect", "language")
        if name and name in test_dataset.column_names
    ]

    fieldnames = ["id", "reference", "prediction", "audio_duration", *metadata_columns]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (reference, prediction) in enumerate(zip(references, predictions)):
            row = {
                "id": index,
                "reference": reference,
                "prediction": prediction,
                "audio_duration": durations[index],
            }
            for column in metadata_columns:
                row[column] = test_dataset[column][index]
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.smoke_test and args.evaluation_only:
        raise ValueError("--smoke_test and --evaluation_only cannot be used together")

    state = PartialState()
    setup_logging(state.is_main_process)
    load_env_file(PROJECT_ROOT / ".env")
    config = load_config(args.config)
    set_seed(config.seed)

    base_experiment_name = config.get_experiment_name()
    experiment_name = (
        f"{base_experiment_name}-smoke-test" if args.smoke_test else base_experiment_name
    )
    if args.smoke_test:
        config.max_steps = 2
        config.num_epochs = 1
        config.eval_steps = 1
        config.save_steps = 1
        config.logging_steps = 1
        config.report_to = "none"
        config.persistent_workers = False
        config.dataloader_num_workers = 0

    run_dir = Path(config.output_dir) / experiment_name
    final_model_dir = run_dir / "final-model"
    processor_dir = run_dir / "processor"
    ctc_dir = run_dir / "ctc_tokenizer"
    if state.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    state.wait_for_everyone()

    splits, speaker_column, split_manifest = load_datasets(
        config,
        state=state,
        smoke_test=args.smoke_test,
    )
    train_dataset = splits["train"]
    validation_dataset = splits["validation"]
    test_dataset = splits.get("test")
    if test_dataset is None or len(test_dataset) == 0:
        raise ValueError("The comparison protocol requires a non-empty held-out test split")

    evaluation_model_source: str | None = None
    if args.evaluation_only:
        if final_model_dir.is_dir():
            evaluation_model_source = str(final_model_dir)
        else:
            evaluation_model_source = newest_checkpoint(run_dir)
        if evaluation_model_source is None:
            raise FileNotFoundError(
                f"No final model or checkpoint found under {run_dir}; "
                "evaluation-only recovery is not possible"
            )
        processor_source = final_model_dir if final_model_dir.is_dir() else processor_dir
        processor = AutoProcessor.from_pretrained(
            processor_source,
            cache_dir=config.cache_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        model = AutoModelForCTC.from_pretrained(
            evaluation_model_source,
            cache_dir=config.cache_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        LOGGER.info("Evaluation-only mode: loading %s", evaluation_model_source)
    else:
        vocab_dict = build_vocabulary(
            config.character_set,
            config.add_language_tokens,
            None,
        )
        if state.is_main_process:
            ctc_dir.mkdir(parents=True, exist_ok=True)
            with (ctc_dir / "vocab.json").open("w", encoding="utf-8") as handle:
                json.dump(vocab_dict, handle, indent=2, ensure_ascii=False)
        state.wait_for_everyone()

        processor = create_processor(config, str(ctc_dir))
        if state.is_main_process:
            processor.save_pretrained(processor_dir)
        state.wait_for_everyone()
        model = create_asr_model(config, processor)

    collator = DataCollatorCTCRawAudioWithPadding(
        processor=processor,
        audio_column=config.audio_column,
        text_column="clean_transcription",
        sampling_rate=config.sampling_rate,
    )
    trainer = create_asr_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processor=processor,
        experiment_name=experiment_name,
        config=config,
    )

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = newest_checkpoint(run_dir)
        if resume is None:
            LOGGER.info("No checkpoint found in %s; starting from the base model", run_dir)

    world_size = max(1, state.num_processes)
    global_batch = (
        trainer.args.per_device_train_batch_size
        * trainer.args.gradient_accumulation_steps
        * world_size
    )
    if state.is_main_process:
        LOGGER.info("Devices/processes: %d; optimizer global batch: %d", world_size, global_batch)
        LOGGER.info("Split manifest: %s", json.dumps(split_manifest, indent=2))

    if args.evaluation_only:
        train_metrics: dict[str, Any] = {}
        train_results_path = run_dir / "train_results.json"
        if train_results_path.is_file():
            train_metrics = json.loads(train_results_path.read_text(encoding="utf-8"))
    else:
        train_result = trainer.train(resume_from_checkpoint=resume)
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_model(str(final_model_dir))
        if state.is_main_process:
            processor.save_pretrained(final_model_dir)
    state.wait_for_everyone()

    trainer.asr_metrics.set_expected_rows(len(validation_dataset))
    validation_metrics = trainer.evaluate(
        eval_dataset=validation_dataset,
        metric_key_prefix="validation",
    )
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)

    trainer.asr_metrics.set_expected_rows(len(test_dataset))
    test_output = trainer.predict(test_dataset, metric_key_prefix="test")
    trainer.log_metrics("test", test_output.metrics)
    trainer.save_metrics("test", test_output.metrics)

    if state.is_main_process:
        write_predictions(
            run_dir / "test_predictions.csv",
            test_output,
            processor,
            test_dataset,
            speaker_column,
        )
        summary = {
            "model_name": config.get_pretrained_model_path(),
            "architecture": "ctc",
            "code_revision": config.code_revision,
            "world_size": world_size,
            "optimizer_global_batch_size": global_batch,
            "split_manifest": split_manifest,
            "config": vars(config),
            "runtime_versions": runtime_versions(),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_output.metrics,
        }
        with (run_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
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
        with (run_dir / "metrics_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["split", "wer_percent", "cer_percent", "loss"],
            )
            writer.writeheader()
            writer.writerows(rows)

        LOGGER.info("Evaluation artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()
