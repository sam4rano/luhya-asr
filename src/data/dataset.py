# src/data/dataset.py
import json
import logging
import os
from dataclasses import asdict
from typing import Dict, Tuple, List, Any, Optional
from datasets import load_dataset, Dataset, Audio, DatasetDict

# for vectorized filtering on large datasets via Arrow
# NOTE: this did not work for some reason; investigate later
import pyarrow.compute as pc


from tqdm import tqdm

from transformers import (
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor, 
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertProcessor,
    Wav2Vec2Processor
)

from src.data.preprocessing import (
    clean_text_batch, 
    prepare_dataset_batch
)
from src.data.splits import prepare_splits
from src.utils.config import ASRConfig

# type alias for processor
ASRProcessor = Wav2Vec2Processor | Wav2Vec2BertProcessor


def _ensure_audio_duration_column(dataset: Dataset, split_name: str) -> Dataset:
    """
    Ensure dataset has an 'audio_duration' column, 
    renaming from 'duration' if present or computing if missing.
    """
    if "audio_duration" in dataset.column_names:
        return dataset
    if "duration" in dataset.column_names:
        return dataset.rename_column("duration", "audio_duration")

    # otherwise, compute audio duration column
    audio_duration_list = []
    for audio in tqdm(
        dataset["audio"],
        total=len(dataset["audio"]),
        desc=f"Calculating audio duration in {split_name} split",
    ):
        try:
            audio_duration_list.append(len(audio["array"]) / audio["sampling_rate"])
        except Exception as e:
            logging.error(f"Error calculating audio duration for audio {audio}: {e}")
            audio_duration_list.append(0.0)
    logging.info(f"Creating audio_duration column in {split_name} split...")

    return dataset.add_column("audio_duration", audio_duration_list)


def _ensure_transcription_column(dataset: Dataset, split_name: str) -> Dataset:
    """
    Ensure dataset has a 'transcription' column, 
    renaming from 'transcript' or 'text' if present.
    """
    if "transcription" in dataset.column_names:
        return dataset
    if "transcript" in dataset.column_names:
        return dataset.rename_column("transcript", "transcription")
    if "text" in dataset.column_names:
        return dataset.rename_column("text", "transcription")
    
    # if no transcription column is found, raise an error
    raise ValueError(
        f"Transcription column was not found in {split_name} split, "
        f"which should be called 'transcript', 'text', or 'transcription'. "
        f"Found columns: {dataset.column_names}."
    )


def _filter_by_max_hours(dataset: Dataset, max_hours: float, seed: int = 42) -> Dataset:
    """Randomly sample a dataset to stay under a maximum number of hours.
    
    Shuffles the dataset and selects samples in order until the cumulative
    audio duration reaches or just exceeds `max_hours`.
    
    Args:
        dataset: Dataset with an 'audio_duration' column
        max_hours: Maximum hours of audio to keep
        seed: Random seed for reproducibility
        
    Returns:
        Filtered dataset with at most `max_hours` of audio
    """
    total_secs = sum(float(d) for d in dataset["audio_duration"])
    total_hours = total_secs / 3600
    logging.info(
        f"Total dataset duration: {total_hours:.2f} hours. "
        f"Filtering to at most {max_hours:.2f} hours..."
    )
    
    if total_hours <= max_hours:
        logging.info("Dataset already under the limit; no filtering applied.")
        return dataset
    
    dataset = dataset.shuffle(seed=seed)
    
    cumulative = 0.0
    keep_indices = []
    for i, duration in enumerate(dataset["audio_duration"]):
        cumulative += float(duration)
        keep_indices.append(i)
        if cumulative / 3600 >= max_hours:
            break
    
    filtered = dataset.select(keep_indices)
    filtered_hours = sum(float(d) for d in filtered["audio_duration"]) / 3600
    logging.info(
        f"After filtering: {len(filtered)} samples, "
        f"{filtered_hours:.2f} hours kept."
    )
    return filtered


def _get_dialect_column(dataset: Dataset) -> Optional[str]:
    """Detect the dialect/language column name in the dataset."""
    if "dialect" in dataset.column_names:
        return "dialect"
    if "language" in dataset.column_names:
        return "language"
    return None


def _make_dialect_filter(dialect_col: str, language: str):
    """Build a module-level filter predicate that is safe to pickle."""
    def _keep(x):
        return str(x[dialect_col]).strip().lower() == language.strip().lower()
    return _keep


def _make_ctc_cleaner(allowed_chars: set[str]):
    """Build a module-level map function that is safe to pickle."""
    def _clean(text: str) -> dict[str, str]:
        return {"clean_transcription": "".join(c for c in text if c in allowed_chars)}
    return _clean


def _filter_by_dialect(dataset: Dataset, dialect: str, dialect_col: str,
                       num_proc: int = 4) -> Dataset:
    """Filter dataset to only include a specific dialect."""
    logging.info(f"Filtering dataset for '{dialect}' dialect (column: '{dialect_col}')...")
    n_before = len(dataset)
    dataset = dataset.filter(
        lambda x: str(x[dialect_col]).strip().lower() == dialect.strip().lower(),
        batch_size=32,
        num_proc=num_proc,
        desc=f"Filtering by dialect '{dialect}'"
    )
    logging.info(f"Dialect filter: {n_before} -> {len(dataset)} samples kept")
    return dataset


def load_datasets(
    config: ASRConfig,
    state: Any = None,
    smoke_test: bool = False,
) -> Tuple[DatasetDict, Optional[str], Dict[str, Any]]:
    """Load and prepare the shared 80/10/10 speaker-disjoint train/validation/test splits.

    Uses the same deterministic split, filtering, and 40-hour cap as the Whisper
    pipeline (``src/data/splits.prepare_splits``) so different models train and
    evaluate on identical clips. Returns ``(splits, speaker_column, manifest)``.
    """
    num_proc = getattr(config, 'num_proc', 4)

    config_dict = asdict(config)
    config_dict.update({
        'speaker_column': getattr(config, 'speaker_column', None) or 'user_id',
        'validation_ratio': float(getattr(config, 'validation_ratio', 0.1)),
        'test_ratio': float(getattr(config, 'test_ratio', 0.1)),
        'rebuild_splits_on_speaker_overlap': bool(
            getattr(config, 'rebuild_splits_on_speaker_overlap', True)
        ),
        'min_audio_length': float(getattr(config, 'min_audio_length', 0.0)),
        'max_audio_length': float(getattr(config, 'max_audio_length', 30.0)),
        'max_train_hours': float(getattr(config, 'max_train_hours', 0.0)),
        'sampling_rate': int(getattr(config, 'sampling_rate', 16000)),
        'lowercase': True,
        'preprocessing_num_proc': num_proc,
        'audio_column': getattr(config, 'audio_column', 'audio'),
        'text_column': getattr(config, 'text_column', 'transcript'),
        'dataset_revision': getattr(config, 'dataset_revision', 'main'),
        'cache_dir': getattr(config, 'cache_dir', None),
        'dataset_path': config.dataset_path,
    })

    prepared, manifest, _audio_col, normalized_col, speaker_column = prepare_splits(
        config_dict,
        state=state,
        smoke_test=smoke_test,
        token=os.environ.get('HF_TOKEN'),
    )

    # Optional dialect filter (config.language defaults to "all" = no filter).
    dialect_col = _get_dialect_column(prepared['train'])
    language = getattr(config, 'language', 'all')
    if language != 'all' and dialect_col:
        logging.info(f"Filtering all splits to dialect '{language}'...")
        keep_dialect = _make_dialect_filter(dialect_col, language)
        for name in prepared:
            prepared[name] = prepared[name].filter(
                keep_dialect,
                num_proc=num_proc,
            )

    # CTC char-set cleaning: keep only characters in the user-provided character
    # set so every target token exists in the CTC vocabulary. This mirrors the
    # Whisper normalization except for characters outside the CTC character set.
    allowed_chars = set(getattr(config, 'character_set', ''))
    clean_ctc = _make_ctc_cleaner(allowed_chars)

    for name, dataset in prepared.items():
        dataset = dataset.map(
            clean_ctc,
            input_columns=[normalized_col],
            num_proc=num_proc,
            desc=f"Building CTC transcriptions for {name}",
        )
        dataset = dataset.remove_columns([normalized_col])
        prepared[name] = dataset

    if getattr(config, 'sample', False):
        size = getattr(config, 'sample_size', 1000)
        logging.info(f"Sampling each split to at most {size} samples...")
        for name in prepared:
            prepared[name] = prepared[name].shuffle(seed=config.seed).select(
                range(min(size, len(prepared[name])))
            )

    for name, dataset in prepared.items():
        durations = [float(d) for d in dataset['audio_duration']]
        logging.info(
            f"Final {name} split: {len(dataset)} samples ({sum(durations)/3600:.2f}h)"
        )

    return prepared, speaker_column, manifest




def add_language_tag_to_transcript(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Add language tags to the beginning of the transcription"""

    lang_key = "language" if "language" in batch else "dialect"

    batch["clean_transcription"] = [
        f"[{lang.upper()}]|{trans}"
        for lang, trans in zip(batch[lang_key], batch["clean_transcription"])
    ]

    return batch


def build_vocabulary(character_set: set[str],
                     add_language_tags: bool = False,
                     language_tags: List[str] = None) -> Dict[str, int]:
    """Build vocabulary from user-provided character set.
    
    Args:
        character_set: Set of characters to include in the vocabulary
        add_language_tags: Whether to add language tags to the vocabulary (only for multilingual models)
        language_tags: List of language tags to add to the vocabulary (only for multilingual models)
        
    Returns:
        Vocabulary dictionary
    """
    # create vocabulary dictionary from the user-provided character set
    vocab_dict = {v: k for k, v in enumerate(sorted(character_set))}

    # handle special tokens
    # add word delimiter token and remove space token
    vocab_dict["|"] = vocab_dict[" "]

    del vocab_dict[" "]

    # add unknown token and padding token
    # find next available index
    next_idx = max(vocab_dict.values()) + 1

    vocab_dict["[UNK]"] = next_idx
    vocab_dict["[PAD]"] = next_idx + 1
    
    if add_language_tags:
        for i, tag in enumerate(language_tags):
            tag_key = f"[{tag.upper()}]"
            vocab_dict[tag_key] = next_idx + 2 + i
    
    return vocab_dict


def create_processor(
        config: ASRConfig, 
        ctc_dir: str) -> ASRProcessor:
    """Create a processor from tokenizer and feature extractor.
    
    Args:
        config: ASR configuration object
        ctc_dir: Path to directory containing CTC tokenizer
        
    Returns:
        Wav2Vec2Processor for processing audio and text
    """
    # initialize tokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        ctc_dir,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    
    # initialize feature extractor
    pretrained_model_path = config.get_pretrained_model_path()
    if "w2v-bert" in pretrained_model_path.lower():
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            pretrained_model_path,
            cache_dir=getattr(config, "cache_dir", None),
        )
        processor = Wav2Vec2BertProcessor(
            feature_extractor=feature_extractor, 
            tokenizer=tokenizer
        )

    else:
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        # combine into processor
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
    
    return processor


def prepare_datasets(train_dataset: Dataset, 
                     eval_dataset: Dataset, 
                     processor: Wav2Vec2Processor) -> Tuple[Dataset, Dataset]:
    """Prepare datasets for training by adding processed inputs.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        processor: Wav2Vec2Processor for processing audio and text
        
    Returns:
        Tuple of prepared (train_dataset, test_dataset)
    """
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset_batch(batch, processor),
        batched=True,
        batch_size=32, # has to be based on available memory
        remove_columns=train_dataset.column_names
    )

    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset_batch(batch, processor),
        batched=True,
        batch_size=32, # has to be based on available memory
        remove_columns=eval_dataset.column_names
    )   
    
    return train_dataset, eval_dataset
