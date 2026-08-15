# src/data/dataset.py
import json
import logging
from typing import Dict, Tuple, List, Any, Optional
from datasets import load_dataset, Dataset, Audio, DatasetDict

from datasets import disable_caching
disable_caching()

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


def load_datasets(config: ASRConfig) -> Tuple[Dataset, Dataset]:
    """Load and prepare datasets for training and evaluation.
    
    Args:
        config: Configuration object containing dataset parameters
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    num_proc = getattr(config, 'num_proc', 4)

    if hasattr(config, 'use_custom_dataset') and config.use_custom_dataset:
        if hasattr(config, 'dataset_path') and config.dataset_path:
            logging.info(f"Loading custom training dataset locally from "
                         f"{config.dataset_path}...")
            dataset = DatasetDict.load_from_disk(config.dataset_path)
        else:
            raise ValueError(f"dataset_path to a local dataset must be specified "
                             f"when use_custom_dataset is True")
    else:
        logging.info(f"Loading dataset from HF hub from "
                     f"{config.dataset_path}...")
        dataset = load_dataset(
            config.dataset_path,
            verification_mode="no_checks",
        )

    logging.info(f"Casting audio column to Audio with 16000 Hz sampling rate...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    train_dataset = dataset[config.train_split]

    dialect_col = _get_dialect_column(train_dataset)
    if dialect_col:
        dialects = sorted(set(str(d) for d in train_dataset[dialect_col]))
        logging.info(f"Detected dialect column '{dialect_col}' with values: {dialects}")
    else:
        logging.info("No dialect/language column found in dataset")

    if config.language != "all" and dialect_col:
        train_dataset = _filter_by_dialect(train_dataset, config.language,
                                           dialect_col, num_proc)

    train_dataset = _ensure_transcription_column(train_dataset, "train")
    train_dataset = _ensure_audio_duration_column(train_dataset, "train")

    validation_split_pct = getattr(config, 'validation_split_pct', 0.0)
    if validation_split_pct > 0:
        logging.info(f"Splitting train data: {1 - validation_split_pct:.0%} train, "
                     f"{validation_split_pct:.0%} validation")
        train_dataset = train_dataset.shuffle(seed=config.seed)
        split_point = int(len(train_dataset) * (1 - validation_split_pct))
        dev_dataset = train_dataset.select(range(split_point, len(train_dataset)))
        train_dataset = train_dataset.select(range(split_point))
        logging.info(f"Split: {len(train_dataset)} train samples, "
                     f"{len(dev_dataset)} validation samples")
    else:
        logging.info(f"Using pre-existing eval split: '{config.eval_split}'")
        dev_dataset = dataset[config.eval_split]
        dev_dataset = _ensure_transcription_column(dev_dataset, "validation")
        dev_dataset = _ensure_audio_duration_column(dev_dataset, "validation")

    if config.sample:
        logging.info(f"Sampling dataset to {config.sample_size} samples...")
        train_dataset = train_dataset.shuffle(seed=config.seed)
        train_dataset = train_dataset.select(range(config.sample_size))
        dev_dataset = dev_dataset.shuffle(seed=config.seed)
        dev_dataset = dev_dataset.select(range(min(2000, len(dev_dataset))))

    if hasattr(config, 'max_data_hours') and config.max_data_hours > 0:
        train_dataset = _filter_by_max_hours(
            train_dataset, config.max_data_hours, config.seed
        )

    eval_dialect = getattr(config, 'eval_dialect', 'all')
    if eval_dialect != "all" and dialect_col:
        dev_dataset = _filter_by_dialect(dev_dataset, eval_dialect,
                                         dialect_col, num_proc)
        if len(dev_dataset) == 0:
            logging.warning(
                f"No samples found for eval dialect '{eval_dialect}'. "
                f"Falling back to unfiltered eval set."
            )
            dev_dataset = dataset[config.eval_split if validation_split_pct == 0
                                  else config.train_split]
            if validation_split_pct == 0:
                dev_dataset = _ensure_transcription_column(dev_dataset, "validation")
                dev_dataset = _ensure_audio_duration_column(dev_dataset, "validation")

    max_audio_length = getattr(config, 'max_audio_length', 0.0)
    if max_audio_length > 0:
        logging.info(f"Removing samples longer than {max_audio_length}s (memory safety)...")
        n_before = len(train_dataset)
        train_dataset = train_dataset.filter(
            lambda x: x["audio_duration"] <= max_audio_length,
            num_proc=num_proc,
            desc="Removing long samples in train split"
        )
        logging.info(f"Removed {n_before - len(train_dataset)} long samples from train split")

        n_before = len(dev_dataset)
        dev_dataset = dev_dataset.filter(
            lambda x: x["audio_duration"] <= max_audio_length,
            num_proc=num_proc,
            desc="Removing long samples in validation split"
        )
        logging.info(f"Removed {n_before - len(dev_dataset)} long samples from validation split")

    def _is_valid_audio(x):
        try:
            arr = x["audio"]["array"]
            return arr is not None and len(arr) > 0
        except Exception:
            return False

    n_before = len(train_dataset)
    train_dataset = train_dataset.filter(_is_valid_audio, num_proc=num_proc,
                                         desc="Removing corrupt audio from train split")
    logging.info(f"Removed {n_before - len(train_dataset)} corrupt audio samples from train split")

    n_before = len(dev_dataset)
    dev_dataset = dev_dataset.filter(_is_valid_audio, num_proc=num_proc,
                                     desc="Removing corrupt audio from validation split")
    logging.info(f"Removed {n_before - len(dev_dataset)} corrupt audio samples from dev split")

    if "audio_filepath" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("audio_filepath", "audio")
    elif "audio" in train_dataset.column_names:
        pass
    else:
        raise ValueError(f"Audio filepath column was not found in train dataset,"
                         f"which should be called 'audio_filepath' or 'audio'."
                         f"Found columns: {train_dataset.column_names}.")

    if "audio_filepath" in dev_dataset.column_names:
        dev_dataset = dev_dataset.rename_column("audio_filepath", "audio")
    elif "audio" in dev_dataset.column_names:
        pass
    else:
        raise ValueError(f"Audio filepath column was not found in validation dataset,"
                         f"which should be called 'audio_filepath' or 'audio'."
                         f"Found columns: {dev_dataset.column_names}.")

    logging.info(f"Removing unnecessary columns...")
    features_to_keep = [
        "audio", "transcription", "audio_duration",
    ]
    if dialect_col:
        features_to_keep.append(dialect_col)

    features_to_remove = [f for f in train_dataset.features if f not in features_to_keep]
    train_dataset = train_dataset.remove_columns(features_to_remove)
    dev_dataset = dev_dataset.remove_columns(features_to_remove)

    logging.info(f"Preprocessing text transcripts...")
    train_dataset = train_dataset.map(
        lambda batch: clean_text_batch(batch, config.character_set, config.apply_accent_replacements),
        batched=True,
        batch_size=64,
        num_proc=num_proc,
        desc="Cleaning text transcripts in train split"
    )
    dev_dataset = dev_dataset.map(
        lambda batch: clean_text_batch(batch, config.character_set, config.apply_accent_replacements),
        batched=True,
        batch_size=64,
        num_proc=num_proc,
        desc="Cleaning text transcripts in validation split"
    )

    if config.add_language_tokens and config.language == "all":
        train_dataset = train_dataset.map(
            lambda batch: add_language_tag_to_transcript(batch),
            batched=True,
            batch_size=16,
            num_proc=num_proc,
            desc="Adding language tags to train transcriptions"
        )

        dev_dataset = dev_dataset.map(
            lambda batch: add_language_tag_to_transcript(batch),
            batched=True,
            batch_size=16,
            num_proc=num_proc,
            desc="Adding language tags to validation transcriptions"
        )

    total_train_secs = sum(float(d) for d in train_dataset["audio_duration"])
    total_eval_secs = sum(float(d) for d in dev_dataset["audio_duration"])
    logging.info(f"Final datasets: train={len(train_dataset)} samples "
                 f"({total_train_secs/3600:.2f}h), "
                 f"eval={len(dev_dataset)} samples "
                 f"({total_eval_secs/3600:.2f}h)")

    return train_dataset, dev_dataset


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
            pretrained_model_path
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
