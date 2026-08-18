# src/training/collator.py
from dataclasses import dataclass
from typing import Dict, List, Union
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2BertProcessor

from src.data.splits import _audio_samples

ASRProcessor = Union[Wav2Vec2Processor, Wav2Vec2BertProcessor]

class DataCollatorCTCWithPadding:
    """
    Data collator that dynamically pads inputs for CTC training.
    """
    def __init__(self, processor: ASRProcessor, padding: Union[bool, str] = True):
        """
        Initialize the data collator.
        """
        self.processor = processor
        self.padding = padding
        self._is_w2vbert = isinstance(processor, Wav2Vec2BertProcessor)
        self._input_key = "input_features" if self._is_w2vbert else "input_values"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate and pad a batch of examples.
        Args:
            features: List of feature dictionaries, 
            each dictionary contains the input features and labels
        Returns:
            Batch dictionary with padded tensors for the input features and labels
        """
        # pad input features
        input_features = [{self._input_key: f[self._input_key]} for f in features
        ]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # get label ids and pad them
        label_ids = [{ "input_ids": f["labels"]} for f in features]

        labels_batch = self.processor.tokenizer.pad(
            label_ids,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        return batch


class DataCollatorCTCRawAudioWithPadding:
    """Compute CTC audio features per batch instead of caching the full 40h set.

    Keeping raw audio in the Dataset avoids materializing several gigabytes of
    mel/input features in Arrow before training. The feature extractor and
    tokenizer are still padded independently, as required for CTC.
    """

    def __init__(
        self,
        processor: ASRProcessor,
        audio_column: str = "audio",
        text_column: str = "clean_transcription",
        sampling_rate: int = 16_000,
    ):
        self.processor = processor
        self.audio_column = audio_column
        self.text_column = text_column
        self.sampling_rate = sampling_rate

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        arrays = []
        texts = []
        for feature in features:
            array, sample_rate = _audio_samples(feature[self.audio_column])
            if sample_rate != self.sampling_rate:
                raise ValueError(
                    f"Expected {self.sampling_rate} Hz after dataset casting, "
                    f"got {sample_rate} Hz"
                )
            arrays.append(array)
            texts.append(feature[self.text_column])

        batch = self.processor(
            audio=arrays,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        return batch


class DataCollatorCTCAndLIDWithPadding(DataCollatorCTCWithPadding):
    """Data collator for CTC + LID multi-task training."""
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate and pad a batch of examples for CTC + LID multi-task training.
        Args:
            features: List of feature dictionaries, 
            each dictionary contains the input features and labels
            
        Returns:
            Batch dictionary with padded tensors for the input features and labels
        """
        batch = super().__call__(features)
        batch["lid_labels"] = torch.tensor(
            [f["lid_labels"] for f in features],
            dtype=torch.long,
        )
        return batch
