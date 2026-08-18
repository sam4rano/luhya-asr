# src/utils/config.py
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import yaml
import datetime


@dataclass
class ASRConfig:
    """Configuration for ASR training and evaluation."""
    # Project settings
    project: str
    output_dir: str
    seed: int
    
    # Model settings
    pretrained_model: str
    experiment_name: Optional[str] = None
    code_revision: Optional[str] = None
    freeze_feature_encoder: bool = True
    report_to: str = "wandb"  # "wandb", "none", "tensorboard", etc.
    
    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 30
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 400
    eval_steps: int = 400
    logging_steps: int = 10
    save_total_limit: int = 2

    # new model config settings
    add_final_layer_adapter: bool = True
    # adapter_kernel_size: int = 3      # optional, default 3
    # adapter_stride: int = 2           # optional, default 2
    # num_adapter_layers: int = 3       # optional, default 3
    # final_dropout: float = 0.0        # optional, default 0.0
    ctc_zero_infinity: bool = True

    # Performance optimizations
    use_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    group_by_length: bool = True
    use_fused_optimizer: bool = False
    persistent_workers: bool = True
    dataloader_num_workers: int = 2
       
    # Step-based training settings
    max_steps: int = -1  # -1 means train for specified epochs, positive value means train for that many steps
    warmup_steps: int = 0  # Specific number of warmup steps (instead of ratio)
    
    # Data settings
    train_split: str = "train"
    eval_split: str = "validation"
    language: str = "all" 
    
    use_custom_dataset: bool = False
    dataset_path: Optional[str] = None
    dataset_revision: str = "main"
    cache_dir: Optional[str] = None
    audio_column: str = "audio"
    text_column: str = "transcript"
    speaker_column: Optional[str] = "user_id"
    sampling_rate: int = 16000
    sample: bool = False
    sample_size: int = 1000
    max_data_hours: float = 0.0  # 0 means no limit; >0 restricts training data to this many hours
    validation_split_pct: float = 0.0  # 0 = use pre-existing eval_split; >0 = split train data into train/val (e.g. 0.2 = 20% val)
    validation_ratio: float = 0.10  # deterministic speaker-disjoint 80/10/10 split (ignored when pre-existing splits are speaker-disjoint)
    test_ratio: float = 0.10
    rebuild_splits_on_speaker_overlap: bool = True
    min_audio_length: float = 0.0  # drop samples shorter than this (seconds); 0 = no minimum
    max_audio_length: float = 30.0  # max individual sample duration in seconds (memory safety; 0 = no limit)
    max_train_hours: float = 40.0  # deterministic cap on the train split's audio; validation/test are not capped
    num_proc: int = 4  # number of CPU processes for dataset preprocessing (lower = less RAM)

    # model vocab settings
    add_language_tokens: bool = False
    character_set: str = "abcdefghijklmnopqrstuvwxyz0123456789 -'"
    apply_accent_replacements: bool = True
    
    # Model mappings
    pretrained_model_map: Dict[str, str] = field(default_factory=lambda: {
        "xlsr-128": "facebook/wav2vec2-xls-r-300m",
        "xlsr-53": "facebook/wav2vec2-large-xlsr-53",
        "mHuBERT-147": "utter-project/mHuBERT-147",
        "w2v-BERT": "facebook/w2v-bert-2.0",
        "mms-300m": "facebook/mms-300m",
        "mms-1b": "facebook/mms-1b-all",
        
    })
    

    def get_pretrained_model_path(self) -> str:
        """Get the actual model path from the model name."""
        return self.pretrained_model_map.get(
            self.pretrained_model, 
            self.pretrained_model
        )
    
    def get_experiment_name(self) -> str:
        """Return a stable configured name, or generate a timestamped fallback."""
        if self.experiment_name:
            return self.experiment_name
        timestamp = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")

        model_name_str = [
            self.pretrained_model.replace("/", "-"),
            timestamp,
        ]

        return f"{'-'.join(model_name_str)}"


def load_config(config_path: str) -> ASRConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return ASRConfig(**config_dict)
