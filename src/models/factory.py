# src/models/factory.py
from typing import Optional, Union
import logging
import torch
from transformers import Wav2Vec2ForCTC, AutoModelForCTC, Wav2Vec2Processor, AutoConfig
from src.utils.config import ASRConfig
from src.models.hubert_with_adapter import HubertForCTCWithAdapter, HubertModel


# def create_asr_model(
#     config: ASRConfig, 
#     processor: Wav2Vec2Processor
# ) -> Union[Wav2Vec2ForCTC, HubertForCTCWithAdapter]:
#     """
#     Create and configure an ASR model.
    
#     Args:
#         config: ASR configuration object
#         processor: Wav2Vec2Processor with tokenizer information
    
#     Returns:
#         Configured model for CTC
#     """
#     pretrained_model_path = config.get_pretrained_model_path()
#     model_config = AutoConfig.from_pretrained(pretrained_model_path)
    
#     # common config settings
#     model_config.attention_dropout = 0.0
#     model_config.hidden_dropout = 0.0
#     model_config.feat_proj_dropout = 0.0
#     model_config.mask_time_prob = 0.0
#     model_config.layerdrop = 0.0
#     model_config.ctc_loss_reduction = "mean"
#     model_config.pad_token_id = processor.tokenizer.pad_token_id
#     model_config.vocab_size = len(processor.tokenizer)
    
#     print(f"model_type: {model_config.model_type}")
    
#     if model_config.model_type == "hubert":
#         if config.add_final_layer_adapter:
#             # use custom hubert with adapter layers
#             model_config.adapter_kernel_size = getattr(config, "adapter_kernel_size", 3)
#             model_config.adapter_stride = getattr(config, "adapter_stride", 2)
#             model_config.num_adapter_layers = getattr(config, "num_adapter_layers", 3)
#             model_config.output_hidden_size = model_config.hidden_size
#             model_config.final_dropout = getattr(config, "final_dropout", 0.0)
#             model_config.ctc_zero_infinity = getattr(config, "ctc_zero_infinity", False)
            
#             # create model with adapter
#             model = HubertForCTCWithAdapter(model_config)
            
#             # load pretrained hubert weights
#             pretrained = HubertModel.from_pretrained(pretrained_model_path)
#             model.hubert.load_state_dict(pretrained.state_dict(), strict=False)
#             del pretrained
#         else:
#             # standard hubert without adapter
#             model = AutoModelForCTC.from_pretrained(
#                 pretrained_model_path,
#                 config=model_config,
#             )
    
#     elif model_config.model_type in ("wav2vec2-bert", "wav2vec2"):
#         # these models natively support add_adapter
#         model_config.add_adapter = config.add_final_layer_adapter
#         model = AutoModelForCTC.from_pretrained(
#             pretrained_model_path,
#             config=model_config,
#         )
    
#     else:
#         # fallback for other model types
#         model = AutoModelForCTC.from_pretrained(
#             pretrained_model_path,
#             config=model_config,
#         )
    
#     return model

# 
#  # src/models/factory.py
# from typing import Optional
# import torch
# from transformers import Wav2Vec2ForCTC, AutoModelForCTC, Wav2Vec2Processor, AutoConfig

# from src.utils.config import ASRConfig


# def create_asr_model(config: ASRConfig, 
#                      processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
#     pretrained_model_path = config.get_pretrained_model_path()
    
#     model_config = AutoConfig.from_pretrained(pretrained_model_path)
#     model_config.attention_dropout = 0.0
#     model_config.hidden_dropout = 0.0
#     model_config.feat_proj_dropout = 0.0
#     model_config.mask_time_prob = 0.0
#     model_config.layerdrop = 0.0
#     model_config.ctc_loss_reduction = "mean"
#     model_config.pad_token_id = processor.tokenizer.pad_token_id
#     model_config.vocab_size = len(processor.tokenizer)
    
#     # add adapter support
#     print(model_config.model_type)
#     if model_config.model_type in ("wav2vec2-bert", "wav2vec2"):
#         model_config.add_adapter = config.add_final_layer_adapter
#     elif model_config.model_type == "hubert":
#         # hubert config is missing these attributes - add them manually
#         model_config.add_adapter = config.add_final_layer_adapter
#         model_config.output_hidden_size = model_config.hidden_size

#     model = AutoModelForCTC.from_pretrained(
#         pretrained_model_path,
#         config=model_config,
#     )
    
#     return model


def create_asr_model(config: ASRConfig, 
                     processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    """
    Create and configure a Wav2Vec2 ASR model.
    
    Args:
        config: ASR configuration object
        processor: Wav2Vec2Processor with tokenizer information
        
    Returns:
        Configured Wav2Vec2ForCTC model
    """
    # Get pretrained model path
    pretrained_model_path = config.get_pretrained_model_path()
    
    # initialize model
    # if the pretrained_model is w2v-bert-2.0, add adapter
    # if pretrained_model_path == "facebook/w2v-bert-2.0":
    #     config.add_final_layer_adapter = True
    # else:
    #     add_final_layer_adapter = False

    # check if the pretrained model is a Hubert model
    if "hubert" in pretrained_model_path.lower():
        model = AutoModelForCTC.from_pretrained(
            pretrained_model_path,
            cache_dir=getattr(config, "cache_dir", None),
            attention_dropout=0.00,
            hidden_dropout=0.00,
            feat_proj_dropout=0.00,
            mask_time_prob=0.00,
            layerdrop=0.00,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
    else:
        model = AutoModelForCTC.from_pretrained(
            pretrained_model_path,
            cache_dir=getattr(config, "cache_dir", None),
            attention_dropout=0.00,
            hidden_dropout=0.00,
            feat_proj_dropout=0.00,
            mask_time_prob=0.00,
            layerdrop=0.00,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            add_adapter=getattr(config, "add_final_layer_adapter", False),  
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )

    # apply freezing configuration
    # Wav2Vec2-BERT does not expose `freeze_feature_encoder`; freezing the base
    # model would freeze the whole encoder (including the CTC head inputs), so we
    # warn instead of silently ignoring the flag.
    if config.freeze_feature_encoder and hasattr(model, 'freeze_feature_encoder'):
        model.freeze_feature_encoder()
    elif config.freeze_feature_encoder:
        logging.warning(
            "freeze_feature_encoder=True is not supported for %s "
            "(no freeze_feature_encoder method); feature encoder will NOT be frozen",
            pretrained_model_path,
        )

    # torch.compile for faster training (PyTorch 2.0+)
    if getattr(config, "use_torch_compile", False):
        compile_mode = getattr(config, "torch_compile_mode", "reduce-overhead")
        if hasattr(torch, "compile"):
            logging.info(f"Applying torch.compile with mode='{compile_mode}'")
            model = torch.compile(model, mode=compile_mode)
        else:
            logging.warning("torch.compile not available (requires PyTorch >= 2.0)")

    return model

