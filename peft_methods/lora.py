import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

def configure_lora(model: nn.Module, config) -> nn.Module:
    """
    Configures the RoBERTa model with LoRA (FedLR).
    """
    print("[PEFT - LoRA] Configuring HF LoRA for RoBERTa...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules
    )
    model = get_peft_model(model, peft_config)
    return model
