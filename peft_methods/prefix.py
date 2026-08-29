import torch
import torch.nn as nn
from peft import get_peft_model, PrefixTuningConfig, TaskType

def configure_prefix(model: nn.Module, config) -> nn.Module:
    """
    Configures Prefix Tuning (FedPF) on the model.
    """
    print("[PEFT - Prefix] Configuring HF PrefixTuning for model...")
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=config.prefix_num_virtual_tokens
    )
    model = get_peft_model(model, peft_config)
    return model
