import torch
import torch.nn as nn
from peft import get_peft_model, PrefixTuningConfig, TaskType

def configure_prefix(model: nn.Module, config) -> nn.Module:
    """
    Configures Prefix Tuning (FedPF) on roberta-base.
    Uses prefix_projection=False to enable linear, direct parameter scaling with virtual tokens.
    """
    num_tokens = getattr(config, "prefix_num_virtual_tokens", 16)
    prefix_proj = getattr(config, "prefix_projection", False)
    print(f"[PEFT - Prefix] Configuring HF PrefixTuning (virtual_tokens={num_tokens}, projection={prefix_proj})...")
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=num_tokens,
        prefix_projection=prefix_proj
    )
    model = get_peft_model(model, peft_config)
    return model
