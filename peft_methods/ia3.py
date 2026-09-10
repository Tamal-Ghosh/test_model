import torch
import torch.nn as nn
from peft import get_peft_model, IA3Config, TaskType

def configure_ia3(model: nn.Module, config) -> nn.Module:
    """
    Configures IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) on RoBERTa.
    Targets key, value, and intermediate/output dense projections with element-wise learned vectors.
    """
    target_modules = getattr(config, "ia3_target_modules", ["key", "value", "dense"])
    feedforward_modules = getattr(config, "ia3_feedforward_modules", ["dense"])
    print(f"[PEFT - IA3] Configuring HF IA3 for RoBERTa (targets={target_modules})...")
    
    peft_config = IA3Config(
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
        feedforward_modules=feedforward_modules
    )
    model = get_peft_model(model, peft_config)
    return model
