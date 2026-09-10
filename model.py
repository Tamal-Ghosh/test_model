# Kaggle/Colab Fix: Bypass torchao version incompatibility bug in PEFT
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

# Import configurations from peft_methods module
from peft_methods.full_ft import configure_full_ft
from peft_methods.adapter import configure_adapter
from peft_methods.prefix import configure_prefix
from peft_methods.lora import configure_lora
from peft_methods.bitfit import configure_bitfit

class RobertaLightweightClassificationHead(nn.Module):
    """
    Single linear classification head for RoBERTa.
    Replaces the 593K-parameter dense projection with a direct linear projection:
    768 -> num_labels (3,076 parameters for 4 classes).
    This allows low parameter budgets (~100K) to be strictly respected.
    """
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, features, **kwargs):
        # RoBERTa takes features[:, 0, :] (the <s> token)
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def get_base_model(config):
    """
    Initializes and returns the base model (before applying any PEFT method).
    """
    print(f"[Model] Loading pretrained {config.model_name} from Hugging Face...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels
    )
    if getattr(config, "use_lightweight_head", True):
        print(f"[Model] Attaching lightweight classification head (768 -> {config.num_labels})...")
        hidden_size = model.config.hidden_size
        model.classifier = RobertaLightweightClassificationHead(
            hidden_size=hidden_size,
            num_labels=config.num_labels
        )
    return model

from peft_methods.ia3 import configure_ia3

def apply_budget_to_config(config, method: str, budget_tier: str):
    """
    Calibrates PEFT hyperparameter knobs to target exact parameter counts:
      - Low    (~100K trainable parameters)
      - Medium (~500K trainable parameters)
      - High   (~1M trainable parameters)
    """
    method = method.lower()
    tier = str(budget_tier).lower()
    config.peft_method = method

    if tier in ["low", "100k", "100000"]:
        config.target_budget = 100_000
        if method == "lora":
            config.lora_r = 3                    # Trainable: 113,668 (~113K)
            config.lora_alpha = 16
        elif method == "adapter":
            config.adapter_bottleneck_dim = 2    # Trainable: 95,284 (~95K)
        elif method == "prefix":
            config.prefix_num_virtual_tokens = 5 # Trainable: 95,236 (~95K)
        elif method == "ia3":
            config.ia3_target_modules = ["key", "value", "dense"] # Trainable: 76,804 (~77K)

    elif tier in ["medium", "500k", "500000"]:
        config.target_budget = 500_000
        if method == "lora":
            config.lora_r = 14                   # Trainable: 519,172 (~519K)
            config.lora_alpha = 32
        elif method == "adapter":
            config.adapter_bottleneck_dim = 13   # Trainable: 501,052 (~501K)
        elif method == "prefix":
            config.prefix_num_virtual_tokens = 27# Trainable: 500,740 (~500K)
        elif method == "ia3":
            # IA3 uses element-wise scaling vectors, so its natural architectural ceiling is ~77K
            config.ia3_target_modules = ["key", "value", "dense"]

    elif tier in ["high", "1m", "1000000"]:
        config.target_budget = 1_000_000
        if method == "lora":
            config.lora_r = 27                   # Trainable: 998,404 (~998K)
            config.lora_alpha = 64
        elif method == "adapter":
            config.adapter_bottleneck_dim = 27   # Trainable: 1,017,484 (~1.01M)
        elif method == "prefix":
            config.prefix_num_virtual_tokens = 54# Trainable: 998,404 (~998K)
        elif method == "ia3":
            config.ia3_target_modules = ["key", "value", "dense"]

    else:
        raise ValueError(f"Unknown budget tier: {budget_tier}. Expected 'low', 'medium', or 'high'.")

    return config

def get_model(config):
    """
    Factory function that loads the base model and configures the requested PEFT method.
    """
    # 1. Load the base model
    model = get_base_model(config)
    
    # 2. Route and apply the chosen parameter-efficient tuning config
    method = config.peft_method.lower()
    if method == "none":
        model = configure_full_ft(model, config)
    elif method == "adapter":
        model = configure_adapter(model, config)
    elif method == "prefix":
        model = configure_prefix(model, config)
    elif method == "lora":
        model = configure_lora(model, config)
    elif method == "ia3":
        model = configure_ia3(model, config)
    elif method == "bitfit":
        model = configure_bitfit(model, config)
    else:
        raise ValueError(f"Unknown PEFT method: {config.peft_method}")
        
    return model

def count_parameters(model):
    """
    Counts and returns:
      - Total parameters in the model
      - Trainable parameters (requires_grad = True)
      - Frozen parameters (requires_grad = False)
      - Trainable percentage
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params) * 100 if total_params > 0 else 0.0
    return total_params, trainable_params, frozen_params, trainable_pct
