import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

# Import configurations from peft_methods module
from peft_methods.full_ft import configure_full_ft
from peft_methods.adapter import configure_adapter
from peft_methods.prefix import configure_prefix
from peft_methods.lora import configure_lora
from peft_methods.bitfit import configure_bitfit

def get_base_model(config):
    """
    Initializes and returns the raw base model (before applying any PEFT method).
    """
    print(f"[Model] Loading raw pretrained {config.model_name} from Hugging Face...")
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels)
    return model

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
