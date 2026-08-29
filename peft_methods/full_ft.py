import torch.nn as nn

def configure_full_ft(model: nn.Module, config) -> nn.Module:
    """
    Configures the model for Full Fine-Tuning (FedFT).
    All parameters are set to require gradients.
    """
    print("[PEFT - Full FT] Enabling gradients for all parameters...")
    for param in model.parameters():
        param.requires_grad = True
    return model
