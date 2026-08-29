import torch.nn as nn

def configure_bitfit(model: nn.Module, config) -> nn.Module:
    """
    Configures BitFit (FedBF) on the model.
    Freezes all weight matrices and keeps only the bias tensors and the classification head trainable.
    """
    print("[PEFT - BitFit] Freezing all weights. Keeping biases and classifier head trainable...")
    
    for name, param in model.named_parameters():
        # Unfreeze if parameter name contains 'bias' or is part of the classification head
        if "bias" in name or "classifier" in name or "score" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    return model
