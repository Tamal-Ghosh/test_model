import torch
import torch.nn as nn

class RobertaAdapterWrapper(nn.Module):
    """
    Houlsby-style bottleneck adapter wrapper for RoBERTa.
    Inserted after self-attention output and intermediate feedforward layer outputs.
    """
    def __init__(self, original_module: nn.Module, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.original_module = original_module
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        
        # Identity initialization: initialize up projection weights/bias to 0
        # so that adapter acts as an identity function at step 0.
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
    def forward(self, hidden_states, input_tensor):
        # 1. Forward through RoBERTa's original module (Dense + Dropout + LayerNorm)
        x = self.original_module(hidden_states, input_tensor)
        # 2. Bottleneck down-projection -> Activation -> Up-projection
        a = self.up(self.act(self.down(x)))
        # 3. Residual connection
        return x + a

def configure_adapter(model: nn.Module, config) -> nn.Module:
    """
    Freezes all RoBERTa base model parameters and injects Houlsby bottleneck adapters.
    """
    print("[PEFT - Adapter] Freezing roberta-base and injecting Houlsby bottleneck adapters...")
    
    # 1. Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False
        
    hidden_dim = model.config.hidden_size # 768 for roberta-base
    if hasattr(config, "adapter_bottleneck_dim") and config.adapter_bottleneck_dim is not None:
        bottleneck_dim = config.adapter_bottleneck_dim
    else:
        bottleneck_dim = hidden_dim // getattr(config, "adapter_reduction_factor", 16)
    
    # 2. Inject adapters after attention output and feedforward output in every RoBERTa encoder layer
    for i in range(len(model.roberta.encoder.layer)):
        layer = model.roberta.encoder.layer[i]
        layer.attention.output = RobertaAdapterWrapper(layer.attention.output, hidden_dim, bottleneck_dim)
        layer.output = RobertaAdapterWrapper(layer.output, hidden_dim, bottleneck_dim)
        
    # 3. Keep sequence classification head trainable
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
                
    # 4. Unfreeze adapter parameters
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
            
    return model
