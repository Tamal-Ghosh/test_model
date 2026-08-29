import torch
import torch.nn as nn

class GenericAdapterWrapper(nn.Module):
    """
    A model-agnostic bottleneck adapter wrapper.
    Can wrap any sub-module (attention output, linear layers, FFN) and handles
    both tensor outputs and tuple outputs gracefully.
    """
    def __init__(self, original_module: nn.Module, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.original_module = original_module
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        
        # Identity initialization
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
    def forward(self, *args, **kwargs):
        # 1. Run the original module forward
        x = self.original_module(*args, **kwargs)
        
        # 2. Extract hidden states (handles tuple outputs like (hidden_states, attentions))
        if isinstance(x, tuple):
            hidden_states = x[0]
            a = self.up(self.act(self.down(hidden_states)))
            return (hidden_states + a,) + x[1:]
        else:
            a = self.up(self.act(self.down(x)))
            return x + a

def configure_adapter(model: nn.Module, config) -> nn.Module:
    """
    Freezes the base model parameters and injects adapter layers.
    Detects model architecture (RoBERTa or ALBERT) and applies wrappers.
    """
    print("[PEFT - Adapter] Freezing base model and injecting adapters...")
    
    # 1. Freeze all base parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    hidden_dim = model.config.hidden_size
    bottleneck_dim = hidden_dim // config.adapter_reduction_factor
    
    # 2. Add adapters depending on model family
    if hasattr(model, "roberta"):
        print("[PEFT - Adapter] Wrapping RoBERTa encoder submodules...")
        for i in range(len(model.roberta.encoder.layer)):
            layer = model.roberta.encoder.layer[i]
            layer.attention.output = GenericAdapterWrapper(layer.attention.output, hidden_dim, bottleneck_dim)
            layer.output = GenericAdapterWrapper(layer.output, hidden_dim, bottleneck_dim)
            
    elif hasattr(model, "albert"):
        print("[PEFT - Adapter] Wrapping ALBERT shared encoder submodules...")
        for group in model.albert.encoder.albert_layer_groups:
            for layer in group.albert_layers:
                layer.attention = GenericAdapterWrapper(layer.attention, hidden_dim, bottleneck_dim)
                layer.ffn_output = GenericAdapterWrapper(layer.ffn_output, hidden_dim, bottleneck_dim)
                
    else:
        # Generic fallback: print warning and raise error if no match
        raise NotImplementedError("Adapter injection is only implemented for RoBERTa and ALBERT model architectures.")
        
    # 3. Keep classification head trainable
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
                
    # 4. Unfreeze adapter parameters
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
            
    return model
