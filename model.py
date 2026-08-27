import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, TaskType

@dataclass
class ModelOutput:
    loss: torch.Tensor = None
    logits: torch.Tensor = None

# --- Toy Model Definition ---

class ToyClassifier(nn.Module):
    def __init__(self, vocab_size=1000, hidden_dim=64, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)  # (batch_size, seq_len, hidden_dim)
        x = x.mean(dim=1)  # Mean pooling: (batch_size, hidden_dim)
        for layer in self.encoder:
            x = x + torch.relu(layer(x))
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return ModelOutput(loss=loss, logits=logits)

# --- Custom PEFT Layers for Toy Model ---

class ToyLoraLinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        self.original_linear = original_linear
        # Freeze original linear layer weights/biases
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = alpha / r
        
    def forward(self, x):
        orig_out = self.original_linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return orig_out + lora_out

class ToyAdapter(nn.Module):
    def __init__(self, layer, dim=64, bottleneck=16):
        super().__init__()
        self.layer = layer
        # Freeze layer weights/biases
        for p in self.layer.parameters():
            p.requires_grad = False
            
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, dim)
        
        # Initialize
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
    def forward(self, x):
        h = self.layer(x)
        a = self.up(self.act(self.down(h)))
        return h + a

class ToyPrefixTuning(nn.Module):
    def __init__(self, toy_model, num_prefix_tokens=5, dim=64):
        super().__init__()
        self.toy_model = toy_model
        self.prefix_embeddings = nn.Parameter(torch.randn(num_prefix_tokens, dim) * 0.01)
        
        # Freeze base model parameters
        for p in self.toy_model.parameters():
            p.requires_grad = False
            
        # Keep classification head trainable
        for p in self.toy_model.classifier.parameters():
            p.requires_grad = True
            
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.toy_model.embedding(input_ids)  # (batch, seq, dim)
        batch_size = x.shape[0]
        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([prefix, x], dim=1)  # (batch, num_prefix_tokens + seq, dim)
        x = x.mean(dim=1)
        
        for layer in self.toy_model.encoder:
            x = x + torch.relu(layer(x))
            
        logits = self.toy_model.classifier(x)
        loss = None
        if labels is not None:
            loss = self.toy_model.loss_fn(logits, labels)
            
        return ModelOutput(loss=loss, logits=logits)

# --- Custom Adapter for RoBERTa (FedAP) ---

class RobertaAdapterWrapper(nn.Module):
    def __init__(self, original_module, bottleneck_dim=64):
        super().__init__()
        self.original_module = original_module
        # Freeze original module parameters
        for p in self.original_module.parameters():
            p.requires_grad = False
            
        in_features = original_module.dense.out_features
        self.down = nn.Linear(in_features, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, in_features)
        
        # Identity initialization at startup
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
    def forward(self, hidden_states, input_tensor):
        x = self.original_module(hidden_states, input_tensor)
        a = self.up(self.act(self.down(x)))
        return x + a

def inject_roberta_adapters(model, bottleneck_dim=64):
    """
    Injects bottleneck adapters after attention output and output dense layers in RoBERTa encoder blocks.
    """
    for i in range(len(model.roberta.encoder.layer)):
        layer = model.roberta.encoder.layer[i]
        layer.attention.output = RobertaAdapterWrapper(layer.attention.output, bottleneck_dim)
        layer.output = RobertaAdapterWrapper(layer.output, bottleneck_dim)

# --- Main Model Factory ---

def get_model(config):
    """
    Loads and configures the model according to the PEFT method.
    """
    if config.model_name == "toy":
        model = ToyClassifier(num_labels=config.num_labels)
        
        if config.peft_method == "none":
            # All parameters trainable
            pass
        elif config.peft_method == "lora":
            # Wrap layers in LoRA
            model.embedding.weight.requires_grad = False
            for i in range(len(model.encoder)):
                model.encoder[i] = ToyLoraLinear(model.encoder[i], r=config.lora_r, alpha=config.lora_alpha)
        elif config.peft_method == "adapter":
            # Wrap layers in Adapter
            model.embedding.weight.requires_grad = False
            for i in range(len(model.encoder)):
                model.encoder[i] = ToyAdapter(model.encoder[i], bottleneck=config.adapter_bottleneck_dim)
        elif config.peft_method == "prefix":
            # Wrap using prefix tuning wrapper
            model = ToyPrefixTuning(model, num_prefix_tokens=config.prefix_num_virtual_tokens)
        elif config.peft_method == "bitfit":
            # Only biases and classifier trainable
            for name, param in model.named_parameters():
                if "bias" in name or "classifier" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
    else:
        # Load HuggingFace Model
        print(f"Loading pre-trained language model: {config.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels)
        
        if config.peft_method == "none":
            # FedFT (Full Fine Tuning): All parameters trainable
            pass
        elif config.peft_method == "lora":
            # FedLR
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules
            )
            model = get_peft_model(model, peft_config)
        elif config.peft_method == "prefix":
            # FedPF
            peft_config = PrefixTuningConfig(
                task_type=TaskType.SEQ_CLS,
                num_virtual_tokens=config.prefix_num_virtual_tokens
            )
            model = get_peft_model(model, peft_config)
        elif config.peft_method == "adapter":
            # FedAP (Bottleneck Adapters)
            inject_roberta_adapters(model, config.adapter_bottleneck_dim)
            # Freeze everything except adapters and classifier head
            for name, param in model.named_parameters():
                if "adapter" in name or "classifier" in name or "score" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif config.peft_method == "bitfit":
            # FedBF
            for name, param in model.named_parameters():
                if "bias" in name or "classifier" in name or "score" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
    return model

def count_parameters(model):
    """
    Returns total parameters, trainable parameters, and percentage trainable.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct_trainable = (trainable_params / total_params) * 100 if total_params > 0 else 0
    return total_params, trainable_params, pct_trainable
