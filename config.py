from dataclasses import dataclass, field
from typing import List

@dataclass
class FLConfig:
    # Model configuration
    model_name: str = "toy"  # Options: "toy", "roberta-base", "distilbert-base-uncased", etc.
    num_labels: int = 2
    max_length: int = 128
    
    # Federated configurations
    num_clients: int = 5
    fraction_fit: float = 0.6  # Fraction of clients trained per round
    num_rounds: int = 3
    local_epochs: int = 1
    local_batch_size: int = 16
    lr: float = 1e-4
    
    # PEFT settings
    # Options: "none" (FedFT), "adapter" (FedAP), "prefix" (FedPF), "lora" (FedLR), "bitfit" (FedBF)
    peft_method: str = "lora"
    
    # PEFT hyperparameters
    # LoRA (FedLR)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    
    # Prefix Tuning (FedPF)
    prefix_num_virtual_tokens: int = 20
    
    # Adapter (FedAP)
    adapter_bottleneck_dim: int = 64
    
    # Partition settings
    partition_type: str = "non_iid"  # "iid" or "non_iid"
    alpha: float = 0.5  # Dirichlet parameter for non-iid partition
