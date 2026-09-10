import os
from dataclasses import dataclass, field
from typing import List

# Ensure OpenMP runtime conflict does not crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@dataclass
class FLConfig:
    # Model Configuration
    model_name: str = "roberta-base"
    num_labels: int = 4               # AG News has 4 classes: World, Sports, Business, Sci/Tech
    max_length: int = 128             # Standard sequence length
    use_lightweight_head: bool = True # Single linear head (3K params) to respect exact parameter budgets

    # Dataset Configuration
    dataset_name: str = "ag_news"
    max_train_samples: int = None     # None for full dataset, int for smoke testing
    max_val_samples: int = None       # None for full dataset, int for smoke testing

    # Federated Learning Parameters
    num_clients: int = 10             # 10 clients total
    fraction_fit: float = 1.0         # 100% participation: all 10 clients every round
    num_rounds: int = 50              # 50 communication rounds
    local_epochs: int = 1             # 1-2 local epochs
    local_batch_size: int = 16        # Batch size
    lr: float = 1e-3                  # Local learning rate
    seed: int = 42                    # Random seed for reproducibility

    # Non-IID Dirichlet Parameter
    # Experimental tiers: 1.0 (mild), 0.5 (medium), 0.1 (strong), 0.01 (extreme)
    dirichlet_alpha: float = 1.0

    # PEFT Method Selection: "lora", "adapter", "prefix", "ia3", "none"
    peft_method: str = "lora"

    # Parameter Budget Target: 100_000 (Low), 500_000 (Medium), 1_000_000 (High)
    target_budget: int = 100_000

    # Specific Tuners (calibrated in Phase 2)
    # LoRA
    lora_r: int = 3
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value"])

    # Prefix Tuning
    prefix_num_virtual_tokens: int = 16

    # Adapter
    adapter_bottleneck_dim: int = 16

    # IA3
    ia3_target_modules: List[str] = field(default_factory=lambda: ["key", "value", "dense"])
    ia3_feedforward_modules: List[str] = field(default_factory=lambda: ["dense"])
