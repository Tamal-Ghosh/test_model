from dataclasses import dataclass, field
from typing import List

@dataclass
class FLConfig:
    # Model Configuration
    # Actual model evaluated in Table 2, 3, 4 of the paper
    model_name: str = "roberta-base"
    num_labels: int = 2
    max_length: int = 128             # Paper default: 128
    
    # Dataset Configuration
    # Options: "sst2", "rte", "mrpc", "qnli", "qqp", "mnli"
    dataset_name: str = "sst2"
    
    # Optional sample limiting (default None = full dataset)
    max_train_samples: int = None
    max_val_samples: int = None
    
    # Federated Learning Parameters (Paper setup)
    num_clients: int = 10             # Cross-silo: 10, Cross-device: 1000
    fraction_fit: float = 1.0         # Cross-silo: 10/10 active, Cross-device: 10/1000 active
    num_rounds: int = 100             # Paper default: 100 communication rounds
    local_epochs: int = 1             # Paper default: 1 local epoch
    local_batch_size: int = 16        # Paper default: 16
    lr: float = 1e-3                  # Default learning rate
    seed: int = 42                    # Reproducibility seed
    
    # Non-IID Parameter (Dirichlet)
    dirichlet_alpha: float = 1.0      # Paper default: 1.0 (also tested 0.1 and 10.0)
    
    # PEFT Method Selection
    # Options: "none" (FedFT), "adapter" (FedAP), "prefix" (FedPF), "lora" (FedLR), "bitfit" (FedBF)
    peft_method: str = "lora"
    
    # PEFT Specific Hyperparameters (Paper setup)
    # LoRA (FedLR)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    
    # Prefix Tuning (FedPF)
    prefix_num_virtual_tokens: int = 16
    
    # Adapter (FedAP)
    adapter_reduction_factor: int = 16  # Bottleneck dimension: 768 / 16 = 48
    
    # Privacy / Gradient Inversion Parameters
    privacy_num_samples: int = 128    # Paper evaluated on 128 samples
    privacy_attack_steps: int = 1000  # DLG optimization steps
    
    def apply_defaults(self):
        """
        Dynamically configures client scaling and learning rates according to the paper:
        - Cross-silo (SST-2, RTE, MRPC): 10 total clients, 10 active per round.
        - Cross-device (QNLI, QQP, MNLI): 1000 total clients, 10 active per round.
        """
        if self.dataset_name.lower() in ["rte", "mrpc", "sst2"]:
            self.num_clients = 10
            self.fraction_fit = 1.0  # 10 active out of 10
        else:
            self.num_clients = 1000
            self.fraction_fit = 0.01 # 10 active out of 1000
            
        # Paper optimal learning rate assignments
        if self.peft_method.lower() == "none":
            self.lr = 1e-4
        else:
            self.lr = 1e-3
