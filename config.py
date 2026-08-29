from dataclasses import dataclass, field
from typing import List

@dataclass
class FLConfig:
    # Model Configurations
    # distilroberta-base (82M params) fits in 2GB VRAM and is 100% compatible with all PEFT methods.
    model_name: str = "distilroberta-base"
    num_labels: int = 2
    max_length: int = 64              
    
    # Dataset Configuration
    dataset_name: str = "sst2"
    
    # Optional Sample Subsampling for resource-constrained devices
    max_train_samples: int = None
    max_val_samples: int = None
    
    # Federated Learning Parameters (Optimized for 2GB VRAM & Laptop execution)
    num_clients: int = 5              
    fraction_fit: float = 0.6         # 3 active clients per round (5 * 0.6)
    num_rounds: int = 5               
    local_epochs: int = 1             
    local_batch_size: int = 4         # Reduced from 16 to fit in 2GB VRAM
    lr: float = 1e-3                  
    seed: int = 42                    
    
    # Non-IID Parameter (Dirichlet)
    dirichlet_alpha: float = 1.0      
    
    # PEFT Method Selection
    # Options: "none" (FedFT), "adapter" (FedAP), "prefix" (FedPF), "lora" (FedLR), "bitfit" (FedBF)
    peft_method: str = "lora"
    
    # PEFT Specific Hyperparameters
    # LoRA (FedLR)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    
    # Prefix Tuning (FedPF)
    prefix_num_virtual_tokens: int = 16
    
    # Adapter (FedAP)
    adapter_reduction_factor: int = 16  
    
    # Privacy / Gradient Inversion Parameters
    privacy_num_samples: int = 4      
    privacy_attack_steps: int = 100   
    
    def apply_defaults(self):
        """
        Keeps the optimized configurations intact for this hardware environment.
        Adjusts learning rate dynamically.
        """





































































































































        if self.peft_method.lower() == "none":
            self.lr = 1e-4
        else:
            self.lr = 1e-3
