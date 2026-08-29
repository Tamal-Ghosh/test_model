import torch
from torch.utils.data import DataLoader
from utils import get_device

def get_trainable_state_dict(model):
    """
    Extracts a dictionary containing only the parameters that require gradients.
    Tensors are cloned and placed on the CPU to prevent device memory leakages.
    """
    return {name: param.data.clone().cpu() for name, param in model.named_parameters() if param.requires_grad}

def load_trainable_state_dict(model, state_dict):
    """
    Loads parameters from a state dict into the model in-place.
    """
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param.data)
        else:
            print(f"Warning: parameter '{name}' not found in model state dict.")

class Client:
    """
    Represents a single federated learning client.
    Handles local dataset iterations, gradient updates, and communication with the server.
    """
    def __init__(self, client_id: int, dataset, config):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        
    def local_train(self, base_model, global_trainable_weights):
        """
        Runs local training epochs on the client's dataset.
        
        Args:
          base_model: The global model template.
          global_trainable_weights: State dict of the current global trainable parameters.
          
        Returns:
          Dict[str, torch.Tensor]: The updated local trainable weights.
          int: The size of the local dataset.
          float: The average training loss during local epochs.
        """
        # 1. Synchronize the local model with the global trainable parameters
        load_trainable_state_dict(base_model, global_trainable_weights)
        
        # 2. Prepare PyTorch DataLoader
        dataloader = DataLoader(self.dataset, batch_size=self.config.local_batch_size, shuffle=True)
        
        # 3. Define Optimizer (only pass parameters that require gradients)
        trainable_params = [p for p in base_model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError(f"Client {self.client_id} has no trainable parameters. Check PEFT setup.")
            
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.lr)
        
        # 4. Device Placement
        device = get_device()
        base_model.to(device)
        base_model.train()
        
        total_loss = 0.0
        batches = 0
        
        # 5. Training Loop
        for epoch in range(self.config.local_epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Move batch data to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
                
        # 6. Clean up device memory
        base_model.to("cpu")
        
        # 7. Extract the updated local trainable weights
        local_updated_weights = get_trainable_state_dict(base_model)
        
        avg_loss = total_loss / batches if batches > 0 else 0.0
        
        return local_updated_weights, len(self.dataset), avg_loss
