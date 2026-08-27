import torch
from torch.utils.data import DataLoader

def get_trainable_state_dict(model):
    """
    Extracts a dictionary of parameters that have requires_grad = True.
    """
    return {name: param.data.clone().cpu() for name, param in model.named_parameters() if param.requires_grad}

def load_trainable_state_dict(model, state_dict):
    """
    Loads trainable weights from a state dict into the model in-place.
    """
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param.data)
        else:
            print(f"Warning: parameter {name} not found in model state dict.")

class Client:
    def __init__(self, client_id, dataset, config):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        
    def local_train(self, base_model, global_trainable_weights):
        """
        Loads the global trainable weights, runs local training, and returns the updated weights.
        """
        # Load current global trainable parameters
        load_trainable_state_dict(base_model, global_trainable_weights)
        
        # Setup DataLoader
        dataloader = DataLoader(self.dataset, batch_size=self.config.local_batch_size, shuffle=True)
        
        # Setup Optimizer: only update parameters that require gradients
        trainable_params = [p for p in base_model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found! Ensure PEFT or Fine-tuning is properly configured.")
            
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.lr)
        
        # Device placement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.to(device)
        base_model.train()
        
        total_loss = 0.0
        
        for epoch in range(self.config.local_epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
        # Bring model back to CPU
        base_model.to("cpu")
        
        # Extract the updated trainable parameters
        updated_weights = get_trainable_state_dict(base_model)
        
        num_batches = len(dataloader)
        avg_loss = total_loss / (num_batches * self.config.local_epochs) if num_batches > 0 else 0
        
        return updated_weights, len(self.dataset), avg_loss
