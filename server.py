import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score

from client import Client, get_trainable_state_dict, load_trainable_state_dict
from federated import federated_average
from utils import get_device

class Server:
    """
    Coordinates the global server actions.
    Responsible for selecting active clients, aggregating parameter updates, 
    and evaluating the global model on the validation dataset.
    """
    def __init__(self, model, train_dataset, val_dataset, client_indices, config):
        self.model = model
        self.val_dataset = val_dataset
        self.config = config
        
        # 1. Instantiate the clients and distribute their partitioned datasets
        self.clients = []
        for k in range(config.num_clients):
            # Take client indices subset of train dataset
            client_subset = Subset(train_dataset, client_indices[k])
            self.clients.append(Client(client_id=k, dataset=client_subset, config=config))
            
        # 2. Extract initial global trainable weights
        self.global_weights = get_trainable_state_dict(self.model)
        
    def evaluate(self):
        """
        Evaluates the global model on the validation dataset.
        Computes validation loss, accuracy, and F1 score.
        """
        device = get_device()
        self.model.to(device)
        self.model.eval()
        
        dataloader = DataLoader(self.val_dataset, batch_size=self.config.local_batch_size, shuffle=False)
        total_loss = 0.0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Send model back to CPU to save device memory
        self.model.to("cpu")
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        # Calculate accuracy and F1 score
        acc = accuracy_score(all_labels, all_preds)
        
        # If binary classification, compute binary F1. For multi-class, use macro F1
        unique_labels = np.unique(all_labels)
        if len(unique_labels) <= 2:
            f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        else:
            f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            
        return avg_loss, acc, f1
        
    def fit_round(self, round_idx: int) -> float:
        """
        Selects a random fraction of clients, conducts local training, 
        and updates the global weights using FedAvg.
        """
        # 1. Client Sampling (without replacement)
        num_sampled = max(1, int(self.config.num_clients * self.config.fraction_fit))
        sampled_indices = np.random.choice(self.config.num_clients, num_sampled, replace=False)
        
        print(f"\n[Server Round {round_idx}] Active Client IDs: {sampled_indices.tolist()}")
        
        client_updates = []
        local_losses = []
        
        # 2. Sequential training of clients (sequential simulation of parallel training)
        for idx in sampled_indices:
            client = self.clients[idx]
            # Instruct client to perform local training
            updated_weights, size, loss = client.local_train(self.model, self.global_weights)
            
            client_updates.append((updated_weights, size))
            local_losses.append(loss)
            
        # 3. Federated Averaging (Aggregation)
        print(f"[Server Round {round_idx}] Aggregating parameters from {len(client_updates)} clients...")
        self.global_weights = federated_average(client_updates)
        
        # 4. Synchronize the global model with the newly aggregated weights
        load_trainable_state_dict(self.model, self.global_weights)
        
        avg_local_loss = np.mean(local_losses)
        return avg_local_loss
