import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from client import Client, get_trainable_state_dict, load_trainable_state_dict
from federated import federated_average

class Server:
    def __init__(self, model, train_dataset, val_dataset, client_indices, config):
        self.model = model
        self.val_dataset = val_dataset
        self.config = config
        
        # Initialize clients with their specific subset of indices
        self.clients = []
        for k in range(config.num_clients):
            client_data = Subset(train_dataset, client_indices[k])
            self.clients.append(Client(client_id=k, dataset=client_data, config=config))
            
        # Extract initial global trainable weights
        self.global_weights = get_trainable_state_dict(self.model)
        
    def evaluate(self):
        """
        Evaluates the global model on the validation dataset.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        dataloader = DataLoader(self.val_dataset, batch_size=self.config.local_batch_size, shuffle=False)
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                # Check output type and fetch loss/logits
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        # Send model back to CPU
        self.model.to("cpu")
        
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy
        
    def fit_round(self, round_idx):
        """
        Coordinates a single round of federated training.
        """
        # Determine number of clients to select
        num_sampled = max(1, int(self.config.num_clients * self.config.fraction_fit))
        sampled_client_indices = np.random.choice(self.config.num_clients, num_sampled, replace=False)
        
        print(f"\n--- Round {round_idx} ---")
        print(f"Selected clients: {sampled_client_indices.tolist()}")
        
        client_updates = []
        local_losses = []
        
        for client_id in sampled_client_indices:
            client = self.clients[client_id]
            print(f"  Training client {client_id} (samples: {len(client.dataset)})...")
            
            # Perform local training
            updated_weights, size, loss = client.local_train(self.model, self.global_weights)
            client_updates.append((updated_weights, size))
            local_losses.append(loss)
            
            print(f"    Client {client_id} training loss: {loss:.4f}")
            
        # FedAvg Aggregation
        print("  Aggregating updates...")
        self.global_weights = federated_average(client_updates)
        
        # Load aggregated weights back to global model
        load_trainable_state_dict(self.model, self.global_weights)
        
        avg_local_loss = np.mean(local_losses)
        return avg_local_loss
