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
        
    def _evaluate_on_loader(self, dataloader):
        device = get_device()
        self.model.to(device)
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        self.model.to("cpu")
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if len(all_labels) > 0 else 0.0
        return avg_loss, acc, f1

    def evaluate(self):
        """
        Evaluates the global model on the validation dataset.
        Computes validation loss, accuracy, and Macro-F1 score.
        """
        dataloader = DataLoader(self.val_dataset, batch_size=self.config.local_batch_size, shuffle=False)
        return self._evaluate_on_loader(dataloader)

    def evaluate_clients(self):
        """
        Evaluates the current global model on each client's partition.
        Returns client-wise {client_id: {'loss': ..., 'acc': ..., 'f1': ...}}
        """
        client_stats = {}
        for client in self.clients:
            dataloader = DataLoader(client.dataset, batch_size=self.config.local_batch_size, shuffle=False)
            loss, acc, f1 = self._evaluate_on_loader(dataloader)
            client_stats[client.client_id] = {"loss": loss, "acc": acc, "f1": f1}
        return client_stats
        
    def fit_round(self, round_idx: int) -> dict:
        """
        Selects active clients (all 10 in our 100% participation setup),
        conducts local training sequentially, aggregates parameter updates using FedAvg,
        and returns round-level training statistics.
        """
        # 1. Client Sampling (without replacement)
        num_sampled = max(1, int(self.config.num_clients * self.config.fraction_fit))
        sampled_indices = np.random.choice(self.config.num_clients, num_sampled, replace=False)
        sampled_indices = sorted(sampled_indices.tolist())
        
        print(f"\n[Server Round {round_idx}] Active Client IDs: {sampled_indices}", flush=True)
        
        client_updates = []
        client_train_losses = {}
        
        # 2. Sequential local training
        for c_num, idx in enumerate(sampled_indices, 1):
            client = self.clients[idx]
            print(f"  --> Client {c_num}/{len(sampled_indices)} (ID: {idx}, {len(client.dataset)} samples) training...", flush=True)
            updated_weights, size, loss = client.local_train(self.model, self.global_weights)
            client_updates.append((updated_weights, size))
            client_train_losses[idx] = loss
            print(f"      Client {idx} finished. Local Loss: {loss:.4f}", flush=True)
            
        # 3. Federated Averaging (Aggregation)
        print(f"[Server Round {round_idx}] Aggregating parameters from {len(client_updates)} clients...", flush=True)
        self.global_weights = federated_average(client_updates)
        
        # 4. Synchronize the global model with the newly aggregated weights
        load_trainable_state_dict(self.model, self.global_weights)
        
        avg_train_loss = float(np.mean(list(client_train_losses.values())))
        
        return {
            "round": round_idx,
            "avg_train_loss": avg_train_loss,
            "client_train_losses": client_train_losses,
        }
