import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from datasets import load_dataset
from transformers import AutoTokenizer

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, max_length=128, num_classes=2):
        self.num_samples = num_samples
        self.max_length = max_length
        # Generate random input_ids, attention_mask, and labels
        self.input_ids = torch.randint(10, 1000, (num_samples, max_length))
        self.attention_mask = torch.ones((num_samples, max_length), dtype=torch.long)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def partition_dataset(dataset, num_clients, partition_type="iid", alpha=0.5):
    """
    Partitions the dataset indices among clients.
    Supports "iid" (equal partition) and "non_iid" (Dirichlet partition on class labels).
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    
    if partition_type == "iid":
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)
        return [c.tolist() for c in client_indices]
        
    elif partition_type == "non_iid":
        # Extract labels from dataset
        if hasattr(dataset, "labels"):
            labels = np.array(dataset.labels)
        elif "labels" in dataset[0]:
            labels = np.array([x["labels"].item() if isinstance(x["labels"], torch.Tensor) else x["labels"] for x in dataset])
        elif "label" in dataset[0]:
            labels = np.array([x["label"].item() if isinstance(x["label"], torch.Tensor) else x["label"] for x in dataset])
        else:
            # Try to fetch from features if Hugging Face Dataset
            try:
                labels = np.array(dataset["label"])
            except Exception:
                try:
                    labels = np.array(dataset["labels"])
                except Exception:
                    raise ValueError("Cannot extract labels from the dataset for partitioning.")
                    
        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(num_clients)]
        
        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            np.random.shuffle(class_indices)
            
            # Dirichlet proportions for class c
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(class_indices)).astype(int)
            
            # Adjustment for rounding errors
            diff = len(class_indices) - proportions.sum()
            for i in range(diff):
                proportions[i % num_clients] += 1
                
            start = 0
            for k in range(num_clients):
                end = start + proportions[k]
                client_indices[k].extend(class_indices[start:end])
                start = end
                
        # Shuffle client indices
        for k in range(num_clients):
            np.random.shuffle(client_indices[k])
            
        return client_indices

def get_datasets(config):
    """
    Loads dataset (SST-2 or Synthetic) and returns (train_dataset, val_dataset, client_partition_indices)
    """
    if config.model_name == "toy":
        print("Using synthetic dataset for toy model...")
        train_dataset = SyntheticDataset(num_samples=600, max_length=config.max_length, num_classes=config.num_labels)
        val_dataset = SyntheticDataset(num_samples=100, max_length=config.max_length, num_classes=config.num_labels)
    else:
        print(f"Loading SST-2 dataset for {config.model_name}...")
        try:
            raw_datasets = load_dataset("glue", "sst2")
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            
            # Add pad_token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            def tokenize_function(examples):
                return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=config.max_length)
                
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
            
            train_dataset = tokenized_datasets["train"]
            val_dataset = tokenized_datasets["validation"]
        except Exception as e:
            print(f"Failed to load SST-2 from Hugging Face: {e}. Falling back to SyntheticDataset.")
            train_dataset = SyntheticDataset(num_samples=600, max_length=config.max_length, num_classes=config.num_labels)
            val_dataset = SyntheticDataset(num_samples=100, max_length=config.max_length, num_classes=config.num_labels)
            
    # Partition indices
    client_indices = partition_dataset(
        train_dataset, 
        num_clients=config.num_clients, 
        partition_type=config.partition_type, 
        alpha=config.alpha
    )
    
    return train_dataset, val_dataset, client_indices
