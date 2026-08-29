import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class ClientDataset(Dataset):
    """
    A lightweight PyTorch Dataset wrapper that holds tokenized tensors
    representing input ids, attention masks, and labels.
    """
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def get_glue_dataset(config):
    """
    Loads and tokenizes the GLUE dataset specified in config.
    Returns (train_dataset, val_dataset).
    """
    # Mapping GLUE tasks to their text fields
    task_to_keys = {
        "sst2": ("sentence", None),
        "rte": ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "mnli": ("premise", "hypothesis"),
    }
    
    task = config.dataset_name.lower()
    if task not in task_to_keys:
        raise ValueError(f"Unsupported dataset name: {config.dataset_name}. Must be in {list(task_to_keys.keys())}")
        
    print(f"[Dataset] Downloading GLUE task '{task}' from Hugging Face...")
    
    # Load HuggingFace dataset
    raw_datasets = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    sentence1_key, sentence2_key = task_to_keys[task]
    
    def tokenize_function(examples):
        # Tokenize single sentence or sentence pairs
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding="max_length", truncation=True, max_length=config.max_length)
        else:
            return tokenizer(examples[sentence1_key], examples[sentence2_key], padding="max_length", truncation=True, max_length=config.max_length)
            
    print(f"[Dataset] Tokenizing '{task}' split datasets...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names)
    
    # For MNLI, validate on validation_matched split. For others, use validation split.
    val_split = "validation_matched" if task == "mnli" else "validation"
    
    train_dataset = ClientDataset(
        input_ids=torch.tensor(tokenized_datasets["train"]["input_ids"]),
        attention_mask=torch.tensor(tokenized_datasets["train"]["attention_mask"]),
        labels=torch.tensor(raw_datasets["train"]["label"])
    )
    
    val_dataset = ClientDataset(
        input_ids=torch.tensor(tokenized_datasets[val_split]["input_ids"]),
        attention_mask=torch.tensor(tokenized_datasets[val_split]["attention_mask"]),
        labels=torch.tensor(raw_datasets[val_split]["label"])
    )
    
    print(f"[Dataset] Preprocessing complete. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_dataset, val_dataset
