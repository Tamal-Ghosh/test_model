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

def get_dataset(config):
    """
    Loads and tokenizes the requested dataset (AG News or GLUE).
    Returns (train_dataset, val_dataset).
    """
    task = config.dataset_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if task == "ag_news":
        print("[Dataset] Loading AG News dataset from Hugging Face...")
        raw_datasets = load_dataset("ag_news")
        train_split = raw_datasets["train"]
        val_split = raw_datasets["test"]  # Standard AG News test split for evaluation

        if config.max_train_samples is not None and config.max_train_samples < len(train_split):
            train_split = train_split.shuffle(seed=config.seed).select(range(config.max_train_samples))
        if config.max_val_samples is not None and config.max_val_samples < len(val_split):
            val_split = val_split.shuffle(seed=config.seed).select(range(config.max_val_samples))

        print(f"[Dataset] Tokenizing AG News (max_length={config.max_length})...")
        train_tok = train_split.map(
            lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=config.max_length),
            batched=True,
            remove_columns=["text"]
        )
        val_tok = val_split.map(
            lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=config.max_length),
            batched=True,
            remove_columns=["text"]
        )

        train_dataset = ClientDataset(
            input_ids=torch.tensor(train_tok["input_ids"]),
            attention_mask=torch.tensor(train_tok["attention_mask"]),
            labels=torch.tensor(train_tok["label"])
        )
        val_dataset = ClientDataset(
            input_ids=torch.tensor(val_tok["input_ids"]),
            attention_mask=torch.tensor(val_tok["attention_mask"]),
            labels=torch.tensor(val_tok["label"])
        )
        print(f"[Dataset] Preprocessing complete. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        return train_dataset, val_dataset

    # Mapping GLUE tasks to their text fields
    task_to_keys = {
        "sst2": ("sentence", None),
        "rte": ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "mnli": ("premise", "hypothesis"),
    }
    
    if task not in task_to_keys:
        raise ValueError(f"Unsupported dataset name: {config.dataset_name}. Must be 'ag_news' or in {list(task_to_keys.keys())}")
        
    print(f"[Dataset] Downloading GLUE task '{task}' from Hugging Face...")
    raw_datasets = load_dataset("glue", task)
    sentence1_key, sentence2_key = task_to_keys[task]
    
    def tokenize_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding="max_length", truncation=True, max_length=config.max_length)
        else:
            return tokenizer(examples[sentence1_key], examples[sentence2_key], padding="max_length", truncation=True, max_length=config.max_length)
            
    print(f"[Dataset] Tokenizing '{task}' split datasets...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names)
    
    val_split_name = "validation_matched" if task == "mnli" else "validation"
    
    train_dataset = ClientDataset(
        input_ids=torch.tensor(tokenized_datasets["train"]["input_ids"]),
        attention_mask=torch.tensor(tokenized_datasets["train"]["attention_mask"]),
        labels=torch.tensor(raw_datasets["train"]["label"])
    )
    
    val_dataset = ClientDataset(
        input_ids=torch.tensor(tokenized_datasets[val_split_name]["input_ids"]),
        attention_mask=torch.tensor(tokenized_datasets[val_split_name]["attention_mask"]),
        labels=torch.tensor(raw_datasets[val_split_name]["label"])
    )
    
    print(f"[Dataset] Preprocessing complete. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_dataset, val_dataset

# Compatibility alias
get_glue_dataset = get_dataset
