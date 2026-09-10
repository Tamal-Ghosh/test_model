# federated_peft_experiment.py
# ==============================================================================
# Equal-Budget PEFT for Federated Fine-Tuning (Kaggle & Colab Ready)
# Methods: LoRA, Bottleneck Adapter, Prefix/Prompt-Tuning
# FL Algorithm: FedAvg
# Non-IID Distribution: Dirichlet Partition
# Dataset: AG News (4-class classification)
# Base model: distilbert-base-uncased
# ==============================================================================

import os
import copy
import time
import json
import math
import random
import argparse
import warnings
from dataclasses import dataclass, asdict

# Suppress all library warning outputs
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

# 🔇 Suppress Hugging Face LOAD REPORT & warnings table
transformers.logging.set_verbosity_error()

# ------------------------------------------------------------------------------
# Colab/Kaggle Fix: Bypass torchao version incompatibility bug in PEFT
# ------------------------------------------------------------------------------
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass

from peft import (
    LoraConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "ag_news"

    num_clients: int = 10
    clients_per_round: int = 5
    rounds: int = 3
    local_epochs: int = 1
    batch_size: int = 16
    lr: float = 2e-4
    max_length: int = 128

    train_samples: int = 4000
    test_samples: int = 1000

    alpha_iid: float = 100.0
    alpha_moderate: float = 0.3
    alpha_severe: float = 0.1

    budgets: tuple = (100_000, 250_000)
    seed: int = 42

    output_dir: str = "/kaggle/working/results_equal_budget_peft" if os.path.exists("/kaggle") else "results_equal_budget_peft"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Dataset Loading (Tokenized ONCE)
# -----------------------------
def load_ag_news(cfg, tokenizer):
    print("\n[Data] Loading and tokenizing AG News dataset...", flush=True)
    ds = load_dataset(cfg.dataset_name)

    train = ds["train"].shuffle(seed=cfg.seed)
    test = ds["test"].shuffle(seed=cfg.seed)

    if cfg.train_samples < len(train):
        train = train.select(range(cfg.train_samples))
    if cfg.test_samples < len(test):
        test = test.select(range(cfg.test_samples))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
        )

    train = train.map(tokenize, batched=True, remove_columns=["text"])
    test = test.map(tokenize, batched=True, remove_columns=["text"])

    train = train.rename_column("label", "labels")
    test = test.rename_column("label", "labels")

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train = train.remove_columns([c for c in train.column_names if c not in keep_cols])
    test = test.remove_columns([c for c in test.column_names if c not in keep_cols])

    train.set_format("torch")
    test.set_format("torch")

    print(f"[Data] Loaded {len(train):,} train samples, {len(test):,} test samples.", flush=True)
    return train, test


# -----------------------------
# Dirichlet Non-IID Partition
# -----------------------------
def dirichlet_partition(labels, num_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cut_points = (np.cumsum(proportions) * len(class_indices)).astype(int)
        split_indices = np.split(class_indices, cut_points[:-1])

        for client_id, idx in enumerate(split_indices):
            client_indices[client_id].extend(idx.tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices


# -----------------------------
# Bottleneck Adapter Architecture
# -----------------------------
class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, hidden_size, bias=True)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(torch.nn.functional.gelu(self.down(x)))


class DistilBertWithAdapter(nn.Module):
    def __init__(self, base_model, bottleneck):
        super().__init__()
        self.base_model = base_model
        hidden = base_model.config.dim
        self.adapter = BottleneckAdapter(hidden, bottleneck)

        for p in self.base_model.parameters():
            p.requires_grad = False

        for p in self.base_model.classifier.parameters():
            p.requires_grad = True

        for p in self.adapter.parameters():
            p.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.adapter(cls)
        logits = self.base_model.pre_classifier(cls)
        logits = torch.nn.functional.relu(logits)
        logits = self.base_model.dropout(logits)
        logits = self.base_model.classifier(logits)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


# -----------------------------
# Model Helpers
# -----------------------------
def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_trainable_state(model):
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def load_trainable_state(model, state):
    named = dict(model.named_parameters())
    for name, tensor in state.items():
        if name not in named:
            raise KeyError(f"Missing parameter during FedAvg: {name}")
        named[name].data.copy_(tensor.to(named[name].device))


# -----------------------------
# Build PEFT Models (Fair Parameter Allocation)
# -----------------------------
def build_lora(base_name, num_labels, rank):
    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=num_labels,
    )
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=max(8, rank * 2),
        lora_dropout=0.05,
        target_modules=["q_lin", "v_lin"],
        bias="none",
    )
    model = get_peft_model(base, cfg)

    for name, p in model.named_parameters():
        if "pre_classifier" in name:
            p.requires_grad = False

    return model


def build_prefix(base_name, num_labels, num_virtual_tokens):
    """
    DistilBERT is an encoder-only model without past_key_values.
    PromptTuningConfig is the prompt-learning PEFT method that prepends
    virtual tokens directly to the input embeddings, natively supporting DistilBERT.
    """
    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=num_labels,
    )
    cfg = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=num_virtual_tokens,
        token_dim=base.config.dim,
        num_layers=base.config.n_layers,
        num_attention_heads=base.config.n_heads,
    )
    model = get_peft_model(base, cfg)

    for name, p in model.named_parameters():
        if "pre_classifier" in name:
            p.requires_grad = False

    return model


def build_adapter(base_name, num_labels, bottleneck):
    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=num_labels,
    )
    return DistilBertWithAdapter(base, bottleneck)


def build_model(method, cfg, param):
    if method == "lora":
        return build_lora(cfg.model_name, 4, int(param))
    elif method == "adapter":
        return build_adapter(cfg.model_name, 4, int(param))
    elif method == "prefix":
        return build_prefix(cfg.model_name, 4, int(param))
    else:
        raise ValueError(f"Unknown PEFT method: {method}")


# -----------------------------
# Parameter Budget Search (Cached & Instant)
# -----------------------------
# Formulas for trainable parameters with DistilBERT and frozen pre_classifier:
# Head classifier (Linear 768->4): 768 * 4 + 4 = 3,076
# LoRA (6 layers * 2 matrices * 2 * 768 * r): 18,432 * r + 3,076
# Prompt/Prefix (768 * k virtual tokens): 768 * k + 3,076
# Adapter (2 * 768 * B + B + 768): 1,537 * B + 3,844
CONFIG_CACHE = {}

def find_best_configuration(method, target_budget, cfg):
    cache_key = (method, target_budget)
    if cache_key in CONFIG_CACHE:
        return CONFIG_CACHE[cache_key]

    if method == "lora":
        candidates = list(range(1, 65))
        calc_fn = lambda r: 18432 * r + 3076
    elif method == "prefix":
        candidates = list(range(1, 400))
        calc_fn = lambda k: 768 * k + 3076
    elif method == "adapter":
        candidates = list(range(1, 257))
        calc_fn = lambda b: 1537 * b + 3844
    else:
        raise ValueError(method)

    best = None
    for param in candidates:
        n = calc_fn(param)
        error = abs(n - target_budget) / max(target_budget, 1)
        if best is None or error < best["relative_error"]:
            best = {
                "method": method,
                "knob": param,
                "trainable_params": n,
                "relative_error": error,
            }

    CONFIG_CACHE[cache_key] = best
    return best


# -----------------------------
# Client Training
# -----------------------------
def train_one_client(client_model, dataset, indices, collator, cfg):
    client_model.to(cfg.device)
    client_model.train()

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        [p for p in client_model.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    for _ in range(cfg.local_epochs):
        for batch in loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = client_model(**batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in client_model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

    state = get_trainable_state(client_model)
    return state


# -----------------------------
# FedAvg Aggregator
# -----------------------------
def fedavg(states, weights):
    total = float(sum(weights))
    avg = {}
    keys = states[0].keys()

    for key in keys:
        value = states[0][key].float() * (weights[0] / total)
        for i in range(1, len(states)):
            value += states[i][key].float() * (weights[i] / total)
        avg[key] = value

    return avg


# -----------------------------
# Global & Client Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_global(model, dataset, collator, cfg):
    model.to(cfg.device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
    )

    correct = 0
    total = 0

    for batch in loader:
        labels = batch["labels"].to(cfg.device)
        inputs = {k: v.to(cfg.device) for k, v in batch.items() if k != "labels"}
        out = model(**inputs)
        preds = out["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_clients(model, dataset, client_indices, collator, cfg):
    model.to(cfg.device)
    model.eval()

    scores = []
    for indices in client_indices:
        if len(indices) == 0:
            continue

        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            collate_fn=collator,
        )

        correct = 0
        total = 0
        for batch in loader:
            labels = batch["labels"].to(cfg.device)
            inputs = {k: v.to(cfg.device) for k, v in batch.items() if k != "labels"}
            out = model(**inputs)
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

        scores.append(correct / max(total, 1))

    if not scores:
        return float("nan"), float("nan"), float("nan")

    return float(np.mean(scores)), float(np.min(scores)), float(np.std(scores))


# -----------------------------
# Single Experiment Pipeline
# -----------------------------
def run_experiment(method, budget, heterogeneity, train_data, test_data, collator, cfg):
    print("\n" + "=" * 70, flush=True)
    print(f"METHOD: {method.upper()} | TARGET BUDGET: {budget:,} | DISTRIBUTION: {heterogeneity.upper()}", flush=True)
    print("=" * 70, flush=True)

    if heterogeneity == "iid":
        alpha = cfg.alpha_iid
    elif heterogeneity == "moderate":
        alpha = cfg.alpha_moderate
    elif heterogeneity == "severe":
        alpha = cfg.alpha_severe
    else:
        raise ValueError(heterogeneity)

    labels = np.array(train_data["labels"])
    client_indices = dirichlet_partition(
        labels=labels,
        num_clients=cfg.num_clients,
        alpha=alpha,
        seed=cfg.seed,
    )

    client_sizes = [len(x) for x in client_indices]
    print(f"Client distribution (min={min(client_sizes)}, max={max(client_sizes)}, avg={np.mean(client_sizes):.1f})", flush=True)

    best = find_best_configuration(method, budget, cfg)
    print(
        f"Config selected: knob={best['knob']} | "
        f"Trainable params={best['trainable_params']:,} (budget deviation: {best['relative_error']*100:.2f}%)",
        flush=True
    )

    # Initialize Global Model
    global_model = build_model(method, cfg, best["knob"])
    global_model.to(cfg.device)

    # Initial evaluation
    print("Evaluating initial global model...", end=" ", flush=True)
    global_acc_before = evaluate_global(global_model, test_data, collator, cfg)
    print(f"Initial Test Acc: {global_acc_before:.4f}", flush=True)

    # Reusable client model (saves redundant downloads)
    client_model = copy.deepcopy(global_model)

    history = []

    for rnd in range(1, cfg.rounds + 1):
        print(f"\n--- Round {rnd}/{cfg.rounds} ---", flush=True)

        rng = random.Random(cfg.seed + rnd)
        available = list(range(cfg.num_clients))
        if cfg.clients_per_round < cfg.num_clients:
            selected_clients = rng.sample(available, cfg.clients_per_round)
        else:
            selected_clients = available

        print(f"Participating clients: {selected_clients}", flush=True)

        states = []
        weights = []
        global_state = get_trainable_state(global_model)

        for i, cid in enumerate(selected_clients, 1):
            n_samples = len(client_indices[cid])
            print(f"  [{i}/{len(selected_clients)}] Training Client {cid} ({n_samples} samples)...", end=" ", flush=True)
            t0 = time.time()

            load_trainable_state(client_model, global_state)
            state = train_one_client(
                client_model=client_model,
                dataset=train_data,
                indices=client_indices[cid],
                collator=collator,
                cfg=cfg,
            )
            states.append(state)
            weights.append(max(n_samples, 1))
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)", flush=True)

        # Server FedAvg
        print("  Aggregating updates via FedAvg...", end=" ", flush=True)
        new_global_state = fedavg(states, weights)
        load_trainable_state(global_model, new_global_state)
        print("done", flush=True)

        # Global & Client Evaluation
        print("  Evaluating round results...", end=" ", flush=True)
        global_acc = evaluate_global(global_model, test_data, collator, cfg)
        avg_client, worst_client, client_std = evaluate_clients(
            global_model,
            train_data,
            client_indices,
            collator,
            cfg,
        )

        row = {
            "round": rnd,
            "global_accuracy": global_acc,
            "avg_client_accuracy": avg_client,
            "worst_client_accuracy": worst_client,
            "client_accuracy_std": client_std,
        }
        history.append(row)

        print(
            f"done!\n"
            f"  >> Global Test Acc: {global_acc:.4f} | Avg Client: {avg_client:.4f} | Worst: {worst_client:.4f}",
            flush=True
        )

    final = history[-1]
    result = {
        "method": method,
        "budget_target": budget,
        "actual_trainable_params": best["trainable_params"],
        "budget_relative_error": best["relative_error"],
        "configuration_knob": best["knob"],
        "heterogeneity": heterogeneity,
        "dirichlet_alpha": alpha,
        "seed": cfg.seed,
        "num_clients": cfg.num_clients,
        "clients_per_round": cfg.clients_per_round,
        "rounds": cfg.rounds,
        "local_epochs": cfg.local_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.lr,
        "global_accuracy_initial": global_acc_before,
        "global_accuracy_final": final["global_accuracy"],
        "avg_client_accuracy_final": final["avg_client_accuracy"],
        "worst_client_accuracy_final": final["worst_client_accuracy"],
        "client_accuracy_std_final": final["client_accuracy_std"],
        "history": history,
    }

    del global_model, client_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# -----------------------------
# Main Runner Function
# -----------------------------
def run_pilot(quick: bool = False, seed: int = 42):
    set_seed(seed)
    cfg = Config(seed=seed)

    if quick:
        print("\n" + "=" * 70, flush=True)
        print("⚡ RUNNING QUICK SANITY CHECK (1-2 minutes)", flush=True)
        print("=" * 70, flush=True)
        cfg.num_clients = 4
        cfg.clients_per_round = 2
        cfg.rounds = 1
        cfg.local_epochs = 1
        cfg.train_samples = 800
        cfg.test_samples = 200
        cfg.budgets = (100_000,)
        methods = ["lora", "adapter", "prefix"]
        heterogeneities = ["iid", "severe"]
    else:
        print("\n" + "=" * 70, flush=True)
        print("🔬 RUNNING FULL BENCHMARK MATRIX (18 Experiments)", flush=True)
        print("=" * 70, flush=True)
        cfg.num_clients = 10
        cfg.clients_per_round = 5
        cfg.rounds = 3
        cfg.local_epochs = 1
        cfg.train_samples = 4000
        cfg.test_samples = 1000
        cfg.budgets = (100_000, 250_000)
        methods = ["lora", "adapter", "prefix"]
        heterogeneities = ["iid", "moderate", "severe"]

    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Device: {cfg.device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Accelerator: {torch.cuda.get_device_name(0)}", flush=True)

    # Load Tokenizer & Dataset ONCE
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_data, test_data = load_ag_news(cfg, tokenizer)

    all_results = []
    total_experiments = len(methods) * len(cfg.budgets) * len(heterogeneities)
    current_idx = 0

    for budget in cfg.budgets:
        for heterogeneity in heterogeneities:
            for method in methods:
                current_idx += 1
                print(f"\n=======================================================", flush=True)
                print(f" PROGRESS: EXPERIMENT {current_idx} OF {total_experiments}", flush=True)
                print(f"=======================================================", flush=True)
                res = run_experiment(
                    method=method,
                    budget=budget,
                    heterogeneity=heterogeneity,
                    train_data=train_data,
                    test_data=test_data,
                    collator=collator,
                    cfg=cfg,
                )
                all_results.append(res)

                # Save intermediate JSON & CSV
                json_path = os.path.join(cfg.output_dir, "results.json")
                with open(json_path, "w") as f:
                    json.dump(all_results, f, indent=2)

                rows = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
                csv_path = os.path.join(cfg.output_dir, "results.csv")
                pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 70, flush=True)
    print("ALL EXPERIMENTS SUCCESSFULLY FINISHED 🎉", flush=True)
    print(f"Results saved to: {cfg.output_dir}", flush=True)
    print("=" * 70, flush=True)

    # Display Summary Table
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in all_results])
    summary_cols = [
        "method",
        "budget_target",
        "actual_trainable_params",
        "heterogeneity",
        "global_accuracy_final",
        "avg_client_accuracy_final",
        "worst_client_accuracy_final",
    ]
    print("\nFINAL BENCHMARK TABLE:", flush=True)
    print(df[summary_cols].to_string(index=False), flush=True)

    # Robustness Analysis: IID vs Severe Drop
    print("\nNON-IID ROBUSTNESS ANALYSIS (IID -> Severe Drop):", flush=True)
    for method in methods:
        for budget in cfg.budgets:
            subset = df[(df["method"] == method) & (df["budget_target"] == budget)]
            iid_rows = subset[subset["heterogeneity"] == "iid"]
            severe_rows = subset[subset["heterogeneity"] == "severe"]

            if len(iid_rows) and len(severe_rows):
                iid_acc = float(iid_rows.iloc[0]["global_accuracy_final"])
                severe_acc = float(severe_rows.iloc[0]["global_accuracy_final"])
                drop = iid_acc - severe_acc
                print(
                    f"Method: {method:8s} | Budget: {budget:7,d} | "
                    f"IID: {iid_acc:.4f} | Severe: {severe_acc:.4f} | Drop: {drop:+.4f}",
                    flush=True
                )

    return df


# -----------------------------
# Single Experiment Helper
# -----------------------------
def run_single(
    method: str = "prefix",
    budget: int = 100_000,
    heterogeneity: str = "iid",
    rounds: int = 3,
    num_clients: int = 10,
    clients_per_round: int = 5,
    local_epochs: int = 1,
    train_samples: int = 4000,
    test_samples: int = 1000,
    seed: int = 42,
):
    """
    Run just ONE single experiment configuration.
    Example: run_single(method="prefix", budget=100_000, heterogeneity="iid", rounds=3)
    """
    set_seed(seed)
    cfg = Config(seed=seed)
    cfg.num_clients = num_clients
    cfg.clients_per_round = clients_per_round
    cfg.rounds = rounds
    cfg.local_epochs = local_epochs
    cfg.train_samples = train_samples
    cfg.test_samples = test_samples

    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"\n=======================================================", flush=True)
    print(f"🚀 RUNNING SINGLE EXPERIMENT: {method.upper()} | BUDGET: {budget:,} | {heterogeneity.upper()}", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Device: {cfg.device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Accelerator: {torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_data, test_data = load_ag_news(cfg, tokenizer)

    result = run_experiment(
        method=method,
        budget=budget,
        heterogeneity=heterogeneity,
        train_data=train_data,
        test_data=test_data,
        collator=collator,
        cfg=cfg,
    )

    print("\n" + "=" * 60, flush=True)
    print("🎉 SINGLE EXPERIMENT COMPLETED SUCCESSFULLY!", flush=True)
    print(f"  Method:                  {result['method'].upper()}", flush=True)
    print(f"  Budget Target:           {result['budget_target']:,}", flush=True)
    print(f"  Actual Trainable Params: {result['actual_trainable_params']:,}", flush=True)
    print(f"  Heterogeneity:           {result['heterogeneity'].upper()}", flush=True)
    print(f"  Initial Accuracy:        {result['global_accuracy_initial']:.4f}", flush=True)
    print(f"  Final Global Accuracy:   {result['global_accuracy_final']:.4f}", flush=True)
    print(f"  Avg Client Accuracy:     {result['avg_client_accuracy_final']:.4f}", flush=True)
    print(f"  Worst Client Accuracy:   {result['worst_client_accuracy_final']:.4f}", flush=True)
    print("=" * 60 + "\n", flush=True)

    # Save to CSV
    single_csv = os.path.join(cfg.output_dir, f"single_result_{method}_{budget}_{heterogeneity}.csv")
    row = {k: v for k, v in result.items() if k != "history"}
    pd.DataFrame([row]).to_csv(single_csv, index=False)
    print(f"Saved result to: {single_csv}", flush=True)

    return result


# -----------------------------
# Entry Point (Jupyter / Colab / Kaggle Safe)
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="Run only one experiment.")
    parser.add_argument("--method", type=str, default="prefix", choices=["lora", "adapter", "prefix"])
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--heterogeneity", type=str, default="iid", choices=["iid", "moderate", "severe"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Run fast test.")
    parser.add_argument("--full", action="store_true", help="Run full benchmark.")
    parser.add_argument("--seed", type=int, default=42)

    args, _ = parser.parse_known_args()

    if args.single:
        run_single(
            method=args.method,
            budget=args.budget,
            heterogeneity=args.heterogeneity,
            rounds=args.rounds,
            seed=args.seed,
        )
    else:
        # Runs full or quick benchmark
        run_pilot(quick=args.quick, seed=args.seed)

