import os
import sys
import time
import torch

# Ensure OpenMP conflict fix on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import FLConfig
from dataset import get_dataset
from partition import dirichlet_partition
from model import get_model, count_parameters
from server import Server
from utils import set_seed, get_device

def run_smoke_test():
    print("=" * 70)
    print("PHASE 1 SMOKE TEST: RoBERTa-base + AG News + 10 Clients + FedAvg")
    print("=" * 70)

    # 1. Setup Configuration for a fast verification run
    config = FLConfig(
        model_name="roberta-base",
        dataset_name="ag_news",
        num_labels=4,
        num_clients=10,
        fraction_fit=1.0,         # All 10 clients participate
        num_rounds=1,             # 1 round for smoke testing
        local_epochs=1,
        local_batch_size=8,
        lr=1e-3,
        seed=42,
        dirichlet_alpha=1.0,
        max_train_samples=200,    # Subsample for fast CPU verification
        max_val_samples=100,      # Subsample for fast evaluation
        use_lightweight_head=True,
        peft_method="lora",
        lora_r=3                  # ~100K budget test
    )

    set_seed(config.seed)
    device = get_device()
    print(f"[Device] Using compute device: {device}")

    # 2. Load and tokenize dataset
    start_time = time.time()
    train_ds, val_ds = get_dataset(config)
    print(f"[Time] Dataset preparation completed in {time.time() - start_time:.2f}s")

    # 3. Partition dataset among 10 clients
    print("\n[Partition] Partitioning data among 10 clients with Dirichlet alpha=1.0...")
    client_indices = dirichlet_partition(train_ds, config.num_clients, config.dirichlet_alpha, seed=config.seed)
    assert len(client_indices) == 10, f"Expected 10 client partitions, got {len(client_indices)}"
    total_assigned = sum(len(indices) for indices in client_indices)
    assert total_assigned == len(train_ds), f"Partition count mismatch: {total_assigned} vs {len(train_ds)}"
    print(f"[Partition Check] Successfully partitioned {total_assigned} samples across 10 clients.")

    # 4. Build Model & Verify Parameter Allocations
    print("\n[Model] Initializing RoBERTa model with PEFT...")
    model = get_model(config)
    total_p, trainable_p, frozen_p, trainable_pct = count_parameters(model)
    print(f"  Total Parameters:     {total_p:,}")
    print(f"  Trainable Parameters: {trainable_p:,}")
    print(f"  Frozen Parameters:    {frozen_p:,}")
    print(f"  Trainable Percentage: {trainable_pct:.4f}%")

    assert trainable_p > 0, "No trainable parameters found in model!"

    # 5. Initialize Server and 10 Clients
    print("\n[Server] Initializing Federated Server and 10 Clients...")
    server = Server(model, train_ds, val_ds, client_indices, config)
    assert len(server.clients) == 10, f"Expected 10 clients, got {len(server.clients)}"

    # 6. Execute 1 Round of FedAvg
    print("\n[FL Round] Executing Round 1 across all 10 clients...")
    round_start = time.time()
    round_result = server.fit_round(round_idx=1)
    print(f"[FL Round 1] Average Local Training Loss: {round_result['avg_train_loss']:.4f}")
    print(f"[FL Round 1] Completed in {time.time() - round_start:.2f}s")

    # 7. Evaluate Global Model
    print("\n[Evaluation] Evaluating Global Model on Validation Split...")
    eval_start = time.time()
    val_loss, val_acc, val_f1 = server.evaluate()
    print(f"[Evaluation Result]")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Accuracy:        {val_acc * 100:.2f}%")
    print(f"  Macro-F1:        {val_f1 * 100:.2f}%")
    print(f"[Time] Evaluation completed in {time.time() - eval_start:.2f}s")

    print("\n" + "=" * 70)
    print("PHASE 1 SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()
