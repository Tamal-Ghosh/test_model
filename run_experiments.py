import os
import sys
import time
import argparse
import copy
import torch
import numpy as np

# Force instant unbuffered real-time stdout for Jupyter/Kaggle/Colab
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Prevent OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Kaggle/Colab Fix: Bypass torchao version incompatibility bug in PEFT
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass

from config import FLConfig
from dataset import get_dataset
from partition import dirichlet_partition
from model import get_model, count_parameters, apply_budget_to_config
from server import Server
from logger import ExperimentLogger
from utils import set_seed, get_device

def run_single_experiment(
    method: str,
    budget_tier: str,
    alpha: float,
    seed: int,
    num_rounds: int = 50,
    local_epochs: int = 1,
    local_batch_size: int = 16,
    lr: float = 1e-3,
    output_dir: str = "results",
    max_train_samples: int = None,
    max_val_samples: int = None,
    eval_clients_every: int = 5,
    hf_repo: str = None
):
    exp_name = f"exp_{method}_{budget_tier}_a{alpha}_s{seed}"
    exp_dir = os.path.join(output_dir, exp_name)
    summary_file = os.path.join(exp_dir, "summary.json")

    # Check for resume capability
    if os.path.exists(summary_file):
        print(f"\n[SKIP] Experiment already completed: {exp_name} (found {summary_file})")
        return

    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"  Method: {method.upper()} | Budget: {budget_tier.upper()} | Alpha: {alpha} | Seed: {seed} | Rounds: {num_rounds}")
    print("=" * 80)

    start_time = time.time()
    set_seed(seed)
    device = get_device()
    print(f"[Device] Running on: {device}")

    # 1. Base Configuration
    config = FLConfig(
        model_name="roberta-base",
        dataset_name="ag_news",
        num_labels=4,
        num_clients=10,
        fraction_fit=1.0, # 100% client participation
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        local_batch_size=local_batch_size,
        lr=lr,
        seed=seed,
        dirichlet_alpha=alpha,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        use_lightweight_head=True
    )
    config.hf_repo = hf_repo

    # 2. Calibrate PEFT hyperparameter knob for the budget
    apply_budget_to_config(config, method=method, budget_tier=budget_tier)

    # 3. Load Tokenized Dataset
    train_ds, val_ds = get_dataset(config)

    # 4. Dirichlet Partition
    client_indices = dirichlet_partition(
        dataset=train_ds,
        num_clients=config.num_clients,
        alpha=config.dirichlet_alpha,
        seed=config.seed
    )

    # 5. Initialize Model
    model = get_model(config)
    total_p, trainable_p, frozen_p, trainable_pct = count_parameters(model)
    print(f"[Model Stats] Total: {total_p:,} | Trainable: {trainable_p:,} ({trainable_pct:.4f}%)")

    # 6. Setup Experiment Logger
    config_dict = {
        "exp_name": exp_name,
        "peft_method": method,
        "budget_tier": budget_tier,
        "target_budget": config.target_budget,
        "trainable_parameters": trainable_p,
        "total_parameters": total_p,
        "trainable_percentage": trainable_pct,
        "dirichlet_alpha": alpha,
        "seed": seed,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "local_batch_size": local_batch_size,
        "lr": lr,
        "num_clients": config.num_clients,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
    }
    logger = ExperimentLogger(output_dir=output_dir, exp_name=exp_name, config_dict=config_dict)

    # 7. Initialize Server
    server = Server(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        client_indices=client_indices,
        config=config
    )

    # 8. Federated Learning Rounds
    for r in range(1, num_rounds + 1):
        round_t0 = time.time()
        print(f"\n--- [Round {r}/{num_rounds}] ({method.upper()} | {budget_tier.upper()} | alpha={alpha}) ---")

        # Fit round across all 10 clients
        round_fit = server.fit_round(round_idx=r)
        train_loss = round_fit["avg_train_loss"]
        client_train_losses = round_fit["client_train_losses"]

        # Evaluate global model on validation set
        val_loss, val_acc, val_f1 = server.evaluate()

        # Optional periodic client-wise validation
        client_val_stats = None
        if eval_clients_every > 0 and (r % eval_clients_every == 0 or r == num_rounds):
            client_val_stats = server.evaluate_clients()

        round_duration = time.time() - round_t0
        print(
            f"[Round {r} Done in {round_duration:.1f}s] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Macro-F1: {val_f1 * 100:.2f}%",
            flush=True
        )

        # Log round metrics
        logger.log_round(
            round_idx=r,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            val_f1=val_f1,
            client_train_losses=client_train_losses,
            client_val_metrics=client_val_stats
        )

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    summary = logger.finalize(total_time_sec=total_time)
    print(f"\n[EXPERIMENT COMPLETED: {exp_name}]")
    print(f"  Best Val Acc:  {summary['best_val_acc']:.2f}% (Round {summary['best_acc_round']})")
    print(f"  Final Val Acc: {summary['final_val_acc']:.2f}%")
    print(f"  Total Time:    {total_time:.1f}s")
    print(f"  Summary saved: {summary_file}")

    # Auto-archive to results_backup.zip for easy 1-click download on Kaggle
    try:
        import shutil
        zip_base = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "results_backup")
        shutil.make_archive(zip_base, "zip", output_dir)
        print(f"[Backup] Updated archive: {zip_base}.zip")
    except Exception as e:
        print(f"[Backup Note] Zip creation skipped: {e}")

    # Auto-sync to Hugging Face Hub if configured
    hf_repo = getattr(config, "hf_repo", None)
    if hf_repo:
        try:
            from huggingface_hub import HfApi
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                api = HfApi(token=hf_token)
                api.upload_folder(
                    folder_path=output_dir,
                    repo_id=hf_repo,
                    repo_type="dataset",
                    commit_message=f"Sync results after {exp_name}"
                )
                print(f"[Backup] Successfully pushed live results to Hugging Face Hub: {hf_repo}")
            else:
                print("[Backup Note] HF_TOKEN environment variable not found. Skipping HF push.")
        except Exception as e:
            print(f"[Backup Warning] Hugging Face push error: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Federated PEFT Experiment Runner (RoBERTa + AG News)")
    parser.add_argument("--methods", nargs="+", default=["lora", "adapter", "prefix", "ia3"],
                        choices=["lora", "adapter", "prefix", "ia3", "all"],
                        help="PEFT method(s) to run")
    parser.add_argument("--budgets", nargs="+", default=["low", "medium", "high"],
                        choices=["low", "medium", "high", "all"],
                        help="Parameter budget tier(s)")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0, 0.5, 0.1, 0.01],
                        help="Dirichlet alpha heterogeneity levels")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                        help="Random seeds (default: 3 seeds: 42, 43, 44)")
    parser.add_argument("--rounds", type=int, default=50, help="Communication rounds (default: 50)")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs (default: 1)")
    parser.add_argument("--batch_size", type=int, default=16, help="Local batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Subsample train set (None=full)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Subsample val set (None=full)")
    parser.add_argument("--eval_clients_every", type=int, default=5, help="Evaluate client-wise stats every N rounds")
    parser.add_argument("--hf_repo", type=str, default=None, help="Hugging Face Dataset repo ID (e.g. username/fl-peft-results) to auto-sync after each experiment")
    return parser.parse_args()

def main():
    args = parse_args()

    methods = ["lora", "adapter", "prefix", "ia3"] if "all" in args.methods else args.methods
    budgets = ["low", "medium", "high"] if "all" in args.budgets else args.budgets
    alphas = args.alphas
    seeds = args.seeds

    total_runs = len(methods) * len(budgets) * len(alphas) * len(seeds)
    print(f"Total experiments queued: {total_runs}")
    print(f"  Methods: {methods}")
    print(f"  Budgets: {budgets}")
    print(f"  Alphas:  {alphas}")
    print(f"  Seeds:   {seeds}")
    if args.hf_repo:
        print(f"  HF Auto-Sync Target: {args.hf_repo}")

    run_count = 0
    for m in methods:
        for b in budgets:
            for a in alphas:
                for s in seeds:
                    run_count += 1
                    print(f"\n>>> Running configuration {run_count}/{total_runs} <<<")
                    run_single_experiment(
                        method=m,
                        budget_tier=b,
                        alpha=a,
                        seed=s,
                        num_rounds=args.rounds,
                        local_epochs=args.local_epochs,
                        local_batch_size=args.batch_size,
                        lr=args.lr,
                        output_dir=args.output_dir,
                        max_train_samples=args.max_train_samples,
                        max_val_samples=args.max_val_samples,
                        eval_clients_every=args.eval_clients_every,
                        hf_repo=args.hf_repo
                    )

if __name__ == "__main__":
    main()
