import argparse
import os
import torch
import numpy as np

# Prevent OpenMP double-linking crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import FLConfig
from utils import set_seed, get_device
from dataset import get_glue_dataset
from partition import dirichlet_partition
from model import get_model, count_parameters
from server import Server
from communication import CommunicationTracker
from privacy.gradient_inversion import simulate_gradient_inversion_attack
from evaluate import generate_comparison_table

def run_federated_tuning(config: FLConfig) -> dict:
    """
    Executes the federated learning training loop for a single PEFT method configuration.
    
    Returns:
      dict: A dictionary of metrics for evaluation report.
    """
    # 1. Setup seed for reproducibility
    set_seed(config.seed)
    
    # 2. Apply defaults based on the dataset settings
    config.apply_defaults()
    
    print("\n" + "="*80)
    print(f"STARTING FEDPETUNING RUN: {config.peft_method.upper()} on '{config.dataset_name}'")
    print(f"Model Name:      {config.model_name}")
    print(f"Num Clients:     {config.num_clients} (Active per round: {int(config.num_clients * config.fraction_fit)})")
    print(f"Comm. Rounds:    {config.num_rounds}")
    print(f"Dirichlet Alpha: {config.dirichlet_alpha}")
    print(f"Learning Rate:   {config.lr}")
    print("="*80)
    
    # 3. Load & Process dataset
    train_dataset, val_dataset = get_glue_dataset(config)
    
    # Optional subsampling for resource-constrained devices
    if config.max_train_samples is not None and config.max_train_samples < len(train_dataset):
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, list(range(config.max_train_samples)))
        print(f"[Dataset] Subsampled training dataset to {config.max_train_samples} samples.")
        
    if config.max_val_samples is not None and config.max_val_samples < len(val_dataset):
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, list(range(config.max_val_samples)))
        print(f"[Dataset] Subsampled validation dataset to {config.max_val_samples} samples.")
    
    # 4. Partition dataset among clients using Dirichlet partitioning
    client_indices = dirichlet_partition(train_dataset, config.num_clients, alpha=config.dirichlet_alpha)
    
    # 5. Initialize the model and configure PEFT layers
    model = get_model(config)
    total_p, trainable_p, frozen_p, trainable_pct = count_parameters(model)
    
    # 6. Initialize Communication Tracker
    # Fetch default trainable state dict
    from client import get_trainable_state_dict
    trainable_state_dict = get_trainable_state_dict(model)
    
    # Calculate FT baseline parameters for tracking reduction percentage
    if config.peft_method.lower() == "none":
        tracker = CommunicationTracker(config, trainable_state_dict, ft_state_dict=trainable_state_dict)
    else:
        # Create a temporary full-FT template state dict for reference
        base_model_temp = get_model(FLConfig(peft_method="none", dataset_name=config.dataset_name))
        ft_state_dict = get_trainable_state_dict(base_model_temp)
        tracker = CommunicationTracker(config, trainable_state_dict, ft_state_dict=ft_state_dict)
        
    # 7. Initialize Server
    server = Server(model, train_dataset, val_dataset, client_indices, config)
    
    # 8. Initial Evaluation
    print("\n[Server] Evaluating initial untrained model...")
    init_loss, init_acc, init_f1 = server.evaluate()
    print(f"  [Initial Model] Loss: {init_loss:.4f} | Accuracy: {init_acc:.4f} | F1: {init_f1:.4f}")
    
    # 9. Federated Training Loop
    history = {"val_acc": [init_acc], "val_f1": [init_f1], "round_losses": []}
    
    for r in range(1, config.num_rounds + 1):
        round_loss = server.fit_round(r)
        val_loss, val_acc, val_f1 = server.evaluate()
        
        history["round_losses"].append(round_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        print(f"  [Round {r} Evaluation] Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
        
    # 10. Privacy/DLG Attack Simulation
    # Sample a batch of private client data
    sample_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.privacy_num_samples, shuffle=True)
    sample_batch = next(iter(sample_loader))
    privacy_results = simulate_gradient_inversion_attack(model, sample_batch["input_ids"], config)
    
    # 11. Compile output metrics
    comm_stats = tracker.report()
    
    return {
        "accuracy": history["val_acc"][-1],
        "f1": history["val_f1"][-1],
        "total_params": total_p,
        "trainable_params": trainable_p,
        "trainable_pct": trainable_pct,
        "comm_total_mb": comm_stats["total_cost_mb"],
        "comm_savings_pct": comm_stats["savings_percentage"],
        "privacy_precision": privacy_results["precision"],
        "privacy_improvement": privacy_results["privacy_improvement"]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedPETuning Faithful Reproduction Runner")
    parser.add_argument("--peft_method", type=str, default="lora", choices=["none", "adapter", "prefix", "lora", "bitfit"], help="PEFT method to evaluate")
    parser.add_argument("--dataset_name", type=str, default="sst2", choices=["sst2", "rte", "mrpc", "qnli", "qqp", "mnli"], help="GLUE task dataset")
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0, help="Dirichlet partition heterogeneity parameter")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit target training samples for faster execution")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit validation samples for faster execution")
    parser.add_argument("--compare_all", action="store_true", help="Runs all 5 methods sequentially and writes final comparison report")
    
    args = parser.parse_args()
    
    if args.compare_all:
        methods = ["none", "adapter", "prefix", "lora", "bitfit"]
        all_results = {}
        
        for method in methods:
            config = FLConfig(
                peft_method=method,
                dataset_name=args.dataset_name,
                dirichlet_alpha=args.dirichlet_alpha,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples
            )
            res = run_federated_tuning(config)
            all_results[method] = res
            
        # Generate Markdown Report
        generate_comparison_table(all_results, target_file_path="README.md")
        print("\n[Comparison Complete] Comparison table printed and saved to README.md.")
        
    else:
        config = FLConfig(
            peft_method=args.peft_method,
            dataset_name=args.dataset_name,
            dirichlet_alpha=args.dirichlet_alpha,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples
        )
        results = run_federated_tuning(config)
        
        print("\n" + "="*80)
        print("RUN COMPLETED STATISTICS")
        print("="*80)
        print(f"Accuracy:                 {results['accuracy']:.4f}")
        print(f"F1 Score:                 {results['f1']:.4f}")
        print(f"Total Parameters:         {results['total_params']:,}")
        print(f"Trainable Parameters:     {results['trainable_params']:,}")
        print(f"Trainable Percentage:     {results['trainable_pct']:.4f}%")
        print(f"Total Comm. Cost (MB):    {results['comm_total_mb']:.2f} MB")
        print(f"Comm. Savings vs FT:      {results['comm_savings_pct']:.2f}%")
        print(f"Privacy Token Precision:  {results['privacy_precision']:.4f}")
        print(f"Privacy Improvement vs FT: {results['privacy_improvement']:.1f}%")
        print("="*80)
