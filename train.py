import argparse
import os
# Avoid Windows-specific OpenMP double-linking library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import FLConfig
from dataset import get_datasets
from model import get_model, count_parameters
from server import Server

def run_federated_tuning(config):
    print("=" * 50)
    print("FedPETuning Framework - Initializing Setup")
    print(f"PEFT Method:  {config.peft_method.upper()}")
    print(f"Model Name:   {config.model_name}")
    print(f"Clients:      {config.num_clients} (Fraction fit: {config.fraction_fit})")
    print(f"Rounds:       {config.num_rounds}")
    print(f"Partition:    {config.partition_type.upper()} (Alpha: {config.alpha})")
    print(f"Learning Rate:{config.lr}")
    print("=" * 50)
    
    # 1. Load Data
    train_dataset, val_dataset, client_indices = get_datasets(config)
    
    # Log class distribution per client for visibility
    print("\nClient Partition Information:")
    for k in range(config.num_clients):
        print(f"  Client {k}: {len(client_indices[k])} samples")
        
    # 2. Setup Model
    model = get_model(config)
    total_params, trainable_params, pct_trainable = count_parameters(model)
    print("\nModel Parameter Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Percentage trainable: {pct_trainable:.4f}%")
    print("-" * 50)
    
    # 3. Setup Server
    server = Server(model, train_dataset, val_dataset, client_indices, config)
    
    # Evaluate model before training
    print("\nEvaluating initial model...")
    val_loss, val_acc = server.evaluate()
    print(f"Initial Model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 4. Federated Training Loop
    history = {"val_loss": [val_loss], "val_acc": [val_acc], "round_loss": []}
    
    for r in range(1, config.num_rounds + 1):
        round_loss = server.fit_round(r)
        val_loss, val_acc = server.evaluate()
        
        history["round_loss"].append(round_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Round {r} Finished - Local Avg Loss: {round_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    print("\n" + "=" * 50)
    print("Training Completed Successfully!")
    print(f"Initial Accuracy: {history['val_acc'][0]:.4f}")
    print(f"Final Accuracy:   {history['val_acc'][-1]:.4f}")
    print("=" * 50)
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedPETuning Framework Entry Point")
    parser.add_argument("--peft_method", type=str, default="lora", choices=["none", "adapter", "prefix", "lora", "bitfit"], help="PEFT method to use")
    parser.add_argument("--model_name", type=str, default="toy", help="Model name or 'toy'")
    parser.add_argument("--partition_type", type=str, default="non_iid", choices=["iid", "non_iid"], help="Dataset partition scheme")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet partition parameter")
    parser.add_argument("--num_clients", type=int, default=5, help="Total number of clients")
    parser.add_argument("--fraction_fit", type=float, default=0.6, help="Fraction of clients to train per round")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per client")
    parser.add_argument("--local_batch_size", type=int, default=16, help="Local batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    args = parser.parse_args()
    
    config = FLConfig(
        model_name=args.model_name,
        peft_method=args.peft_method,
        partition_type=args.partition_type,
        alpha=args.alpha,
        num_clients=args.num_clients,
        fraction_fit=args.fraction_fit,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        local_batch_size=args.local_batch_size,
        lr=args.lr
    )
    
    run_federated_tuning(config)
