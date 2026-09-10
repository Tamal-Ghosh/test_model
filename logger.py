import os
import json
import csv
import pandas as pd
from typing import Dict, Any, List

class ExperimentLogger:
    """
    Handles logging of round-by-round federated training metrics,
    client-wise statistics, and final experiment summaries to CSV and JSON.
    """
    def __init__(self, output_dir: str, exp_name: str, config_dict: Dict[str, Any]):
        self.exp_dir = os.path.join(output_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.config_dict = config_dict
        self.round_history: List[Dict[str, Any]] = []
        self.client_history: List[Dict[str, Any]] = []

        # Save config immediately
        with open(os.path.join(self.exp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        self.round_csv_path = os.path.join(self.exp_dir, "round_metrics.csv")
        self.client_csv_path = os.path.join(self.exp_dir, "client_metrics.csv")
        self.summary_json_path = os.path.join(self.exp_dir, "summary.json")

    def log_round(
        self,
        round_idx: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        val_f1: float,
        client_train_losses: Dict[int, float],
        client_val_metrics: Dict[int, Dict[str, float]] = None
    ):
        round_entry = {
            "round": round_idx,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc * 100, 2),
            "val_f1": round(val_f1 * 100, 2),
        }
        self.round_history.append(round_entry)

        # Log client-level details
        if client_val_metrics is None:
            client_val_metrics = {}

        for client_id, loss in client_train_losses.items():
            val_stat = client_val_metrics.get(client_id, {})
            self.client_history.append({
                "round": round_idx,
                "client_id": client_id,
                "train_loss": round(loss, 4),
                "val_loss": round(val_stat.get("loss", 0.0), 4),
                "val_acc": round(val_stat.get("acc", 0.0) * 100, 2),
                "val_f1": round(val_stat.get("f1", 0.0) * 100, 2),
            })

        # Continuously flush to CSV so data is never lost
        pd.DataFrame(self.round_history).to_csv(self.round_csv_path, index=False)
        pd.DataFrame(self.client_history).to_csv(self.client_csv_path, index=False)

    def finalize(self, total_time_sec: float) -> Dict[str, Any]:
        if not self.round_history:
            return {}

        best_acc_entry = max(self.round_history, key=lambda x: x["val_acc"])
        best_f1_entry = max(self.round_history, key=lambda x: x["val_f1"])
        final_entry = self.round_history[-1]

        summary = {
            "method": self.config_dict.get("peft_method"),
            "budget": self.config_dict.get("target_budget"),
            "budget_tier": self.config_dict.get("budget_tier", "custom"),
            "alpha": self.config_dict.get("dirichlet_alpha"),
            "seed": self.config_dict.get("seed"),
            "total_rounds": len(self.round_history),
            "total_time_sec": round(total_time_sec, 2),
            "best_val_acc": best_acc_entry["val_acc"],
            "best_acc_round": best_acc_entry["round"],
            "best_val_f1": best_f1_entry["val_f1"],
            "final_val_acc": final_entry["val_acc"],
            "final_val_f1": final_entry["val_f1"],
            "final_val_loss": final_entry["val_loss"],
            "final_train_loss": final_entry["train_loss"],
            "trainable_parameters": self.config_dict.get("trainable_parameters"),
            "total_parameters": self.config_dict.get("total_parameters"),
        }

        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return summary
