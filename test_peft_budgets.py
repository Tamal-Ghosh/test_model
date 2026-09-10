import os
import torch
import torch.nn as nn

# Ensure OpenMP fix on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import FLConfig
from model import get_model, count_parameters, apply_budget_to_config

def test_all_peft_budgets():
    print("=" * 80)
    print("PHASE 2: 4 PEFT METHODS & PARAMETER BUDGET CALIBRATION")
    print("=" * 80)

    methods = ["lora", "adapter", "prefix", "ia3"]
    budgets = ["low", "medium", "high"]

    results = []

    # Dummy batch for forward and gradient verification
    dummy_input = torch.randint(0, 1000, (2, 32))
    dummy_mask = torch.ones((2, 32), dtype=torch.long)
    dummy_labels = torch.tensor([0, 2], dtype=torch.long)

    for method in methods:
        print(f"\n{'#' * 30} Testing Method: {method.upper()} {'#' * 30}")
        for budget in budgets:
            config = FLConfig(model_name="roberta-base", num_labels=4)
            apply_budget_to_config(config, method=method, budget_tier=budget)

            print(f"\n--- Method: {method.upper()} | Budget: {budget.upper()} (Target: ~{config.target_budget:,}) ---")
            model = get_model(config)
            tot, tr, fr, pct = count_parameters(model)

            # Test forward pass and loss
            outputs = model(input_ids=dummy_input, attention_mask=dummy_mask, labels=dummy_labels)
            loss = outputs.loss
            assert loss is not None and not torch.isnan(loss), f"Forward loss failed for {method} {budget}"

            # Test backward pass to verify gradient flow
            loss.backward()

            # Verify that only trainable parameters received gradients
            grad_params = sum(1 for p in model.parameters() if p.grad is not None)
            trainable_params_count = sum(1 for p in model.parameters() if p.requires_grad)
            assert grad_params == trainable_params_count, (
                f"Gradient mismatch: {grad_params} params have grad, but {trainable_params_count} are trainable!"
            )

            # Store result
            knob_info = ""
            if method == "lora":
                knob_info = f"r={config.lora_r}"
            elif method == "adapter":
                knob_info = f"bottleneck={config.adapter_bottleneck_dim}"
            elif method == "prefix":
                knob_info = f"tokens={config.prefix_num_virtual_tokens}"
            elif method == "ia3":
                knob_info = f"modules={config.ia3_target_modules}"

            results.append({
                "method": method.upper(),
                "budget": budget.upper(),
                "target": config.target_budget,
                "trainable": tr,
                "percentage": pct,
                "knob": knob_info,
                "loss": loss.item()
            })

            print(f"  Trainable: {tr:,} ({pct:.4f}%) | Knob: {knob_info} | Loss: {loss.item():.4f}")

    print("\n" + "=" * 80)
    print(f"{'METHOD':<10} | {'BUDGET':<8} | {'TARGET':<10} | {'ACTUAL TRAINABLE':<18} | {'KNOB':<15} | {'STATUS'}")
    print("-" * 80)
    for r in results:
        status = "PASSED"
        print(f"{r['method']:<10} | {r['budget']:<8} | {r['target']:<10,} | {r['trainable']:<18,} | {r['knob']:<15} | {status}")
    print("=" * 80)

if __name__ == "__main__":
    test_all_peft_budgets()
