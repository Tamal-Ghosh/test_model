import os
# Avoid Windows-specific OpenMP double-linking library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import FLConfig
from train import run_federated_tuning

def run_tests():
    # Test all five methods from the paper
    methods = ["none", "adapter", "prefix", "lora", "bitfit"]
    results = {}
    
    print("==================================================")
    print("FEDPETUNING END-TO-END SANITY TESTS")
    print("==================================================")
    
    for method in methods:
        print(f"\n[TESTING METHOD: {method.upper()}]")
        config = FLConfig(
            model_name="toy",
            peft_method=method,
            partition_type="non_iid",
            alpha=0.5,
            num_clients=5,
            fraction_fit=0.6,
            num_rounds=2,  # 2 rounds are sufficient for validation
            local_epochs=1,
            local_batch_size=16,
            lr=1e-3
        )
        try:
            history = run_federated_tuning(config)
            init_acc = history["val_acc"][0]
            final_acc = history["val_acc"][-1]
            results[method] = {
                "status": "PASS",
                "initial_accuracy": init_acc,
                "final_accuracy": final_acc,
                "val_loss_history": [f"{l:.4f}" for l in history["val_loss"]]
            }
            print(f"PASS: {method.upper()} completed successfully.")
        except Exception as e:
            results[method] = {
                "status": "FAIL",
                "error": str(e)
            }
            print(f"FAIL: {method.upper()} raised error: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "=" * 50)
    print("FINAL SANITY TEST SUMMARY")
    print("=" * 50)
    all_pass = True
    for method, res in results.items():
        status = res["status"]
        if status == "PASS":
            print(f"  {method.upper():<10}: {status} (Acc: {res['initial_accuracy']:.4f} -> {res['final_accuracy']:.4f})")
        else:
            all_pass = False
            print(f"  {method.upper():<10}: {status} (Error: {res['error']})")
    print("=" * 50)
    
    if not all_pass:
        print("Some tests failed! Please check traceback logs.")
        exit(1)
    else:
        print("All sanity tests passed successfully!")
        
if __name__ == "__main__":
    run_tests()
