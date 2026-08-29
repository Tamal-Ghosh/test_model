import torch
from typing import List, Tuple, Dict

def federated_average(client_updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """
    Computes the weighted parameter average (FedAvg) over a list of client updates.
    
    Args:
      client_updates: A list of tuples (client_trainable_state_dict, client_dataset_size).
      
    Returns:
      Dict[str, torch.Tensor]: The aggregated global trainable state dict.
    """
    if not client_updates:
        raise ValueError("No client updates provided for FedAvg aggregation.")
        
    # 1. Calculate the total number of samples across all sampled clients
    total_samples = sum(size for _, size in client_updates)
    
    # 2. Initialize the aggregated weights dictionary with zeros
    first_client_weights = client_updates[0][0]
    global_weights = {}
    for key, tensor in first_client_weights.items():
        # Ensure we aggregate in float32 on CPU for numeric stability
        global_weights[key] = torch.zeros_like(tensor, dtype=torch.float32)
        
    # 3. Sum the weighted updates key by key
    for weights, size in client_updates:
        # Weight factor of client k: |D_k| / sum(|D_j|)
        weight_factor = size / total_samples
        for key in weights:
            global_weights[key] += weights[key].to(torch.float32) * weight_factor
            
    # 4. Cast weights back to their original dtypes (e.g. float16/bf16 if the model was loaded so)
    for key in global_weights:
        orig_dtype = first_client_weights[key].dtype
        global_weights[key] = global_weights[key].to(orig_dtype)
        
    return global_weights
