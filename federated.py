import torch

def federated_average(client_updates):
    """
    Computes the weighted average of client model parameters (FedAvg).
    
    client_updates: A list of tuples (client_trainable_weights, dataset_size)
    Returns: A dictionary containing the aggregated weights.
    """
    if not client_updates:
        raise ValueError("No client updates provided for aggregation.")
        
    total_samples = sum(size for _, size in client_updates)
    
    # Initialize aggregated weights
    first_update = client_updates[0][0]
    aggregated_weights = {}
    for key, tensor in first_update.items():
        # Aggregate on CPU in float32 for precision
        aggregated_weights[key] = torch.zeros_like(tensor, dtype=torch.float32)
        
    # Accumulate updates
    for weights, size in client_updates:
        weight_factor = size / total_samples
        for key in weights:
            aggregated_weights[key] += weights[key].to(torch.float32) * weight_factor
            
    return aggregated_weights
