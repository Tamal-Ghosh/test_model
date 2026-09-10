import numpy as np
import torch

def dirichlet_partition(dataset, num_clients: int, alpha: float = 1.0, seed: int = 42):
    """
    Partitions dataset indices among clients using Dirichlet distribution on labels.
    
    alpha (float): controls data heterogeneity.
      - alpha -> infinity: IID partition (equal distributions).
      - alpha -> 0: Highly non-IID partition (clients only get 1 or few classes).
    seed (int): Random seed for reproducible partitioning across runs.
    
    Returns:
      List[List[int]]: A list of index lists where list[i] contains samples allocated to client i.
    """
    rng = np.random.default_rng(seed)

    # 1. Extract labels from dataset
    if hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
    elif isinstance(dataset, torch.utils.data.Dataset) and len(dataset) > 0:
        sample = dataset[0]
        if "labels" in sample:
            labels = np.array([x["labels"].item() if isinstance(x["labels"], torch.Tensor) else x["labels"] for x in dataset])
        elif "label" in sample:
            labels = np.array([x["label"].item() if isinstance(x["label"], torch.Tensor) else x["label"] for x in dataset])
        else:
            raise ValueError("Dataset items do not contain 'labels' or 'label' key.")
    else:
        raise ValueError("Dataset does not contain recognizable labels.")
        
    num_samples = len(dataset)
    num_classes = len(np.unique(labels))
    
    client_indices = [[] for _ in range(num_clients)]
    
    # 2. Distribute samples class by class
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)
        
        # Draw a distribution vector from Dirichlet([alpha, ...])
        dirichlet_draw = rng.dirichlet([alpha] * num_clients)
        
        # Calculate sample counts per client for class c
        proportions = (dirichlet_draw * len(class_indices)).astype(int)
        
        diff = len(class_indices) - proportions.sum()
        for i in range(diff):
            proportions[i % num_clients] += 1
            
        start_idx = 0
        for k in range(num_clients):
            end_idx = start_idx + proportions[k]
            client_indices[k].extend(class_indices[start_idx:end_idx].tolist())
            start_idx = end_idx
            
    # Guarantee minimum samples if alpha is extremely small
    for k in range(num_clients):
        rng.shuffle(client_indices[k])
        
    # Log distribution statistics
    print(f"[Partition] Dirichlet partition completed (alpha={alpha}, seed={seed}).")
    for k in range(min(5, num_clients)):
        client_k_labels = labels[client_indices[k]] if len(client_indices[k]) > 0 else []
        class_counts = [int(np.sum(client_k_labels == c)) for c in range(num_classes)]
        print(f"  Client {k}: {len(client_indices[k])} samples | Class counts: {class_counts}")
    if num_clients > 5:
        print(f"  ... and {num_clients - 5} more clients configured.")
        
    return client_indices
