import numpy as np
import torch

def dirichlet_partition(dataset, num_clients: int, alpha: float = 1.0):
    """
    Partitions dataset indices among clients using Dirichlet distribution on labels.
    
    alpha (float): controls data heterogeneity.
      - alpha -> infinity: IID partition (equal distributions).
      - alpha -> 0: Highly non-IID partition (clients only get 1 or few classes).
    
    Returns:
      List[List[int]]: A list of index lists where list[i] contains samples allocated to client i.
    """
    # 1. Extract labels from dataset
    if hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
    elif "labels" in dataset[0]:
        labels = np.array([x["labels"].item() if isinstance(x["labels"], torch.Tensor) else x["labels"] for x in dataset])
    elif "label" in dataset[0]:
        labels = np.array([x["label"].item() if isinstance(x["label"], torch.Tensor) else x["label"] for x in dataset])
    else:
        raise ValueError("Dataset does not contain labels attribute or key.")
        
    num_samples = len(dataset)
    num_classes = len(np.unique(labels))
    
    # client_indices[k] stores the list of dataset indices allocated to client k
    client_indices = [[] for _ in range(num_clients)]
    
    # 2. Distribute samples class by class
    for c in range(num_classes):
        # Extract indices belonging to class c
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        # Draw a distribution vector from Dirichlet([alpha, alpha, ...]) of size num_clients
        # E.g., for 5 clients: [0.1, 0.4, 0.05, 0.35, 0.1]
        dirichlet_draw = np.random.dirichlet([alpha] * num_clients)
        
        # Calculate sample counts per client for class c based on the draw
        proportions = (dirichlet_draw * len(class_indices)).astype(int)
        
        # Rounding adjustment: because of integer casting, the sum of proportions might
        # be slightly less than the total count of class_indices.
        diff = len(class_indices) - proportions.sum()
        for i in range(diff):
            proportions[i % num_clients] += 1
            
        # Distribute the slice of indices to each client list
        start_idx = 0
        for k in range(num_clients):
            end_idx = start_idx + proportions[k]
            client_indices[k].extend(class_indices[start_idx:end_idx].tolist())
            start_idx = end_idx
            
    # Shuffle each client's indices to mix labels
    for k in range(num_clients):
        np.random.shuffle(client_indices[k])
        
    # Log distribution statistics for verification
    print(f"[Partition] Dirichlet partition completed (alpha={alpha}).")
    for k in range(min(5, num_clients)):
        print(f"  Client {k}: {len(client_indices[k])} samples")
    if num_clients > 5:
        print(f"  ... and {num_clients - 5} more clients.")
        
    return client_indices
