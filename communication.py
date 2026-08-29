import torch
from typing import Dict

def get_state_dict_size_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Calculates the size of a state dict in bytes.
    Each parameter size = number of elements * byte size of its data type.
    """
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes

def get_state_dict_num_parameters(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Counts the total number of elements/parameters inside a state dict.
    """
    return sum(tensor.numel() for tensor in state_dict.values())

class CommunicationTracker:
    """
    Calculates the communication overhead (upload + download) in Megabytes (MB).
    Compares the communication cost of PEFT methods with the Full Fine-Tuning baseline.
    """
    def __init__(self, config, trainable_state_dict: Dict[str, torch.Tensor], ft_state_dict: Dict[str, torch.Tensor] = None):
        self.config = config
        
        # Count parameters
        self.num_params = get_state_dict_num_parameters(trainable_state_dict)
        self.size_bytes_per_transfer = get_state_dict_size_bytes(trainable_state_dict)
        
        # Round trip = Download + Upload (2x the parameter transfer size)
        self.round_trip_bytes_per_client = 2 * self.size_bytes_per_transfer
        self.round_trip_mb_per_client = self.round_trip_bytes_per_client / (1024 * 1024)
        
        # Calculate full FT baseline parameters for comparison
        if ft_state_dict is not None:
            self.ft_params = get_state_dict_num_parameters(ft_state_dict)
            self.ft_bytes = get_state_dict_size_bytes(ft_state_dict)
        else:
            # Fallback estimation (if not provided, estimate from the full model params)
            # RoBERTa base model has roughly 125M parameters (approx. 477 MB in float32)
            self.ft_params = 124647170  # exact RoBERTa base params for classification
            self.ft_bytes = self.ft_params * 4
            
        self.ft_round_trip_bytes_per_client = 2 * self.ft_bytes
        self.ft_round_trip_mb_per_client = self.ft_round_trip_bytes_per_client / (1024 * 1024)
        
    def get_round_communication_mb(self) -> float:
        """
        Returns the communication cost of a single round for all active clients.
        """
        num_sampled_clients = max(1, int(self.config.num_clients * self.config.fraction_fit))
        return self.round_trip_mb_per_client * num_sampled_clients
        
    def get_total_communication_mb(self) -> float:
        """
        Returns the total communication cost across all rounds.
        """
        return self.get_round_communication_mb() * self.config.num_rounds
        
    def get_baseline_ft_total_mb(self) -> float:
        """
        Returns the total communication cost for Full Fine-Tuning baseline.
        """
        num_sampled_clients = max(1, int(self.config.num_clients * self.config.fraction_fit))
        return self.ft_round_trip_mb_per_client * num_sampled_clients * self.config.num_rounds
        
    def calculate_savings_percentage(self) -> float:
        """
        Returns the percentage reduction in communication overhead compared to FedFT.
        """
        current_cost = self.get_total_communication_mb()
        ft_cost = self.get_baseline_ft_total_mb()
        if ft_cost == 0:
            return 0.0
        return ((ft_cost - current_cost) / ft_cost) * 100
        
    def report(self) -> dict:
        """
        Generates a summary dictionary of communication metrics.
        """
        return {
            "trainable_parameters": self.num_params,
            "size_per_client_roundtrip_mb": self.round_trip_mb_per_client,
            "round_cost_mb": self.get_round_communication_mb(),
            "total_cost_mb": self.get_total_communication_mb(),
            "savings_percentage": self.calculate_savings_percentage()
        }
