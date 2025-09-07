import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from collections import OrderedDict
import copy
from sklearn.cluster import DBSCAN
from scipy import stats

class DefenseMechanisms:
    """Collection of defense mechanisms against poisoning attacks."""
    
    @staticmethod
    def krum_aggregation(client_updates: List[torch.Tensor], num_malicious: int = 1, multi_krum: bool = False):
        """
        Krum/Multi-Krum aggregation for Byzantine robustness.
        Selects updates with lowest sum of squared distances to nearest updates.
        """
        n_clients = len(client_updates)
        if n_clients <= 2 * num_malicious:
            raise ValueError("Not enough clients for Krum defense")
        
        # Calculate pairwise distances
        distances = torch.zeros(n_clients, n_clients)
        for i in range(n_clients):
            for j in range(n_clients):
                if i != j:
                    distances[i, j] = torch.norm(client_updates[i] - client_updates[j])**2
        
        # Calculate Krum scores (sum of distances to k closest neighbors)
        k = n_clients - num_malicious - 2
        krum_scores = []
        
        for i in range(n_clients):
            sorted_distances, _ = torch.sort(distances[i])
            krum_scores.append(torch.sum(sorted_distances[1:k+1]))  # Exclude self (distance=0)
        
        if multi_krum:
            # Multi-Krum: average of m best updates
            m = n_clients - num_malicious
            _, selected_indices = torch.topk(torch.tensor(krum_scores), m, largest=False)
            selected_updates = [client_updates[i] for i in selected_indices]
            return torch.mean(torch.stack(selected_updates), dim=0)
        else:
            # Original Krum: select single best update
            best_idx = torch.argmin(torch.tensor(krum_scores))
            return client_updates[best_idx]
    
    @staticmethod
    def trimmed_mean_aggregation(client_updates: List[torch.Tensor], trim_ratio: float = 0.2):
        """
        Trimmed mean aggregation - removes extreme values before averaging.
        """
        if not client_updates:
            return None
        
        stacked_updates = torch.stack(client_updates)
        n_clients = len(client_updates)
        n_trim = int(n_clients * trim_ratio)
        
        if n_trim * 2 >= n_clients:
            return torch.mean(stacked_updates, dim=0)
        
        # Sort along client dimension and trim extremes
        sorted_updates, _ = torch.sort(stacked_updates, dim=0)
        trimmed_updates = sorted_updates[n_trim:n_clients-n_trim]
        
        return torch.mean(trimmed_updates, dim=0)
    
    @staticmethod
    def median_aggregation(client_updates: List[torch.Tensor]):
        """Coordinate-wise median aggregation."""
        if not client_updates:
            return None
        
        stacked_updates = torch.stack(client_updates)
        return torch.median(stacked_updates, dim=0)[0]
    
    @staticmethod
    def cosine_similarity_filter(client_updates: List[torch.Tensor], threshold: float = 0.7):
        """
        Filter clients based on cosine similarity to the mean.
        Remove updates that are too dissimilar.
        """
        if len(client_updates) < 3:
            return client_updates
        
        # Calculate initial mean
        mean_update = torch.mean(torch.stack(client_updates), dim=0)
        
        # Calculate cosine similarities
        similarities = []
        for update in client_updates:
            cos_sim = torch.nn.functional.cosine_similarity(
                update.flatten(), mean_update.flatten(), dim=0
            )
            similarities.append(cos_sim.item())
        
        # Filter updates above threshold
        filtered_updates = [
            update for update, sim in zip(client_updates, similarities) 
            if sim >= threshold
        ]
        
        return filtered_updates if filtered_updates else client_updates
    
    @staticmethod
    def differential_privacy_noise(client_update: torch.Tensor, noise_scale: float = 0.1):
        """Add Gaussian noise for differential privacy."""
        noise = torch.normal(0, noise_scale, size=client_update.shape)
        return client_update + noise
    
    @staticmethod
    def gradient_clipping(client_update: torch.Tensor, max_norm: float = 1.0):
        """Clip gradient norms to prevent large malicious updates."""
        norm = torch.norm(client_update)
        if norm > max_norm:
            return client_update * (max_norm / norm)
        return client_update

class AnomalyDetector:
    """Detect anomalous client behavior."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.client_history = {}
    
    def update_history(self, client_id: str, metrics: Dict[str, float]):
        """Update client performance history."""
        if client_id not in self.client_history:
            self.client_history[client_id] = []
        
        self.client_history[client_id].append(metrics)
        
        # Keep only recent history
        if len(self.client_history[client_id]) > self.window_size:
            self.client_history[client_id].pop(0)
    
    def detect_anomalies(self, threshold: float = 2.0) -> List[str]:
        """Detect clients with anomalous behavior using statistical methods."""
        anomalous_clients = []
        
        if len(self.client_history) < 3:
            return anomalous_clients
        
        # Collect recent accuracies
        recent_accuracies = {}
        for client_id, history in self.client_history.items():
            if history:
                recent_accuracies[client_id] = [h.get('accuracy', 0) for h in history]
        
        # Calculate z-scores for each client
        all_accuracies = [acc for accs in recent_accuracies.values() for acc in accs]
        if len(all_accuracies) < 5:
            return anomalous_clients
        
        mean_acc = np.mean(all_accuracies)
        std_acc = np.std(all_accuracies)
        
        for client_id, accuracies in recent_accuracies.items():
            if accuracies:
                client_mean = np.mean(accuracies)
                z_score = abs(client_mean - mean_acc) / (std_acc + 1e-8)
                
                if z_score > threshold:
                    anomalous_clients.append(client_id)
        
        return anomalous_clients