#!/usr/bin/env python3
"""
AI-Enhanced Defense Mechanisms for Federated Learning Security.
Includes adaptive aggregation, differential privacy, and smart filtering.
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class AIEnhancedDefenseMechanisms:
    """A collection of advanced, AI-enhanced defense mechanisms."""

    def __init__(self):
        self.defense_history = defaultdict(lambda: deque(maxlen=20))
        self.performance_history = defaultdict(list)
        self.client_trust_scores = defaultdict(lambda: 1.0) # Initialize trust score for each client

    def get_defense_explanation(self, defense_name: str, threat_assessment: Dict) -> str:
        """Provides a human-readable explanation of a selected defense."""
        threat_level = threat_assessment.get('threat_level', 'LOW')
        num_malicious = len(threat_assessment.get('malicious_clients', []))
        
        explanations = {
            "adaptive_krum": f"Selected Adaptive Krum. With a threat level of {threat_level} and {num_malicious} suspicious clients, this defense is robust against a small number of strong outliers by selecting the best client updates.",
            "intelligent_trimmed_mean": f"Selected Intelligent Trimmed Mean. This defense is effective against mixed attack types by intelligently trimming updates that are outliers based on AI analysis.",
            "trust_weighted": f"Selected Trust-Weighted Aggregation. This defense mitigates threats by down-weighting updates from clients with low trust scores, which is an appropriate response to the current threat assessment.",
            "cluster_filtering": f"Selected Cluster-Based Filtering. This is effective against coordinated attacks by grouping similar updates and isolating malicious clusters.",
            "byzantine_resilient": f"Selected Byzantine-Resilient Aggregation. This is a strong, general-purpose defense against a broad range of attacks, especially effective for a high threat level with multiple suspicious clients."
        }
        return explanations.get(defense_name, "Standard defense selected based on current conditions.")

    def update_defense_history(self, defense_name: str, effectiveness: float):
        """Updates the performance history of a defense."""
        self.performance_history[defense_name].append(effectiveness)

    def get_defense_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Provides performance statistics for all defenses."""
        stats = {}
        for defense, history in self.performance_history.items():
            if history:
                stats[defense] = {
                    "avg_effectiveness": np.mean(history),
                    "std_effectiveness": np.std(history)
                }
        return stats

    def adaptive_defense_selection(self, client_updates: List[torch.Tensor], malicious_scores: Dict[str, float], client_ids: List[str], attack_types: Dict[str, str], round_num: int) -> torch.Tensor:
        """Selects and applies the best defense based on AI analysis."""
        
        # Determine the primary threat type
        primary_threat = "unknown"
        if len(set(attack_types.values())) == 1 and list(attack_types.values())[0] != "unknown":
            primary_threat = list(attack_types.values())[0]
        elif "coordinated" in attack_types.values():
            primary_threat = "coordinated"
        
        if primary_threat == "label_flip":
            defense = "intelligent_trimmed_mean"
        elif primary_threat == "trigger" or primary_threat == "stealth_trigger":
            defense = "adaptive_krum"
        elif primary_threat == "coordinated":
            defense = "cluster_filtering"
        else: # Default to a general-purpose, robust defense
            if len(client_updates) > len(malicious_scores) * 2:
                defense = "trust_weighted"
            else:
                defense = "byzantine_resilient"
        
        # Apply the selected defense
        if defense == "adaptive_krum":
            return self.adaptive_krum(client_updates, malicious_scores, client_ids)
        elif defense == "intelligent_trimmed_mean":
            return self.intelligent_trimmed_mean(client_updates, malicious_scores, client_ids)
        elif defense == "trust_weighted":
            return self.trust_weighted_aggregation(client_updates, malicious_scores, client_ids)
        elif defense == "cluster_filtering":
            return self.cluster_based_filtering(client_updates, malicious_scores, client_ids)
        elif defense == "byzantine_resilient":
            return self.byzantine_resilient_aggregation(client_updates, malicious_scores, client_ids)
        else:
            return self.fedavg_aggregation(client_updates)

    def adaptive_krum(self, client_updates: List[torch.Tensor], malicious_scores: Dict[str, float], client_ids: List[str], num_malicious: int = 1) -> torch.Tensor:
        """Krum defense with AI-enhanced client exclusion."""
        if len(client_updates) <= num_malicious:
            return torch.mean(torch.stack(client_updates), dim=0)
            
        # Use AI scores to exclude highly malicious clients
        trusted_clients = [cid for cid, score in malicious_scores.items() if score < 0.5]
        trusted_updates = [client_updates[client_ids.index(cid)] for cid in trusted_clients]
        
        # Fallback if no trusted clients
        if not trusted_updates:
            return self.fedavg_aggregation(client_updates)
            
        distances = []
        for i, update_i in enumerate(trusted_updates):
            dist_list = [torch.norm(update_i - update_j).item() for update_j in trusted_updates]
            dist_list.sort()
            distances.append(np.sum(dist_list[1:len(trusted_updates) - num_malicious]))
            
        best_client_idx = np.argmin(distances)
        return trusted_updates[best_client_idx]
        
    def intelligent_trimmed_mean(self, client_updates: List[torch.Tensor], malicious_scores: Dict[str, float], client_ids: List[str]) -> torch.Tensor:
        """Trimmed Mean with an AI-informed trimming ratio."""
        if not client_updates:
            return None
        
        num_malicious = sum(1 for score in malicious_scores.values() if score > 0.5)
        trim_ratio = (num_malicious / len(client_updates)) * 1.5
        trim_ratio = min(max(0.1, trim_ratio), 0.4)
        
        updates_tensor = torch.stack(client_updates)
        updates_sorted, _ = torch.sort(updates_tensor, dim=0)
        
        trim_size = int(updates_sorted.size(0) * trim_ratio)
        trimmed_updates = updates_sorted[trim_size:updates_sorted.size(0) - trim_size]
        
        return torch.mean(trimmed_updates, dim=0)

    def trust_weighted_aggregation(self, client_updates: List[torch.Tensor], malicious_scores: Dict[str, float], client_ids: List[str]) -> torch.Tensor:
        """Aggregates updates based on client trust scores."""
        
        # Normalize malicious scores to trust scores (1.0 = high trust, 0.0 = low trust)
        trust_scores = {cid: 1.0 - score for cid, score in malicious_scores.items()}
        total_trust = sum(trust_scores.values())
        
        if total_trust == 0:
            return self.fedavg_aggregation(client_updates)
        
        normalized_weights = [trust_scores[cid] / total_trust for cid in client_ids]
        
        weighted_updates = [update * weight for update, weight in zip(client_updates, normalized_weights)]
        
        return torch.sum(torch.stack(weighted_updates), dim=0)
        
    def cluster_based_filtering(self, client_updates: List[torch.Tensor], malicious_scores: Dict[str, float], client_ids: List[str]) -> torch.Tensor:
        """Filters out malicious clusters using KMeans."""
        if len(client_updates) < 3:
            return self.fedavg_aggregation(client_updates)
            
        updates_tensor = torch.stack(client_updates)
        updates_np = updates_tensor.detach().numpy()
        
        # Reshape for clustering
        flat_updates = updates_np.reshape(len(client_updates), -1)
        
        # Use KMeans to find two clusters (benign and malicious)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(flat_updates)
        
        # Identify the benign cluster
        cluster_0_updates = updates_tensor[clusters == 0]
        cluster_1_updates = updates_tensor[clusters == 1]
        
        # The smaller cluster is likely malicious, especially if it contains clients with high malicious scores
        benign_cluster = cluster_0_updates
        if len(cluster_1_updates) > len(cluster_0_updates):
            benign_cluster = cluster_1_updates
        
        # Verify the choice using malicious scores
        cluster_0_ids = [client_ids[i] for i, c in enumerate(clusters) if c == 0]
        cluster_1_ids = [client_ids[i] for i, c in enumerate(clusters) if c == 1]
        
        avg_score_0 = np.mean([malicious_scores[cid] for cid in cluster_0_ids if cid in malicious_scores]) if cluster_0_ids else 0
        avg_score_1 = np.mean([malicious_scores[cid] for cid in cluster_1_ids if cid in malicious_scores]) if cluster_1_ids else 0
        
        if avg_score_1 < avg_score_0:
            benign_cluster = cluster_1_updates
        else:
            benign_cluster = cluster_0_updates
            
        if len(benign_cluster) > 0:
            return torch.mean(benign_cluster, dim=0)
        else:
            return self.fedavg_aggregation(client_updates)

    def byzantine_resilient_aggregation(self, client_updates: List[torch.Tensor], malicious_scores: Dict[str, float], client_ids: List[str]) -> torch.Tensor:
        """An advanced defense that prunes the most likely malicious updates."""
        
        # Sort clients by malicious score in ascending order
        sorted_clients = sorted(client_ids, key=lambda cid: malicious_scores.get(cid, 0.0))
        
        # Determine how many updates to keep (e.g., top 80%)
        num_updates_to_keep = int(len(sorted_clients) * 0.8)
        
        # Take updates from the least malicious clients
        pruned_updates = [client_updates[client_ids.index(cid)] for cid in sorted_clients[:num_updates_to_keep]]
        
        if not pruned_updates:
            return self.fedavg_aggregation(client_updates)
            
        return torch.mean(torch.stack(pruned_updates), dim=0)

    def differential_privacy_enhanced(self, update: torch.Tensor, threat_level: str) -> torch.Tensor:
        """Adds differential privacy noise based on threat level."""
        
        if threat_level == 'HIGH':
            sensitivity = 0.5
            epsilon = 0.1
        elif threat_level == 'MEDIUM':
            sensitivity = 0.2
            epsilon = 0.5
        else:
            return update
            
        noise_scale = sensitivity / epsilon
        noise = torch.randn_like(update) * noise_scale
        
        return update + noise

    def fedavg_aggregation(self, client_updates: List[torch.Tensor]) -> torch.Tensor:
        """Standard FedAvg aggregation."""
        return torch.mean(torch.stack(client_updates), dim=0)

class SmartAnomalyDetector:
    """Detects anomalies in client behavior using a history-based approach."""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.client_profiles = defaultdict(lambda: {
            'accuracy': deque(maxlen=window_size),
            'loss': deque(maxlen=window_size),
            'update_norm': deque(maxlen=window_size)
        })

    def update_client_profile(self, client_id: str, accuracy: float, loss: float, update_norm: float):
        """Adds new metrics to a client's historical profile."""
        self.client_profiles[client_id]['accuracy'].append(accuracy)
        self.client_profiles[client_id]['loss'].append(loss)
        self.client_profiles[client_id]['update_norm'].append(update_norm)

    def detect_anomalies(self, client_id: str, current_accuracy: float, current_loss: float, current_update_norm: float) -> bool:
        """Detects anomalies based on current metrics compared to historical profile."""
        profile = self.client_profiles.get(client_id, None)
        if not profile or len(profile['accuracy']) < 5:
            return False # Not enough history to detect anomalies
        
        # Calculate deviation from historical mean for each metric
        acc_deviation = abs(current_accuracy - np.mean(profile['accuracy'])) / (np.std(profile['accuracy']) + 1e-6)
        loss_deviation = abs(current_loss - np.mean(profile['loss'])) / (np.std(profile['loss']) + 1e-6)
        norm_deviation = abs(current_update_norm - np.mean(profile['update_norm'])) / (np.std(profile['update_norm']) + 1e-6)
        
        # A simple combined anomaly score
        anomaly_score = acc_deviation + loss_deviation + norm_deviation
        
        return anomaly_score > 3.0 # Threshold can be tuned
