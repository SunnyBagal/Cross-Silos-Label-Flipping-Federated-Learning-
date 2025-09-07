#!/usr/bin/env python3
"""
Integration script to add defense mechanisms to your existing FL system.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

# Import defense mechanisms
from defence_mechanisms import DefenseMechanisms, AnomalyDetector
from evaluation_metrics import ComprehensiveEvaluator

class DefensiveServer:
    """Enhanced FL server with defense mechanisms."""
    
    def __init__(self, defense_type="fedavg", num_malicious=1, enable_anomaly_detection=True):
        self.defense_type = defense_type
        self.num_malicious = num_malicious
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Initialize defense components
        self.defense_mechanisms = DefenseMechanisms()
        self.anomaly_detector = AnomalyDetector() if enable_anomaly_detection else None
        self.evaluator = ComprehensiveEvaluator()
        
        print(f"Defensive server initialized with {defense_type} defense")
        print(f"Expected malicious clients: {num_malicious}")
        print(f"Anomaly detection: {'Enabled' if enable_anomaly_detection else 'Disabled'}")

class DefensiveStrategy(fl.server.strategy.FedAvg):
    """Enhanced FedAvg strategy with defense mechanisms."""
    
    def __init__(self, defense_server: DefensiveServer, **kwargs):
        super().__init__(**kwargs)
        self.defense_server = defense_server
        self.round_num = 0
        self.client_metrics_history = {}
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Apply defense mechanisms during aggregation."""
        if not results:
            return None, {}
        
        self.round_num = server_round
        print(f"\nRound {server_round}: Applying {self.defense_server.defense_type} defense")
        
        # Extract parameters and metrics
        parameters_list = []
        client_metrics = []
        
        for client_proxy, fit_res in results:
            # Convert parameters to tensors
            params = [torch.tensor(p) for p in fit_res.parameters]
            flattened_params = torch.cat([p.flatten() for p in params])
            parameters_list.append(flattened_params)
            
            # Store client metrics for anomaly detection
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                client_id = str(hash(str(client_proxy)))[:8]  # Simple client ID
                client_metrics.append((client_id, fit_res.metrics))
        
        # Update anomaly detector
        if self.defense_server.anomaly_detector:
            for client_id, metrics in client_metrics:
                self.defense_server.anomaly_detector.update_history(client_id, metrics)
            
            # Detect anomalies
            anomalous_clients = self.defense_server.anomaly_detector.detect_anomalies()
            if anomalous_clients:
                print(f"Detected anomalous clients: {anomalous_clients}")
        
        # Apply defense mechanism
        try:
            aggregated_params = self._apply_defense_mechanism(parameters_list)
            
            if aggregated_params is None:
                print("Warning: Defense mechanism failed, falling back to FedAvg")
                aggregated_params = torch.mean(torch.stack(parameters_list), dim=0)
        
        except Exception as e:
            print(f"Defense mechanism error: {e}")
            aggregated_params = torch.mean(torch.stack(parameters_list), dim=0)
        
        # Convert back to list format
        aggregated_parameters = self._tensor_to_parameters(aggregated_params, results[0][1].parameters)
        
        # Calculate metrics for monitoring
        self._log_defense_metrics(parameters_list, aggregated_params)
        
        return aggregated_parameters, {}
    
    def _apply_defense_mechanism(self, parameters_list: List[torch.Tensor]) -> torch.Tensor:
        """Apply the specified defense mechanism."""
        defense_type = self.defense_server.defense_type
        defense = self.defense_server.defense_mechanisms
        
        if defense_type == "krum":
            return defense.krum_aggregation(
                parameters_list, 
                num_malicious=self.defense_server.num_malicious
            )
        
        elif defense_type == "multi_krum":
            return defense.krum_aggregation(
                parameters_list, 
                num_malicious=self.defense_server.num_malicious,
                multi_krum=True
            )
        
        elif defense_type == "trimmed_mean":
            trim_ratio = 0.1 if len(parameters_list) <= 5 else 0.2
            return defense.trimmed_mean_aggregation(parameters_list, trim_ratio=trim_ratio)
        
        elif defense_type == "median":
            return defense.median_aggregation(parameters_list)
        
        elif defense_type == "cosine_filter":
            filtered_params = defense.cosine_similarity_filter(parameters_list, threshold=0.6)
            if len(filtered_params) >= 2:
                print(f"Cosine filter: kept {len(filtered_params)}/{len(parameters_list)} clients")
                return torch.mean(torch.stack(filtered_params), dim=0)
            else:
                print("Cosine filter: too few clients passed filter, using all")
                return torch.mean(torch.stack(parameters_list), dim=0)
        
        elif defense_type == "differential_privacy":
            fedavg_result = torch.mean(torch.stack(parameters_list), dim=0)
            return defense.differential_privacy_noise(fedavg_result, noise_scale=0.1)
        
        else:  # fedavg (default)
            return torch.mean(torch.stack(parameters_list), dim=0)
    
    def _tensor_to_parameters(self, flattened_tensor: torch.Tensor, original_params):
        """Convert flattened tensor back to parameter list format."""
        parameters = []
        start_idx = 0
        
        for param_array in original_params:
            param_shape = param_array.shape
            param_size = np.prod(param_shape)
            
            # Extract and reshape parameter
            param_data = flattened_tensor[start_idx:start_idx + param_size]
            param_reshaped = param_data.view(param_shape).cpu().numpy()
            parameters.append(param_reshaped)
            
            start_idx += param_size
        
        return parameters
    
    def _log_defense_metrics(self, parameters_list: List[torch.Tensor], aggregated_params: torch.Tensor):
        """Log metrics about the defense mechanism performance."""
        # Calculate statistics
        individual_norms = [torch.norm(p).item() for p in parameters_list]
        aggregated_norm = torch.norm(aggregated_params).item()
        
        mean_norm = np.mean(individual_norms)
        std_norm = np.std(individual_norms)
        max_norm = max(individual_norms)
        min_norm = min(individual_norms)
        
        print(f"Client update norms - Mean: {mean_norm:.4f}, Std: {std_norm:.4f}")
        print(f"Client update norms - Min: {min_norm:.4f}, Max: {max_norm:.4f}")
        print(f"Aggregated update norm: {aggregated_norm:.4f}")
        
        # Detect potential attacks based on norm distribution
        outlier_threshold = mean_norm + 2 * std_norm
        outliers = [i for i, norm in enumerate(individual_norms) if norm > outlier_threshold]
        if outliers:
            print(f"Potential malicious clients (high norm): {outliers}")

def create_defensive_server(defense_type="krum", num_malicious=1, port=8082):
    """Create and start a defensive FL server."""
    print("Creating Defensive Federated Learning Server")
    print("=" * 50)
    
    # Initialize defensive server
    defensive_server = DefensiveServer(
        defense_type=defense_type,
        num_malicious=num_malicious,
        enable_anomaly_detection=True
    )
    
    # Create defensive strategy
    strategy = DefensiveStrategy(
        defense_server=defensive_server,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    
    # Start server
    print(f"Starting defensive server on port {port}")
    print(f"Defense mechanism: {defense_type}")
    print("Waiting for clients to connect...")
    
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

def run_defense_comparison():
    """Run multiple defense mechanisms sequentially for comparison."""
    defense_types = ["fedavg", "krum", "trimmed_mean", "median", "cosine_filter"]
    
    print("Running Defense Mechanism Comparison")
    print("=" * 50)
    
    for defense_type in defense_types:
        print(f"\nTesting {defense_type.upper()} defense...")
        print("-" * 30)
        
        try:
            # This would typically involve running separate server instances
            # For demo purposes, we'll simulate the process
            defensive_server = DefensiveServer(
                defense_type=defense_type,
                num_malicious=1,
                enable_anomaly_detection=True
            )
            
            print(f"{defense_type} server configured successfully")
            # In practice, you would start the server here and run experiments
            
        except Exception as e:
            print(f"Error with {defense_type}: {e}")
    
    print("\nComparison completed!")
    print("To run actual experiments, start each server separately and connect clients.")

def main():
    """Main function to run defense mechanisms."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FL Server with Defense Mechanisms')
    parser.add_argument('--defense', type=str, default='krum',
                       choices=['fedavg', 'krum', 'multi_krum', 'trimmed_mean', 'median', 
                               'cosine_filter', 'differential_privacy'],
                       help='Defense mechanism to use')
    parser.add_argument('--malicious', type=int, default=1,
                       help='Expected number of malicious clients')
    parser.add_argument('--port', type=int, default=8082,
                       help='Server port')
    parser.add_argument('--test', action='store_true',
                       help='Run defense mechanism tests instead of server')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison of different defense mechanisms')
    
    args = parser.parse_args()
    
    if args.test:
        # Run standalone tests
        print("Running defense mechanism tests...")
        exec(open('test_defenses.py').read())
    elif args.compare:
        run_defense_comparison()
    else:
        # Start defensive server
        create_defensive_server(
            defense_type=args.defense,
            num_malicious=args.malicious,
            port=args.port
        )

if __name__ == "__main__":
    main()