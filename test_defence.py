#!/usr/bin/env python3
"""
Standalone script to test and run defense mechanisms.
"""

import torch
import numpy as np
from defence_mechanisms import DefenseMechanisms, AnomalyDetector

def simulate_client_updates(num_clients=5, update_size=100, num_malicious=1, attack_strength=2.0):
    """Simulate client updates with some malicious ones."""
    client_updates = []
    
    # Generate benign updates
    for i in range(num_clients - num_malicious):
        # Normal updates are small and follow similar distribution
        update = torch.randn(update_size) * 0.1  # Small variance
        client_updates.append(update)
    
    # Generate malicious updates
    for i in range(num_malicious):
        if attack_strength > 0:
            # Malicious updates are larger or have different distribution
            update = torch.randn(update_size) * attack_strength
        else:
            # Byzantine attack - completely different direction
            update = -torch.randn(update_size) * 0.5
        client_updates.append(update)
    
    return client_updates

def test_defense_mechanisms():
    """Test all defense mechanisms with simulated attacks."""
    print("Testing Defense Mechanisms")
    print("=" * 50)
    
    defense = DefenseMechanisms()
    
    # Test different attack scenarios
    scenarios = [
        {"name": "No Attack", "num_malicious": 0, "attack_strength": 0},
        {"name": "Single Malicious (Large Updates)", "num_malicious": 1, "attack_strength": 3.0},
        {"name": "Multiple Malicious", "num_malicious": 2, "attack_strength": 2.0},
        {"name": "Byzantine Attack", "num_malicious": 1, "attack_strength": -1},
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 30)
        
        # Generate client updates
        client_updates = simulate_client_updates(
            num_clients=6, 
            update_size=50,
            num_malicious=scenario['num_malicious'],
            attack_strength=scenario['attack_strength']
        )
        
        # Test different aggregation methods
        try:
            # FedAvg (baseline)
            fedavg_result = torch.mean(torch.stack(client_updates), dim=0)
            fedavg_norm = torch.norm(fedavg_result).item()
            
            # Krum
            krum_result = defense.krum_aggregation(
                client_updates, 
                num_malicious=scenario['num_malicious']
            )
            krum_norm = torch.norm(krum_result).item()
            
            # Trimmed Mean
            trimmed_result = defense.trimmed_mean_aggregation(
                client_updates, 
                trim_ratio=0.2
            )
            trimmed_norm = torch.norm(trimmed_result).item()
            
            # Median
            median_result = defense.median_aggregation(client_updates)
            median_norm = torch.norm(median_result).item()
            
            # Cosine Similarity Filter
            filtered_updates = defense.cosine_similarity_filter(
                client_updates, 
                threshold=0.5
            )
            if filtered_updates:
                cosine_result = torch.mean(torch.stack(filtered_updates), dim=0)
                cosine_norm = torch.norm(cosine_result).item()
            else:
                cosine_norm = 0.0
            
            print(f"FedAvg norm:        {fedavg_norm:.4f}")
            print(f"Krum norm:          {krum_norm:.4f}")
            print(f"Trimmed Mean norm:  {trimmed_norm:.4f}")
            print(f"Median norm:        {median_norm:.4f}")
            print(f"Cosine Filter norm: {cosine_norm:.4f}")
            print(f"Filtered clients:   {len(filtered_updates)}/{len(client_updates)}")
            
        except Exception as e:
            print(f"Error in scenario: {e}")

def test_anomaly_detection():
    """Test anomaly detection with simulated client metrics."""
    print("\n" + "=" * 50)
    print("Testing Anomaly Detection")
    print("=" * 50)
    
    detector = AnomalyDetector(window_size=5)
    
    # Simulate multiple rounds of training
    num_rounds = 8
    num_clients = 5
    
    print("Simulating federated learning rounds...")
    
    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}:")
        
        # Simulate normal clients
        for client_id in range(num_clients - 1):  # All but last client
            # Normal performance with some variation
            accuracy = 0.85 + np.random.normal(0, 0.03)
            loss = 0.25 + np.random.normal(0, 0.05)
            
            metrics = {
                'accuracy': max(0, min(1, accuracy)),  # Clamp between 0-1
                'loss': max(0.01, loss),
                'round': round_num
            }
            
            detector.update_history(f"client_{client_id}", metrics)
        
        # Simulate malicious client (last one)
        if round_num >= 2:  # Start being malicious after round 2
            # Degraded performance
            accuracy = 0.60 + np.random.normal(0, 0.08)
            loss = 0.60 + np.random.normal(0, 0.10)
        else:
            # Normal performance initially
            accuracy = 0.84 + np.random.normal(0, 0.03)
            loss = 0.26 + np.random.normal(0, 0.05)
        
        malicious_metrics = {
            'accuracy': max(0, min(1, accuracy)),
            'loss': max(0.01, loss),
            'round': round_num
        }
        
        detector.update_history("client_malicious", malicious_metrics)
        
        # Detect anomalies
        anomalous_clients = detector.detect_anomalies(threshold=1.5)
        
        if anomalous_clients:
            print(f"  Detected anomalous clients: {anomalous_clients}")
        else:
            print(f"  No anomalies detected")
    
    # Final anomaly detection with stricter threshold
    print(f"\nFinal anomaly detection (strict threshold=1.0):")
    final_anomalies = detector.detect_anomalies(threshold=1.0)
    print(f"Detected anomalous clients: {final_anomalies}")

def run_defense_comparison_experiment():
    """Run a comprehensive comparison of defense mechanisms."""
    print("\n" + "=" * 50)
    print("Defense Mechanism Comparison Experiment")
    print("=" * 50)
    
    defense = DefenseMechanisms()
    
    # Different attack intensities
    attack_strengths = [0, 1.0, 2.0, 3.0, 5.0]
    defense_methods = ["FedAvg", "Krum", "Trimmed Mean", "Median"]
    
    results = {method: [] for method in defense_methods}
    
    for strength in attack_strengths:
        print(f"\nAttack Strength: {strength}")
        print("-" * 20)
        
        # Generate updates
        client_updates = simulate_client_updates(
            num_clients=8, 
            update_size=100,
            num_malicious=2,
            attack_strength=strength
        )
        
        # Test each defense method
        # FedAvg
        fedavg_result = torch.mean(torch.stack(client_updates), dim=0)
        fedavg_norm = torch.norm(fedavg_result).item()
        results["FedAvg"].append(fedavg_norm)
        
        # Krum
        try:
            krum_result = defense.krum_aggregation(client_updates, num_malicious=2)
            krum_norm = torch.norm(krum_result).item()
            results["Krum"].append(krum_norm)
        except:
            results["Krum"].append(float('inf'))
        
        # Trimmed Mean
        trimmed_result = defense.trimmed_mean_aggregation(client_updates, trim_ratio=0.25)
        trimmed_norm = torch.norm(trimmed_result).item()
        results["Trimmed Mean"].append(trimmed_norm)
        
        # Median
        median_result = defense.median_aggregation(client_updates)
        median_norm = torch.norm(median_result).item()
        results["Median"].append(median_norm)
        
        print(f"FedAvg: {fedavg_norm:.3f}, Krum: {results['Krum'][-1]:.3f}, "
              f"Trimmed: {trimmed_norm:.3f}, Median: {median_norm:.3f}")
    
    # Print summary
    print(f"\nSummary (Attack Strength vs Update Norm):")
    print(f"{'Strength':<10} {'FedAvg':<8} {'Krum':<8} {'Trimmed':<8} {'Median':<8}")
    print("-" * 50)
    
    for i, strength in enumerate(attack_strengths):
        print(f"{strength:<10} {results['FedAvg'][i]:<8.3f} {results['Krum'][i]:<8.3f} "
              f"{results['Trimmed Mean'][i]:<8.3f} {results['Median'][i]:<8.3f}")

def main():
    """Run all defense mechanism tests."""
    print("Defense Mechanism Testing Suite")
    print("=" * 60)
    
    # Test 1: Basic defense mechanisms
    test_defense_mechanisms()
    
    # Test 2: Anomaly detection
    test_anomaly_detection()
    
    # Test 3: Comprehensive comparison
    run_defense_comparison_experiment()
    
    print(f"\n" + "=" * 60)
    print("All defense mechanism tests completed!")
    print("Key Observations:")
    print("- Krum typically has lower norms when attacks are present")
    print("- Trimmed Mean provides good balance between robustness and accuracy")
    print("- Median is very robust but can be conservative")
    print("- Cosine similarity filtering removes outliers effectively")
    print("- Anomaly detection can identify consistently poor-performing clients")

if __name__ == "__main__":
    main()