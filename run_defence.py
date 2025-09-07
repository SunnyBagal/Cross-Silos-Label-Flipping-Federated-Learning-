#!/usr/bin/env python3
"""
Simple script to run and test defense mechanisms.
Usage: python run_defenses.py [options]
"""

import torch
import numpy as np
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from defence_mechanisms import DefenseMechanisms, AnomalyDetector

def create_sample_updates(num_clients=5, num_malicious=1, attack_type="large_norm"):
    """Create sample client updates for testing."""
    print(f"Creating {num_clients} client updates ({num_malicious} malicious)")
    
    updates = []
    update_size = 1000  # Size of model parameters
    
    # Create benign updates
    for i in range(num_clients - num_malicious):
        update = torch.randn(update_size) * 0.1  # Small, normal updates
        updates.append(update)
        print(f"Client {i}: benign update (norm: {torch.norm(update).item():.3f})")
    
    # Create malicious updates
    for i in range(num_malicious):
        client_id = num_clients - num_malicious + i
        
        if attack_type == "large_norm":
            update = torch.randn(update_size) * 2.0  # Large updates
        elif attack_type == "sign_flip":
            update = -torch.randn(update_size) * 0.5  # Opposite direction
        elif attack_type == "targeted":
            update = torch.ones(update_size) * 1.5  # All positive values
        else:
            update = torch.randn(update_size) * 1.0
        
        updates.append(update)
        print(f"Client {client_id}: MALICIOUS update (norm: {torch.norm(update).item():.3f})")
    
    return updates

def run_single_defense(defense_name, client_updates, num_malicious=1):
    """Run a single defense mechanism."""
    defense = DefenseMechanisms()
    
    print(f"\n--- Running {defense_name.upper()} Defense ---")
    
    try:
        if defense_name == "fedavg":
            result = torch.mean(torch.stack(client_updates), dim=0)
            
        elif defense_name == "krum":
            result = defense.krum_aggregation(client_updates, num_malicious=num_malicious)
            
        elif defense_name == "multi_krum":
            result = defense.krum_aggregation(client_updates, num_malicious=num_malicious, multi_krum=True)
            
        elif defense_name == "trimmed_mean":
            result = defense.trimmed_mean_aggregation(client_updates, trim_ratio=0.2)
            
        elif defense_name == "median":
            result = defense.median_aggregation(client_updates)
            
        elif defense_name == "cosine_filter":
            filtered_updates = defense.cosine_similarity_filter(client_updates, threshold=0.5)
            result = torch.mean(torch.stack(filtered_updates), dim=0) if filtered_updates else torch.zeros_like(client_updates[0])
            print(f"Cosine filter kept {len(filtered_updates)}/{len(client_updates)} clients")
            
        else:
            print(f"Unknown defense: {defense_name}")
            return None
        
        result_norm = torch.norm(result).item()
        print(f"Result norm: {result_norm:.4f}")
        return result
        
    except Exception as e:
        print(f"Error running {defense_name}: {e}")
        return None

def compare_all_defenses():
    """Compare all defense mechanisms."""
    print("Comparing All Defense Mechanisms")
    print("=" * 50)
    
    # Test different attack scenarios
    scenarios = [
        {"name": "No Attack", "num_malicious": 0, "attack_type": "none"},
        {"name": "Single Large Norm", "num_malicious": 1, "attack_type": "large_norm"},
        {"name": "Multiple Large Norm", "num_malicious": 2, "attack_type": "large_norm"},
        {"name": "Sign Flip Attack", "num_malicious": 1, "attack_type": "sign_flip"},
        {"name": "Targeted Attack", "num_malicious": 1, "attack_type": "targeted"},
    ]
    
    defenses = ["fedavg", "krum", "trimmed_mean", "median", "cosine_filter"]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print('='*60)
        
        # Create updates for this scenario
        updates = create_sample_updates(
            num_clients=6,
            num_malicious=scenario['num_malicious'],
            attack_type=scenario['attack_type']
        )
        
        results = {}
        
        # Test each defense
        for defense_name in defenses:
            result = run_single_defense(defense_name, updates, scenario['num_malicious'])
            if result is not None:
                results[defense_name] = torch.norm(result).item()
        
        # Print comparison
        print(f"\nResults Summary for {scenario['name']}:")
        print("-" * 40)
        for defense_name, norm in results.items():
            print(f"{defense_name:<15}: {norm:.4f}")

def test_anomaly_detection():
    """Test the anomaly detection system."""
    print("\nTesting Anomaly Detection System")
    print("=" * 40)
    
    detector = AnomalyDetector(window_size=3)
    
    # Simulate client performance over multiple rounds
    clients = ["client_0", "client_1", "client_2", "malicious_client"]
    
    for round_num in range(6):
        print(f"\nRound {round_num + 1}:")
        
        for client in clients:
            if client == "malicious_client" and round_num >= 2:
                # Malicious client shows degraded performance after round 2
                accuracy = 0.5 + np.random.normal(0, 0.1)
                loss = 0.8 + np.random.normal(0, 0.2)
            else:
                # Normal clients
                accuracy = 0.85 + np.random.normal(0, 0.05)
                loss = 0.25 + np.random.normal(0, 0.1)
            
            metrics = {
                'accuracy': max(0, min(1, accuracy)),
                'loss': max(0.01, loss)
            }
            
            detector.update_history(client, metrics)
            print(f"  {client}: acc={metrics['accuracy']:.3f}, loss={metrics['loss']:.3f}")
        
        # Check for anomalies
        anomalies = detector.detect_anomalies(threshold=1.5)
        if anomalies:
            print(f"  -> Detected anomalies: {anomalies}")

def interactive_defense_test():
    """Interactive testing interface."""
    print("Interactive Defense Mechanism Tester")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Test single defense mechanism")
        print("2. Compare all defenses")  
        print("3. Test anomaly detection")
        print("4. Create custom attack scenario")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\nAvailable defenses: fedavg, krum, trimmed_mean, median, cosine_filter")
            defense_name = input("Enter defense name: ").strip().lower()
            
            num_clients = int(input("Number of clients (default 5): ") or "5")
            num_malicious = int(input("Number of malicious clients (default 1): ") or "1")
            
            updates = create_sample_updates(num_clients, num_malicious)
            run_single_defense(defense_name, updates, num_malicious)
            
        elif choice == "2":
            compare_all_defenses()
            
        elif choice == "3":
            test_anomaly_detection()
            
        elif choice == "4":
            print("\nCustom Attack Scenario")
            num_clients = int(input("Number of clients: "))
            num_malicious = int(input("Number of malicious clients: "))
            attack_type = input("Attack type (large_norm/sign_flip/targeted): ").strip()
            
            updates = create_sample_updates(num_clients, num_malicious, attack_type)
            
            print("\nTesting with all defenses:")
            defenses = ["fedavg", "krum", "trimmed_mean", "median"]
            for defense in defenses:
                run_single_defense(defense, updates, num_malicious)
                
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1-5.")

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='Defense Mechanism Testing Tool')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'compare', 'single', 'anomaly'],
                       help='Test mode')
    parser.add_argument('--defense', type=str, default='krum',
                       help='Defense mechanism name')
    parser.add_argument('--clients', type=int, default=5,
                       help='Number of clients')
    parser.add_argument('--malicious', type=int, default=1,
                       help='Number of malicious clients')
    parser.add_argument('--attack', type=str, default='large_norm',
                       choices=['large_norm', 'sign_flip', 'targeted'],
                       help='Attack type')
    
    args = parser.parse_args()
    
    print("Defense Mechanism Testing Tool")
    print("=" * 40)
    
    if args.mode == 'interactive':
        interactive_defense_test()
    elif args.mode == 'compare':
        compare_all_defenses()
    elif args.mode == 'single':
        updates = create_sample_updates(args.clients, args.malicious, args.attack)
        run_single_defense(args.defense, updates, args.malicious)
    elif args.mode == 'anomaly':
        test_anomaly_detection()

if __name__ == "__main__":
    main()