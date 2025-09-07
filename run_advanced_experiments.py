#!/usr/bin/env python3
"""
Main script to run advanced federated learning experiments with attacks and defenses.
This script demonstrates how to use all the advanced components together.
"""

import torch
import numpy as np
import argparse
import json
import time
from datetime import datetime

# Import our components
from comprehensive_experiment_runner import ComprehensiveExperimentRunner
from advanced_attacks import AdvancedAttacks
from defence_mechanisms import DefenseMechanisms, AnomalyDetector
from evaluation_metrics import ComprehensiveEvaluator

def run_single_attack_defense_experiment():
    """Run a single experiment to demonstrate functionality."""
    print("Running Single Attack vs Defense Experiment")
    print("=" * 50)
    
    runner = ComprehensiveExperimentRunner(
        num_clients=6, 
        num_rounds=8
    )
    
    # Test trigger attack vs Krum defense
    report, evaluator = runner.run_single_experiment(
        attack_type="trigger",
        defense_type="krum", 
        malicious_clients=1,
        poison_rate=0.15,
        verbose=True
    )
    
    # Plot results if matplotlib available
    try:
        evaluator.plot_metrics()
    except Exception as e:
        print(f"Could not plot metrics: {e}")
    
    # Save metrics
    evaluator.save_metrics_to_csv("single_experiment_metrics.csv")
    
    return report

def run_comprehensive_evaluation():
    """Run comprehensive evaluation across all attack-defense combinations."""
    print("Running Comprehensive Evaluation")
    print("=" * 50)
    
    runner = ComprehensiveExperimentRunner(
        num_clients=5, 
        num_rounds=6
    )
    
    start_time = time.time()
    all_results, summary = runner.run_comprehensive_evaluation()
    end_time = time.time()
    
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")
    
    # Generate detailed comparison report
    runner.generate_comparison_report()
    
    # Save results
    runner.save_results('comprehensive_results.pkl')
    
    # Save summary to JSON
    try:
        with open('evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("Summary saved to evaluation_summary.json")
    except Exception as e:
        print(f"Could not save summary: {e}")
    
    return all_results, summary

def demonstrate_individual_components():
    """Demonstrate individual components working separately."""
    print("Demonstrating Individual Components")
    print("=" * 50)
    
    # Generate sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    print("1. Testing Advanced Attacks")
    print("-" * 30)
    
    # Test semantic attack
    X_semantic, y_semantic = AdvancedAttacks.semantic_attack(
        None, (X_tensor, y_tensor), target_class=1, semantic_pattern="high_values"
    )
    poisoned_ratio = (y_semantic != y_tensor).float().mean().item()
    print(f"Semantic attack poisoned {poisoned_ratio:.2%} of samples")
    
    # Test delayed activation
    for round_num in range(6):
        poison_rate = AdvancedAttacks.delayed_activation_attack(
            round_num, activation_round=3, normal_poison_rate=0.05, activation_poison_rate=0.3
        )
        print(f"Round {round_num}: Poison rate = {poison_rate:.3f}")
    
    print("\n2. Testing Defense Mechanisms")
    print("-" * 30)
    
    # Create dummy client updates
    client_updates = []
    for i in range(5):
        # Normal update
        update = torch.randn(100) * 0.1
        if i == 4:  # Make last client malicious
            update = torch.randn(100) * 2.0  # Much larger update
        client_updates.append(update)
    
    defense = DefenseMechanisms()
    
    # Test different aggregation methods
    fedavg_result = torch.mean(torch.stack(client_updates), dim=0)
    krum_result = defense.krum_aggregation(client_updates, num_malicious=1)
    trimmed_result = defense.trimmed_mean_aggregation(client_updates, trim_ratio=0.2)
    median_result = defense.median_aggregation(client_updates)
    
    print(f"FedAvg norm: {torch.norm(fedavg_result).item():.3f}")
    print(f"Krum norm: {torch.norm(krum_result).item():.3f}")
    print(f"Trimmed mean norm: {torch.norm(trimmed_result).item():.3f}")
    print(f"Median norm: {torch.norm(median_result).item():.3f}")
    
    print("\n3. Testing Anomaly Detection")
    print("-" * 30)
    
    detector = AnomalyDetector(window_size=3)
    
    # Simulate client metrics over rounds
    for round_num in range(5):
        # Normal clients
        for client_id in range(3):
            metrics = {'accuracy': 0.8 + np.random.normal(0, 0.05), 'loss': 0.3 + np.random.normal(0, 0.1)}
            detector.update_history(f"client_{client_id}", metrics)
        
        # Malicious client with degraded performance
        malicious_metrics = {'accuracy': 0.6 + np.random.normal(0, 0.1), 'loss': 0.8 + np.random.normal(0, 0.2)}
        detector.update_history("client_malicious", malicious_metrics)
    
    anomalous_clients = detector.detect_anomalies(threshold=1.5)
    print(f"Detected anomalous clients: {anomalous_clients}")
    
    print("\n4. Testing Evaluation Metrics")
    print("-" * 30)
    
    from model import HealthNet
    evaluator = ComprehensiveEvaluator()
    
    # Create a simple model for testing
    model = HealthNet(10, 2)
    test_data = (X_tensor[:200], y_tensor[:200])
    
    # Simulate evaluation over rounds
    for round_num in range(5):
        # Create some attack data
        X_attack = X_tensor[:50].clone()
        X_attack[:, -1] += 3.14  # Add trigger
        y_attack = torch.ones(50, dtype=torch.long)
        attack_data = (X_attack, y_attack)
        
        metrics = evaluator.evaluate_round(model, test_data, attack_data, round_num)
        print(f"Round {round_num}: Acc={metrics['benign_accuracy']:.3f}, "
              f"ASR={metrics['attack_success_rate']:.3f}")
    
    # Generate report
    final_report = evaluator.generate_report()
    print(f"\nFinal Report Summary:")
    print(f"Total rounds: {final_report.get('total_rounds', 0)}")
    print(f"Final accuracy: {final_report.get('final_accuracy', 0):.3f}")
    print(f"Max ASR: {final_report.get('max_asr', 0):.3f}")

def run_custom_experiment():
    """Run a custom experiment with specific parameters."""
    print("Running Custom Experiment")
    print("=" * 50)
    
    runner = ComprehensiveExperimentRunner(
        num_clients=8, 
        num_rounds=10
    )
    
    experiments = [
        {"attack": "label_flip", "defense": "fedavg", "malicious": 2, "poison": 0.2},
        {"attack": "trigger", "defense": "krum", "malicious": 1, "poison": 0.15},
        {"attack": "semantic", "defense": "trimmed_mean", "malicious": 1, "poison": 0.1},
        {"attack": "delayed_activation", "defense": "median", "malicious": 1, "poison": 0.1},
    ]
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\nCustom Experiment {i}/{len(experiments)}")
        print(f"Attack: {exp['attack']}, Defense: {exp['defense']}")
        
        report, evaluator = runner.run_single_experiment(
            attack_type=exp["attack"],
            defense_type=exp["defense"],
            malicious_clients=exp["malicious"],
            poison_rate=exp["poison"],
            verbose=True
        )
        
        results.append({
            'experiment': exp,
            'final_accuracy': report.get('final_accuracy', 0),
            'final_asr': report.get('final_asr', 0),
            'max_asr': report.get('max_asr', 0)
        })
    
    # Print summary
    print("\nCustom Experiments Summary:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        exp = result['experiment']
        print(f"Exp {i}: {exp['attack']:<15} vs {exp['defense']:<12} | "
              f"Acc: {result['final_accuracy']:.3f} | ASR: {result['final_asr']:.3f}")
    
    return results

def main():
    """Main function to run experiments based on command line arguments."""
    parser = argparse.ArgumentParser(description='Run Advanced Federated Learning Experiments')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'comprehensive', 'components', 'custom', 'all'],
                       help='Experiment mode to run')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("Advanced Federated Learning Experiment Runner")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'single':
        run_single_attack_defense_experiment()
    elif args.mode == 'comprehensive':
        run_comprehensive_evaluation()
    elif args.mode == 'components':
        demonstrate_individual_components()
    elif args.mode == 'custom':
        run_custom_experiment()
    elif args.mode == 'all':
        print("Running all experiment modes...\n")
        demonstrate_individual_components()
        print("\n" + "=" * 60 + "\n")
        run_single_attack_defense_experiment()
        print("\n" + "=" * 60 + "\n")
        run_custom_experiment()
        print("\n" + "=" * 60 + "\n")
        run_comprehensive_evaluation()
    
    print(f"\nExperiments completed! Check {args.output_dir} for results.")

if __name__ == "__main__":
    main()