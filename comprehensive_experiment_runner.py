import torch
import torch.nn as nn
import copy
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional

# Import our modules
from advanced_attacks import AdvancedAttacks
from defence_mechanisms import DefenseMechanisms, AnomalyDetector
from evaluation_metrics import ComprehensiveEvaluator
from model import HealthNet

class ComprehensiveExperimentRunner:
    """Run comprehensive experiments comparing attacks and defenses."""
    
    def __init__(self, num_clients=5, num_rounds=10, data_generator=None):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.data_generator = data_generator or self._default_data_generator
        self.results = {}
    
    def _default_data_generator(self):
        """Generate default synthetic healthcare data."""
        X, y = make_classification(
            n_samples=5000, n_features=10, n_classes=2, 
            n_informative=8, n_redundant=1, random_state=42
        )
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
    def run_single_experiment(self, attack_type="none", defense_type="fedavg", 
                             malicious_clients=1, poison_rate=0.1, verbose=True):
        """Run a single experiment configuration."""
        if verbose:
            print(f"Running experiment: {attack_type} vs {defense_type}")
            print(f"Malicious clients: {malicious_clients}, Poison rate: {poison_rate}")
            print("-" * 50)
        
        # Initialize components
        evaluator = ComprehensiveEvaluator()
        defense_mechanisms = DefenseMechanisms()
        anomaly_detector = AnomalyDetector()
        
        # Generate data
        X, y = self.data_generator()
        
        # Split data among clients
        client_data = self._split_data_among_clients(X, y)
        
        # Global test set
        test_size = len(X) // 5
        X_test, y_test = X[:test_size], y[:test_size]
        
        # Initialize global model
        global_model = self._create_model(X.shape[1], len(torch.unique(y)))
        
        # Run federated learning rounds
        for round_num in range(self.num_rounds):
            if verbose and (round_num + 1) % 2 == 0:
                print(f"Round {round_num + 1}/{self.num_rounds}")
            
            client_updates = []
            client_metrics = []
            
            for client_id in range(self.num_clients):
                is_malicious = client_id < malicious_clients
                X_client, y_client = client_data[client_id]
                
                # Apply attacks if malicious
                if is_malicious and attack_type != "none":
                    X_client, y_client = self._apply_attack(
                        X_client, y_client, attack_type, poison_rate, round_num, global_model
                    )
                
                # Train local model
                local_model, local_metrics = self._train_local_model(
                    global_model, X_client, y_client, client_id=client_id
                )
                client_updates.append(self._get_model_update(global_model, local_model))
                client_metrics.append(local_metrics)
                
                # Update anomaly detector
                anomaly_detector.update_history(f"client_{client_id}", local_metrics)
            
            # Detect anomalies
            anomalous_clients = anomaly_detector.detect_anomalies(threshold=1.5)
            if verbose and anomalous_clients:
                print(f"Detected anomalous clients: {anomalous_clients}")
            
            # Apply defense mechanism
            aggregated_update = self._apply_defense(
                client_updates, defense_type, malicious_clients, defense_mechanisms
            )
            
            if aggregated_update is not None:
                # Update global model
                self._apply_update_to_model(global_model, aggregated_update)
            
            # Create attack test data for evaluation
            attack_data = self._create_attack_test_data(X_test, y_test, attack_type)
            
            # Evaluate round
            metrics = evaluator.evaluate_round(
                global_model, (X_test, y_test), attack_data, round_num
            )
            
            if verbose:
                print(f"  Benign Acc: {metrics['benign_accuracy']:.4f}, "
                      f"ASR: {metrics['attack_success_rate']:.4f}, "
                      f"Loss: {metrics['loss']:.4f}")
        
        # Generate final report
        report = evaluator.generate_report()
        
        # Store results
        experiment_key = f"{attack_type}_vs_{defense_type}_m{malicious_clients}_p{poison_rate}"
        self.results[experiment_key] = {
            'report': report,
            'evaluator': evaluator,
            'final_model': global_model
        }
        
        if verbose:
            print(f"\nExperiment completed!")
            print(f"Final accuracy: {report.get('final_accuracy', 0):.4f}")
            print(f"Final ASR: {report.get('final_asr', 0):.4f}")
            print(f"Max ASR: {report.get('max_asr', 0):.4f}")
            print("=" * 50)
        
        return report, evaluator
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation across multiple configurations."""
        attack_types = ["none", "label_flip", "trigger", "semantic", "delayed_activation"]
        defense_types = ["fedavg", "krum", "trimmed_mean", "median"]
        
        print("Starting comprehensive evaluation...")
        print("=" * 60)
        
        summary_results = []
        
        for attack in attack_types:
            for defense in defense_types:
                try:
                    report, evaluator = self.run_single_experiment(
                        attack_type=attack, 
                        defense_type=defense,
                        malicious_clients=1,
                        poison_rate=0.1,
                        verbose=False
                    )
                    
                    summary_results.append({
                        'attack': attack,
                        'defense': defense,
                        'final_accuracy': report.get('final_accuracy', 0),
                        'final_asr': report.get('final_asr', 0),
                        'max_asr': report.get('max_asr', 0),
                        'min_accuracy': report.get('min_accuracy', 0)
                    })
                    
                    print(f"{attack:15} vs {defense:12} | "
                          f"Acc: {report.get('final_accuracy', 0):.3f} | "
                          f"ASR: {report.get('final_asr', 0):.3f}")
                    
                except Exception as e:
                    print(f"Error in {attack} vs {defense}: {e}")
        
        # Print summary table
        self._print_summary_table(summary_results)
        
        return self.results, summary_results
    
    def _print_summary_table(self, results):
        """Print a formatted summary table."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 80)
        print(f"{'Attack':<15} {'Defense':<12} {'Final Acc':<10} {'Final ASR':<10} {'Max ASR':<10}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['attack']:<15} {result['defense']:<12} "
                  f"{result['final_accuracy']:<10.3f} {result['final_asr']:<10.3f} "
                  f"{result['max_asr']:<10.3f}")
        
        print("=" * 80)
    
    def _apply_defense(self, client_updates, defense_type, malicious_clients, defense_mechanisms):
        """Apply specified defense mechanism."""
        try:
            if defense_type == "krum":
                return defense_mechanisms.krum_aggregation(
                    client_updates, num_malicious=malicious_clients
                )
            elif defense_type == "trimmed_mean":
                return defense_mechanisms.trimmed_mean_aggregation(
                    client_updates, trim_ratio=0.2
                )
            elif defense_type == "median":
                return defense_mechanisms.median_aggregation(client_updates)
            elif defense_type == "cosine_filter":
                filtered_updates = defense_mechanisms.cosine_similarity_filter(
                    client_updates, threshold=0.7
                )
                return torch.mean(torch.stack(filtered_updates), dim=0)
            else:  # fedavg
                return torch.mean(torch.stack(client_updates), dim=0)
        except Exception as e:
            print(f"Defense mechanism error: {e}")
            return torch.mean(torch.stack(client_updates), dim=0)  # Fallback to FedAvg
    
    def _create_attack_test_data(self, X_test, y_test, attack_type):
        """Create test data for evaluating attack success."""
        if attack_type == "trigger":
            # Create triggered test data
            X_triggered = X_test.clone()
            X_triggered[:, -1] += 3.14  # Add trigger pattern
            y_triggered = torch.ones_like(y_test)  # Target label
            return (X_triggered, y_triggered)
        elif attack_type == "semantic":
            # Create semantic attack test data
            mask = torch.mean(X_test, dim=1) > torch.median(torch.mean(X_test, dim=1))
            X_semantic = X_test[mask]
            y_semantic = torch.ones(len(X_semantic), dtype=torch.long)  # Target label
            if len(X_semantic) > 0:
                return (X_semantic, y_semantic)
        return None
    
    def _split_data_among_clients(self, X, y):
        """Split data among clients (IID distribution)."""
        n_per_client = len(X) // self.num_clients
        client_data = []
        
        for i in range(self.num_clients):
            start_idx = i * n_per_client
            end_idx = start_idx + n_per_client
            client_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))
        
        return client_data
    
    def _create_model(self, input_dim, num_classes):
        """Create a neural network model."""
        return HealthNet(input_dim, num_classes)
    
    def _apply_attack(self, X, y, attack_type, poison_rate, round_num, global_model=None):
        """Apply specified attack to client data."""
        if attack_type == "label_flip":
            # Simple label flipping
            n_poison = int(len(y) * poison_rate)
            if n_poison > 0:
                indices = torch.randperm(len(y))[:n_poison]
                y_attacked = y.clone()
                y_attacked[indices] = 1 - y_attacked[indices]  # Flip binary labels
                return X, y_attacked
            
        elif attack_type == "trigger":
            # Trigger attack
            n_poison = int(len(X) * poison_rate)
            if n_poison > 0:
                indices = torch.randperm(len(X))[:n_poison]
                X_attacked = X.clone()
                X_attacked[indices, -1] += 3.14  # Add trigger
                y_attacked = y.clone()
                y_attacked[indices] = 1  # Target label
                return X_attacked, y_attacked
                
        elif attack_type == "semantic":
            # Semantic attack targeting high-value features
            return AdvancedAttacks.semantic_attack(
                None, (X, y), target_class=1, semantic_pattern="high_values"
            )
            
        elif attack_type == "delayed_activation":
            # Delayed activation attack
            current_poison_rate = AdvancedAttacks.delayed_activation_attack(
                round_num, activation_round=3, normal_poison_rate=poison_rate,
                activation_poison_rate=poison_rate * 3
            )
            return self._apply_attack(X, y, "trigger", current_poison_rate, round_num)
            
        elif attack_type == "adaptive" and global_model is not None:
            # Adaptive attack based on current global model
            return AdvancedAttacks.adaptive_attack(
                global_model, (X, y), defense_type="fedavg", attack_strength=poison_rate
            )
        
        return X, y
    
    def _train_local_model(self, global_model, X, y, client_id=0, epochs=3):
        """Train local model on client data and return metrics."""
        local_model = copy.deepcopy(global_model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
        
        local_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = local_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Calculate local metrics
        local_model.eval()
        with torch.no_grad():
            outputs = local_model(X)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).float().mean().item()
            loss = criterion(outputs, y).item()
        
        metrics = {
            'accuracy': accuracy,
            'loss': loss,
            'client_id': client_id
        }
        
        return local_model, metrics
    
    def _get_model_update(self, global_model, local_model):
        """Calculate model update (difference between local and global)."""
        global_params = torch.cat([p.data.flatten() for p in global_model.parameters()])
        local_params = torch.cat([p.data.flatten() for p in local_model.parameters()])
        return local_params - global_params
    
    def _apply_update_to_model(self, model, update):
        """Apply flattened update to model parameters."""
        start_idx = 0
        for param in model.parameters():
            param_size = param.numel()
            param.data += update[start_idx:start_idx + param_size].view(param.shape)
            start_idx += param_size
    
    def save_results(self, filename='experiment_results.pkl'):
        """Save experiment results to file."""
        try:
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(self.results, f)
            print(f"Results saved to {filename}")
        except ImportError:
            print("Pickle not available for saving results")
    
    def generate_comparison_report(self, attack_types=None, defense_types=None):
        """Generate detailed comparison report."""
        if not self.results:
            print("No results to compare. Run experiments first.")
            return
        
        print("\n" + "=" * 80)
        print("DETAILED COMPARISON REPORT")
        print("=" * 80)
        
        # Best defenses against each attack
        attack_defense_performance = {}
        
        for experiment_key, result in self.results.items():
            parts = experiment_key.split('_vs_')
            if len(parts) >= 2:
                attack = parts[0]
                defense_part = parts[1].split('_m')[0]  # Remove malicious client info
                
                if attack not in attack_defense_performance:
                    attack_defense_performance[attack] = []
                
                report = result['report']
                attack_defense_performance[attack].append({
                    'defense': defense_part,
                    'final_accuracy': report.get('final_accuracy', 0),
                    'final_asr': report.get('final_asr', 0),
                    'max_asr': report.get('max_asr', 0)
                })
        
        # Print best defenses for each attack
        for attack, defenses in attack_defense_performance.items():
            print(f"\nAttack: {attack.upper()}")
            print("-" * 40)
            
            # Sort by final accuracy (descending) and ASR (ascending)
            defenses.sort(key=lambda x: (x['final_accuracy'], -x['final_asr']), reverse=True)
            
            for i, defense in enumerate(defenses[:3], 1):  # Top 3
                print(f"{i}. {defense['defense']:<15} | "
                      f"Acc: {defense['final_accuracy']:.3f} | "
                      f"ASR: {defense['final_asr']:.3f} | "
                      f"Max ASR: {defense['max_asr']:.3f}")
        
        print("\n" + "=" * 80)