import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class ComprehensiveEvaluator:
    """Comprehensive evaluation of FL systems under attack."""
    
    def __init__(self):
        self.metrics_history = []
        self.round_metrics = {}
    
    def evaluate_round(self, global_model, test_data, attack_data=None, round_num=0):
        """Evaluate model performance for one round."""
        X_test, y_test = test_data
        global_model.eval()
        
        with torch.no_grad():
            # Benign accuracy
            outputs = global_model(X_test)
            _, predicted = torch.max(outputs, 1)
            benign_acc = (predicted == y_test).float().mean().item()
            
            # Attack success rate (if attack data provided)
            asr = 0.0
            if attack_data is not None:
                X_attack, y_attack_target = attack_data
                attack_outputs = global_model(X_attack)
                _, attack_predicted = torch.max(attack_outputs, 1)
                asr = (attack_predicted == y_attack_target).float().mean().item()
            
            # Model confidence
            probs = torch.softmax(outputs, dim=1)
            confidence = torch.max(probs, dim=1)[0].mean().item()
            
            # Loss calculation
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y_test).item()
        
        metrics = {
            'round': round_num,
            'benign_accuracy': benign_acc,
            'attack_success_rate': asr,
            'confidence': confidence,
            'loss': loss,
            'model_norm': self._calculate_model_norm(global_model)
        }
        
        self.metrics_history.append(metrics)
        self.round_metrics[round_num] = metrics
        
        return metrics
    
    def _calculate_model_norm(self, model):
        """Calculate L2 norm of model parameters."""
        total_norm = 0.0
        for param in model.parameters():
            total_norm += torch.norm(param).item() ** 2
        return total_norm ** 0.5
    
    def detect_performance_degradation(self, window_size=3, threshold=0.1):
        """Detect significant performance degradation."""
        if len(self.metrics_history) < window_size + 1:
            return False
        
        recent_acc = [m['benign_accuracy'] for m in self.metrics_history[-window_size:]]
        baseline_acc = [m['benign_accuracy'] for m in self.metrics_history[:-window_size]]
        
        recent_mean = np.mean(recent_acc)
        baseline_mean = np.mean(baseline_acc)
        
        degradation = baseline_mean - recent_mean
        return degradation > threshold
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        if not self.metrics_history:
            return {}
        
        # Calculate statistics
        accuracies = [m['benign_accuracy'] for m in self.metrics_history]
        asrs = [m['attack_success_rate'] for m in self.metrics_history]
        losses = [m['loss'] for m in self.metrics_history]
        
        report = {
            'total_rounds': len(self.metrics_history),
            'final_accuracy': accuracies[-1],
            'final_asr': asrs[-1],
            'accuracy_trend': np.polyfit(range(len(accuracies)), accuracies, 1)[0] if len(accuracies) > 1 else 0.0,
            'asr_trend': np.polyfit(range(len(asrs)), asrs, 1)[0] if len(asrs) > 1 else 0.0,
            'max_asr': max(asrs),
            'min_accuracy': min(accuracies),
            'accuracy_variance': np.var(accuracies),
            'convergence_round': self._find_convergence_round(),
            'performance_stable': self._is_performance_stable()
        }
        
        return report
    
    def _find_convergence_round(self, tolerance=0.01):
        """Find the round where performance converged."""
        if len(self.metrics_history) < 5:
            return -1
        
        accuracies = [m['benign_accuracy'] for m in self.metrics_history]
        
        for i in range(3, len(accuracies)):
            recent_window = accuracies[i-3:i]
            if max(recent_window) - min(recent_window) < tolerance:
                return i
        
        return len(accuracies)  # Never converged
    
    def _is_performance_stable(self, window_size=5, variance_threshold=0.001):
        """Check if recent performance is stable."""
        if len(self.metrics_history) < window_size:
            return False
        
        recent_acc = [m['benign_accuracy'] for m in self.metrics_history[-window_size:]]
        return np.var(recent_acc) < variance_threshold
    
    def plot_metrics(self):
        """Plot performance metrics over rounds."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.metrics_history:
                print("No metrics to plot")
                return
            
            rounds = [m['round'] for m in self.metrics_history]
            accuracies = [m['benign_accuracy'] for m in self.metrics_history]
            asrs = [m['attack_success_rate'] for m in self.metrics_history]
            losses = [m['loss'] for m in self.metrics_history]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Benign accuracy
            ax1.plot(rounds, accuracies, 'b-o', label='Benign Accuracy')
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Benign Accuracy Over Rounds')
            ax1.grid(True)
            ax1.legend()
            
            # Attack success rate
            ax2.plot(rounds, asrs, 'r-o', label='Attack Success Rate')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('ASR')
            ax2.set_title('Attack Success Rate Over Rounds')
            ax2.grid(True)
            ax2.legend()
            
            # Loss
            ax3.plot(rounds, losses, 'g-o', label='Loss')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Loss')
            ax3.set_title('Loss Over Rounds')
            ax3.grid(True)
            ax3.legend()
            
            # Combined view
            ax4_twin = ax4.twinx()
            ax4.plot(rounds, accuracies, 'b-o', label='Accuracy')
            ax4_twin.plot(rounds, asrs, 'r-s', label='ASR')
            ax4.set_xlabel('Round')
            ax4.set_ylabel('Accuracy', color='b')
            ax4_twin.set_ylabel('Attack Success Rate', color='r')
            ax4.set_title('Accuracy vs ASR')
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig('fl_metrics.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install it with: pip install matplotlib")
    
    def save_metrics_to_csv(self, filename='fl_metrics.csv'):
        """Save metrics to CSV file."""
        try:
            import pandas as pd
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(filename, index=False)
            print(f"Metrics saved to {filename}")
        except ImportError:
            print("Pandas not available. Install it with: pip install pandas")