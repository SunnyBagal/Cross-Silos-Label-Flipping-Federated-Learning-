#!/usr/bin/env python3
"""
AI-Enhanced evaluation metrics for federated learning security systems.
Includes metrics specific to AI-driven detection and adaptive defenses.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import json
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class AIEnhancedEvaluator:
    """Enhanced evaluator with AI-specific metrics."""
    
    def __init__(self):
        # Standard metrics
        self.round_metrics = []
        self.detection_history = []
        self.defense_performance = defaultdict(list)
        
        # AI-specific metrics
        self.ai_detection_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'detection_scores': [],
            'ground_truth_labels': []
        }
        
        self.adaptive_defense_metrics = {
            'defense_switches': 0,
            'threat_level_accuracy': [],
            'defense_effectiveness_by_type': defaultdict(list),
            'adaptation_timing': []
        }
        
        self.attack_resilience_metrics = {
            'attack_type_performance': defaultdict(list),
            'coordinated_attack_defense': [],
            'evasion_resistance': []
        }
        
    def evaluate_ai_round(self, global_model, test_data, attack_data,
                         threat_assessment, selected_defense, ground_truth_malicious,
                         round_num):
        """Evaluate a single round with AI-specific metrics."""
        X_test, y_test = test_data
        
        # Standard evaluation
        global_model.eval()
        with torch.no_grad():
            outputs = global_model(X_test)
            _, predicted = torch.max(outputs, 1)
            benign_accuracy = (predicted == y_test).float().mean().item()
            
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y_test).item()
            
            # Confidence metrics
            probs = torch.softmax(outputs, dim=1)
            confidence = probs.max(dim=1)[0].mean().item()
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
        
        # Attack success rate
        attack_success_rate = 0.0
        if attack_data is not None:
            X_attack, y_attack = attack_data
            with torch.no_grad():
                attack_outputs = global_model(X_attack)
                _, attack_predicted = torch.max(attack_outputs, 1)
                attack_success_rate = (attack_predicted == y_attack).float().mean().item()
        
        # AI Detection Evaluation
        detected_malicious = threat_assessment.get('malicious_clients', [])
        malicious_scores = threat_assessment.get('client_scores', {})
        
        self._update_detection_metrics(detected_malicious, ground_truth_malicious, 
                                     malicious_scores)
        
        # Adaptive Defense Evaluation
        self._update_adaptive_metrics(selected_defense, threat_assessment, 
                                    attack_success_rate, round_num)
        
        # Round metrics
        round_metrics = {
            'round': round_num,
            'benign_accuracy': benign_accuracy,
            'attack_success_rate': attack_success_rate,
            'loss': loss,
            'confidence': confidence,
            'entropy': entropy,
            'threat_level': threat_assessment.get('threat_level', 'LOW'),
            'selected_defense': selected_defense,
            'num_detected_malicious': len(detected_malicious),
            'max_malicious_score': threat_assessment.get('max_malicious_score', 0.0),
            'defense_adaptation': selected_defense != 'fedavg'
        }
        
        self.round_metrics.append(round_metrics)
        return round_metrics
    
    def _update_detection_metrics(self, detected_malicious, ground_truth_malicious, 
                                malicious_scores):
        """Update AI detection metrics."""
        if ground_truth_malicious is None:
            return
        
        # Convert to sets for easier comparison
        detected_set = set(detected_malicious)
        truth_set = set(ground_truth_malicious)
        
        # Calculate confusion matrix elements
        tp = len(detected_set.intersection(truth_set))
        fp = len(detected_set - truth_set)
        fn = len(truth_set - detected_set)
        tn = len(malicious_scores) - tp - fp - fn
        
        self.ai_detection_metrics['true_positives'] += tp
        self.ai_detection_metrics['false_positives'] += fp
        self.ai_detection_metrics['true_negatives'] += tn
        self.ai_detection_metrics['false_negatives'] += fn
        
        # Store scores and labels for ROC analysis
        for client_id, score in malicious_scores.items():
            self.ai_detection_metrics['detection_scores'].append(score)
            is_malicious = 1 if client_id in truth_set else 0
            self.ai_detection_metrics['ground_truth_labels'].append(is_malicious)
    
    def _update_adaptive_metrics(self, selected_defense, threat_assessment, 
                               attack_success_rate, round_num):
        """Update adaptive defense metrics."""
        # Track defense switches
        if len(self.round_metrics) > 0:
            prev_defense = self.round_metrics[-1].get('selected_defense', 'fedavg')
            if selected_defense != prev_defense:
                self.adaptive_defense_metrics['defense_switches'] += 1
                self.adaptive_defense_metrics['adaptation_timing'].append(round_num)
        
        # Track defense effectiveness by type
        effectiveness = 1.0 - attack_success_rate  # Higher is better
        self.adaptive_defense_metrics['defense_effectiveness_by_type'][selected_defense].append(effectiveness)
        
        # Track threat level prediction accuracy (if available)
        # This would require external validation of actual threat levels
        
    def calculate_ai_detection_performance(self) -> Dict[str, float]:
        """Calculate comprehensive AI detection performance metrics."""
        metrics = self.ai_detection_metrics
        
        tp, fp, tn, fn = metrics['true_positives'], metrics['false_positives'], \
                        metrics['true_negatives'], metrics['false_negatives']
        
        # Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # ROC AUC (if we have score data)
        roc_auc = 0.0
        if len(metrics['detection_scores']) > 0 and len(set(metrics['ground_truth_labels'])) > 1:
            try:
                roc_auc = roc_auc_score(metrics['ground_truth_labels'], metrics['detection_scores'])
            except Exception:
                roc_auc = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'specificity': specificity,
            'roc_auc': roc_auc,
            'total_detections': tp + fp,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def calculate_adaptive_defense_performance(self) -> Dict[str, any]:
        """Calculate adaptive defense performance metrics."""
        metrics = self.adaptive_defense_metrics
        
        # Defense diversity
        defense_types_used = set()
        for round_metric in self.round_metrics:
            defense_types_used.add(round_metric.get('selected_defense', 'fedavg'))
        
        defense_diversity = len(defense_types_used)
        
        # Adaptation frequency
        total_rounds = len(self.round_metrics)
        adaptation_frequency = metrics['defense_switches'] / total_rounds if total_rounds > 0 else 0.0
        
        # Defense effectiveness comparison
        defense_effectiveness = {}
        for defense_type, effectiveness_scores in metrics['defense_effectiveness_by_type'].items():
            if effectiveness_scores:
                defense_effectiveness[defense_type] = {
                    'mean_effectiveness': np.mean(effectiveness_scores),
                    'std_effectiveness': np.std(effectiveness_scores),
                    'usage_count': len(effectiveness_scores)
                }
        
        # Threat response analysis
        threat_response_accuracy = self._analyze_threat_response()
        
        return {
            'defense_diversity': defense_diversity,
            'adaptation_frequency': adaptation_frequency,
            'total_adaptations': metrics['defense_switches'],
            'defense_effectiveness': defense_effectiveness,
            'threat_response_accuracy': threat_response_accuracy,
            'adaptation_timing': metrics['adaptation_timing']
        }
    
    def _analyze_threat_response(self) -> float:
        """Analyze how well the system responds to different threat levels."""
        if not self.round_metrics:
            return 0.0
        
        correct_responses = 0
        total_responses = 0
        
        for round_metric in self.round_metrics:
            threat_level = round_metric.get('threat_level', 'LOW')
            selected_defense = round_metric.get('selected_defense', 'fedavg')
            
            # Define expected defense for each threat level
            expected_defenses = {
                'LOW': ['fedavg', 'reputation_based'],
                'MEDIUM': ['trust_weighted', 'intelligent_trimmed_mean'],
                'HIGH': ['adaptive_krum', 'cluster_filtering', 'byzantine_resilient']
            }
            
            if threat_level in expected_defenses:
                if selected_defense in expected_defenses[threat_level]:
                    correct_responses += 1
                total_responses += 1
        
        return correct_responses / total_responses if total_responses > 0 else 0.0
    
    def calculate_attack_resilience_metrics(self) -> Dict[str, any]:
        """Calculate metrics for resilience against different attack types."""
        if not self.round_metrics:
            return {}
        
        # Group by attack types (this would need to be tracked separately)
        attack_performance = defaultdict(list)
        
        # For now, analyze based on ASR patterns
        asrs = [rm['attack_success_rate'] for rm in self.round_metrics]
        accuracies = [rm['benign_accuracy'] for rm in self.round_metrics]
        
        return {
            'overall_attack_resilience': 1.0 - np.mean(asrs),
            'consistency_score': 1.0 - np.std(accuracies),
            'recovery_speed': self._calculate_recovery_speed(),
            'worst_case_asr': max(asrs) if asrs else 0.0,
            'best_case_accuracy': max(accuracies) if accuracies else 0.0
        }
    
    def _calculate_recovery_speed(self) -> float:
        """Calculate how quickly the system recovers from attacks."""
        if len(self.round_metrics) < 3:
            return 0.0
        
        recovery_times = []
        for i in range(1, len(self.round_metrics)):
            prev_asr = self.round_metrics[i-1]['attack_success_rate']
            curr_asr = self.round_metrics[i]['attack_success_rate']
            
            if prev_asr > 0.3 and curr_asr < 0.1:  # Recovery detected
                recovery_times.append(1)  # Recovered in 1 round
        
        return np.mean(recovery_times) if recovery_times else 0.0
    
    def generate_comprehensive_ai_report(self) -> Dict[str, any]:
        """Generate comprehensive AI-enhanced evaluation report."""
        # Standard metrics
        if self.round_metrics:
            final_accuracy = self.round_metrics[-1]['benign_accuracy']
            final_asr = self.round_metrics[-1]['attack_success_rate']
            avg_accuracy = np.mean([rm['benign_accuracy'] for rm in self.round_metrics])
            max_asr = max([rm['attack_success_rate'] for rm in self.round_metrics])
            min_accuracy = min([rm['benign_accuracy'] for rm in self.round_metrics])
        else:
            final_accuracy = final_asr = avg_accuracy = max_asr = min_accuracy = 0.0
        
        # AI-specific metrics
        ai_detection_perf = self.calculate_ai_detection_performance()
        adaptive_defense_perf = self.calculate_adaptive_defense_performance()
        attack_resilience = self.calculate_attack_resilience_metrics()
        
        # System intelligence metrics
        intelligence_metrics = self._calculate_system_intelligence()
        
        report = {
            # Standard metrics
            'total_rounds': len(self.round_metrics),
            'final_accuracy': final_accuracy,
            'final_asr': final_asr,
            'average_accuracy': avg_accuracy,
            'max_asr': max_asr,
            'min_accuracy': min_accuracy,
            
            # AI Detection Performance
            'ai_detection_performance': ai_detection_perf,
            
            # Adaptive Defense Performance
            'adaptive_defense_performance': adaptive_defense_perf,
            
            # Attack Resilience
            'attack_resilience_metrics': attack_resilience,
            
            # System Intelligence
            'system_intelligence_metrics': intelligence_metrics,
            
            # Summary statistics
            'threat_level_distribution': self._get_threat_level_distribution(),
            'defense_usage_statistics': self._get_defense_usage_statistics(),
            
            # Timestamp
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_system_intelligence(self) -> Dict[str, float]:
        """Calculate metrics that reflect the intelligence of the system."""
        if not self.round_metrics:
            return {}
        
        # Learning efficiency: how quickly the system improves
        early_rounds = self.round_metrics[:len(self.round_metrics)//3] if len(self.round_metrics) > 3 else self.round_metrics
        late_rounds = self.round_metrics[len(self.round_metrics)//3:] if len(self.round_metrics) > 3 else []
        
        learning_improvement = 0.0
        if early_rounds and late_rounds:
            early_avg_asr = np.mean([rm['attack_success_rate'] for rm in early_rounds])
            late_avg_asr = np.mean([rm['attack_success_rate'] for rm in late_rounds])
            learning_improvement = max(0.0, early_avg_asr - late_avg_asr)
        
        # Adaptation intelligence: appropriate responses to threats
        threat_response_score = 0.0
        if self.round_metrics:
            appropriate_responses = 0
            for rm in self.round_metrics:
                threat_level = rm.get('threat_level', 'LOW')
                defense_adapted = rm.get('defense_adaptation', False)
                
                # High threat should trigger adaptation
                if threat_level == 'HIGH' and defense_adapted:
                    appropriate_responses += 1
                elif threat_level == 'LOW' and not defense_adapted:
                    appropriate_responses += 1
                elif threat_level == 'MEDIUM':
                    appropriate_responses += 0.5  # Partial credit
            
            threat_response_score = appropriate_responses / len(self.round_metrics)
        
        # Proactive defense: early threat detection
        proactive_score = 0.0
        if self.adaptive_defense_metrics['adaptation_timing']:
            avg_adaptation_timing = np.mean(self.adaptive_defense_metrics['adaptation_timing'])
            total_rounds = len(self.round_metrics)
            proactive_score = max(0.0, 1.0 - (avg_adaptation_timing / total_rounds))
        
        return {
            'learning_efficiency': learning_improvement,
            'threat_response_intelligence': threat_response_score,
            'proactive_defense_score': proactive_score,
            'overall_intelligence_score': (learning_improvement + threat_response_score + proactive_score) / 3.0
        }
    
    def _get_threat_level_distribution(self) -> Dict[str, int]:
        """Get distribution of threat levels across rounds."""
        distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for rm in self.round_metrics:
            level = rm.get('threat_level', 'LOW')
            if level in distribution:
                distribution[level] += 1
        return distribution
    
    def _get_defense_usage_statistics(self) -> Dict[str, int]:
        """Get statistics on defense mechanism usage."""
        usage_stats = defaultdict(int)
        for rm in self.round_metrics:
            defense = rm.get('selected_defense', 'unknown')
            usage_stats[defense] += 1
        return dict(usage_stats)
    
    def save_ai_metrics_to_json(self, filename: str):
        """Save AI-enhanced metrics to JSON file."""
        try:
            report = self.generate_comprehensive_ai_report()
            
            # Add raw data for further analysis
            report['raw_round_metrics'] = self.round_metrics
            report['ai_detection_raw'] = {
                'detection_scores': self.ai_detection_metrics['detection_scores'][-100:],  # Last 100
                'ground_truth_labels': self.ai_detection_metrics['ground_truth_labels'][-100:]
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"AI metrics saved to {filename}")
            
        except Exception as e:
            print(f"Error saving AI metrics: {e}")
    
    def print_ai_summary(self):
        """Print a summary of AI-enhanced evaluation results."""
        report = self.generate_comprehensive_ai_report()
        
        print("\n" + "="*80)
        print("AI-ENHANCED FEDERATED LEARNING SECURITY EVALUATION")
        print("="*80)
        
        print(f"Total Rounds: {report['total_rounds']}")
        print(f"Final Accuracy: {report['final_accuracy']:.4f}")
        print(f"Final Attack Success Rate: {report['final_asr']:.4f}")
        
        print("\n--- AI DETECTION PERFORMANCE ---")
        ai_perf = report['ai_detection_performance']
        print(f"Detection Precision: {ai_perf['precision']:.3f}")
        print(f"Detection Recall: {ai_perf['recall']:.3f}")
        print(f"Detection F1-Score: {ai_perf['f1_score']:.3f}")
        print(f"ROC AUC: {ai_perf['roc_auc']:.3f}")
        
        print("\n--- ADAPTIVE DEFENSE PERFORMANCE ---")
        adapt_perf = report['adaptive_defense_performance']
        print(f"Defense Diversity: {adapt_perf['defense_diversity']} mechanisms used")
        print(f"Adaptation Frequency: {adapt_perf['adaptation_frequency']:.3f}")
        print(f"Total Adaptations: {adapt_perf['total_adaptations']}")
        
        print("\n--- SYSTEM INTELLIGENCE ---")
        intel_metrics = report['system_intelligence_metrics']
        print(f"Learning Efficiency: {intel_metrics.get('learning_efficiency', 0):.3f}")
        print(f"Threat Response Intelligence: {intel_metrics.get('threat_response_intelligence', 0):.3f}")
        print(f"Overall Intelligence Score: {intel_metrics.get('overall_intelligence_score', 0):.3f}")
        
        print("\n--- THREAT ANALYSIS ---")
        threat_dist = report['threat_level_distribution']
        print(f"Threat Distribution: HIGH={threat_dist['HIGH']}, MEDIUM={threat_dist['MEDIUM']}, LOW={threat_dist['LOW']}")
        
        print("="*80)