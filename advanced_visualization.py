#!/usr/bin/env python3
"""
Advanced visualization and analysis system for federated learning security experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

class AdvancedVisualizationSystem:
    """Advanced visualization system for FL security analysis."""
    
    def __init__(self, output_dir="results/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_attack_defense_heatmap(self, results_data, save_path=None):
        """Create heatmap showing attack success rates across defense mechanisms."""
        # Convert results to DataFrame
        if isinstance(results_data, list):
            df = pd.DataFrame(results_data)
        else:
            df = pd.DataFrame.from_dict(results_data, orient='index')
        
        # Pivot table for heatmap
        heatmap_data = df.pivot_table(
            values='final_asr', 
            index='attack', 
            columns='defense', 
            fill_value=0
        )
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        ax = sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt='.3f',
            cmap='RdYlBu_r',
            cbar_kws={'label': 'Attack Success Rate'},
            linewidths=0.5
        )
        
        plt.title('Attack Success Rate by Defense Mechanism', fontsize=16, pad=20)
        plt.xlabel('Defense Mechanism', fontsize=12)
        plt.ylabel('Attack Type', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.output_dir}/attack_defense_heatmap.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        return heatmap_data
    
    def create_defense_effectiveness_comparison(self, results_data, save_path=None):
        """Create bar chart comparing defense effectiveness."""
        if isinstance(results_data, list):
            df = pd.DataFrame(results_data)
        else:
            df = pd.DataFrame.from_dict(results_data, orient='index')
        
        # Calculate average ASR for each defense
        defense_effectiveness = df.groupby('defense').agg({
            'final_asr': 'mean',
            'final_accuracy': 'mean',
            'max_asr': 'mean'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ASR comparison
        bars1 = ax1.bar(defense_effectiveness['defense'], defense_effectiveness['final_asr'], 
                       color='coral', alpha=0.7)
        ax1.set_title('Average Attack Success Rate by Defense', fontsize=14)
        ax1.set_ylabel('Final ASR')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Accuracy comparison
        bars2 = ax2.bar(defense_effectiveness['defense'], defense_effectiveness['final_accuracy'],
                       color='lightblue', alpha=0.7)
        ax2.set_title('Average Final Accuracy by Defense', fontsize=14)
        ax2.set_ylabel('Final Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.output_dir}/defense_effectiveness.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        return defense_effectiveness
    
    def create_attack_progression_analysis(self, experiment_results, save_path=None):
        """Analyze how attacks progress over rounds."""
        plt.figure(figsize=(15, 10))
        
        # Extract round-by-round data from experiments
        attack_types = ['label_flip', 'trigger', 'semantic', 'delayed_activation']
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, attack in enumerate(attack_types):
            # Find experiments with this attack
            attack_experiments = {k: v for k, v in experiment_results.items() 
                                if attack in k and 'evaluator' in v}
            
            if attack_experiments:
                # Get first experiment for this attack
                exp_key = list(attack_experiments.keys())[0]
                evaluator = attack_experiments[exp_key]['evaluator']
                
                if hasattr(evaluator, 'metrics_history'):
                    rounds = [m['round'] for m in evaluator.metrics_history]
                    asrs = [m['attack_success_rate'] for m in evaluator.metrics_history]
                    accuracies = [m['benign_accuracy'] for m in evaluator.metrics_history]
                    
                    plt.subplot(2, 2, 1)
                    plt.plot(rounds, asrs, label=attack.replace('_', ' ').title(), 
                            color=colors[i], marker='o')
                    plt.title('Attack Success Rate Over Rounds')
                    plt.xlabel('Round')
                    plt.ylabel('ASR')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(2, 2, 2)
                    plt.plot(rounds, accuracies, label=attack.replace('_', ' ').title(),
                            color=colors[i], marker='s')
                    plt.title('Benign Accuracy Over Rounds')
                    plt.xlabel('Round')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
        
        # Add additional analysis plots
        plt.subplot(2, 2, 3)
        # Attack vs Defense effectiveness scatter
        if isinstance(experiment_results, dict) and len(experiment_results) > 0:
            first_key = list(experiment_results.keys())[0]
            if 'report' in experiment_results[first_key]:
                asrs = [v['report'].get('final_asr', 0) for v in experiment_results.values()]
                accs = [v['report'].get('final_accuracy', 0) for v in experiment_results.values()]
                
                plt.scatter(asrs, accs, alpha=0.6, s=60)
                plt.xlabel('Final ASR')
                plt.ylabel('Final Accuracy')
                plt.title('ASR vs Accuracy Trade-off')
                plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # Defense mechanism comparison
        defense_types = ['fedavg', 'krum', 'trimmed_mean', 'median']
        defense_asrs = []
        
        for defense in defense_types:
            defense_experiments = [v for k, v in experiment_results.items() 
                                 if defense in k and 'report' in v]
            if defense_experiments:
                avg_asr = np.mean([exp['report'].get('final_asr', 0) 
                                 for exp in defense_experiments])
                defense_asrs.append(avg_asr)
            else:
                defense_asrs.append(0)
        
        plt.bar(defense_types, defense_asrs, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
        plt.title('Average ASR by Defense Type')
        plt.ylabel('Average ASR')
        plt.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.output_dir}/attack_progression_analysis.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_hospital_performance_analysis(self, hospital_data, save_path=None):
        """Analyze performance across different hospitals."""
        plt.figure(figsize=(12, 8))
        
        hospital_names = list(hospital_data.keys())
        train_sizes = [data['total_samples'] for data in hospital_data.values()]
        
        plt.subplot(2, 2, 1)
        plt.bar(range(len(hospital_names)), train_sizes, color='lightblue')
        plt.title('Data Distribution Across Hospitals')
        plt.xlabel('Hospital')
        plt.ylabel('Number of Samples')
        plt.xticks(range(len(hospital_names)), [f'H{i+1}' for i in range(len(hospital_names))], 
                   rotation=45)
        
        plt.subplot(2, 2, 2)
        # Simulate performance metrics for each hospital
        performances = np.random.normal(0.8, 0.1, len(hospital_names))
        plt.scatter(train_sizes, performances, s=100, alpha=0.7, color='coral')
        plt.title('Performance vs Data Size')
        plt.xlabel('Training Samples')
        plt.ylabel('Simulated Performance')
        
        plt.subplot(2, 2, 3)
        # Class distribution analysis
        class_ratios = []
        for data in hospital_data.values():
            if len(data['y_train']) > 0:
                unique, counts = np.unique(data['y_train'].numpy(), return_counts=True)
                ratio = counts[1] / len(data['y_train']) if len(counts) > 1 else 0.5
                class_ratios.append(ratio)
            else:
                class_ratios.append(0.5)
        
        plt.bar(range(len(hospital_names)), class_ratios, color='lightgreen')
        plt.title('Class Distribution (Positive Class Ratio)')
        plt.xlabel('Hospital')
        plt.ylabel('Positive Class Ratio')
        plt.xticks(range(len(hospital_names)), [f'H{i+1}' for i in range(len(hospital_names))],
                   rotation=45)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Balanced')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        # Data quality simulation
        quality_scores = np.random.uniform(0.7, 1.0, len(hospital_names))
        colors = plt.cm.RdYlGn(quality_scores)
        plt.bar(range(len(hospital_names)), quality_scores, color=colors)
        plt.title('Simulated Data Quality Scores')
        plt.xlabel('Hospital')
        plt.ylabel('Quality Score')
        plt.xticks(range(len(hospital_names)), [f'H{i+1}' for i in range(len(hospital_names))],
                   rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.output_dir}/hospital_analysis.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_comprehensive_report(self, experiment_results, hospital_data=None):
        """Create comprehensive analysis report."""
        print("Generating Comprehensive Security Analysis Report")
        print("=" * 60)
        
        # Create all visualizations
        if isinstance(experiment_results, dict) and len(experiment_results) > 0:
            # Extract summary data
            summary_data = []
            for key, result in experiment_results.items():
                if 'report' in result:
                    parts = key.split('_vs_')
                    if len(parts) >= 2:
                        attack = parts[0]
                        defense = parts[1].split('_m')[0]  # Remove malicious client info
                        
                        summary_data.append({
                            'attack': attack,
                            'defense': defense,
                            'final_asr': result['report'].get('final_asr', 0),
                            'final_accuracy': result['report'].get('final_accuracy', 0),
                            'max_asr': result['report'].get('max_asr', 0),
                            'min_accuracy': result['report'].get('min_accuracy', 0)
                        })
            
            if summary_data:
                # Create visualizations
                self.create_attack_defense_heatmap(summary_data)
                self.create_defense_effectiveness_comparison(summary_data)
                self.create_attack_progression_analysis(experiment_results)
                
                # Save detailed report
                self.save_detailed_report(summary_data, experiment_results)
        
        if hospital_data:
            self.create_hospital_performance_analysis(hospital_data)
        
        print(f"Comprehensive report generated in {self.output_dir}")
    
    def save_detailed_report(self, summary_data, experiment_results):
        """Save detailed analysis report."""
        report_path = f"{self.output_dir}/detailed_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Federated Learning Security Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            df = pd.DataFrame(summary_data)
            f.write("## Executive Summary\n\n")
            f.write(f"- Total experiments conducted: {len(summary_data)}\n")
            f.write(f"- Attack types tested: {len(df['attack'].unique())}\n")
            f.write(f"- Defense mechanisms evaluated: {len(df['defense'].unique())}\n")
            f.write(f"- Average final ASR: {df['final_asr'].mean():.3f}\n")
            f.write(f"- Average final accuracy: {df['final_accuracy'].mean():.3f}\n\n")
            
            # Best and worst performers
            f.write("## Key Findings\n\n")
            
            # Best defenses
            best_defenses = df.groupby('defense')['final_asr'].mean().sort_values()
            f.write("### Most Effective Defenses (by lowest ASR):\n")
            for defense, asr in best_defenses.head(3).items():
                f.write(f"1. **{defense}**: {asr:.3f} average ASR\n")
            f.write("\n")
            
            # Most dangerous attacks
            worst_attacks = df.groupby('attack')['final_asr'].mean().sort_values(ascending=False)
            f.write("### Most Successful Attacks (by highest ASR):\n")
            for i, (attack, asr) in enumerate(worst_attacks.head(3).items(), 1):
                f.write(f"{i}. **{attack}**: {asr:.3f} average ASR\n")
            f.write("\n")
            
            # Detailed results table
            f.write("## Detailed Results\n\n")
            f.write("| Attack | Defense | Final ASR | Final Accuracy | Max ASR |\n")
            f.write("|--------|---------|-----------|----------------|----------|\n")
            
            for row in summary_data:
                f.write(f"| {row['attack']} | {row['defense']} | "
                       f"{row['final_asr']:.3f} | {row['final_accuracy']:.3f} | "
                       f"{row['max_asr']:.3f} |\n")
            
            f.write("\n## Recommendations\n\n")
            
            # Generate recommendations based on results
            best_overall = df.loc[df['final_asr'].idxmin()]
            f.write(f"1. **Recommended Defense**: {best_overall['defense']} showed the lowest ASR "
                   f"({best_overall['final_asr']:.3f}) against {best_overall['attack']} attacks.\n\n")
            
            worst_overall = df.loc[df['final_asr'].idxmax()]
            f.write(f"2. **Highest Risk**: {worst_overall['attack']} attacks achieved "
                   f"{worst_overall['final_asr']:.3f} ASR against {worst_overall['defense']}, "
                   f"requiring additional countermeasures.\n\n")
            
            # Accuracy preservation analysis
            high_acc_defenses = df[df['final_accuracy'] > 0.8]['defense'].unique()
            if len(high_acc_defenses) > 0:
                f.write(f"3. **Accuracy Preservation**: {', '.join(high_acc_defenses)} "
                       f"maintained accuracy > 0.8 while providing security.\n\n")
            
            f.write("4. **Future Work**: Consider implementing ensemble defenses combining "
                   f"multiple mechanisms for enhanced protection.\n\n")
        
        print(f"Detailed report saved to {report_path}")

# Utility functions for enhanced experiments
def create_adversarial_scenarios():
    """Create sophisticated adversarial scenarios for testing."""
    return {
        'coordinated_attack': {
            'description': 'Multiple malicious clients coordinate their attacks',
            'malicious_clients': 3,
            'poison_rate': 0.15,
            'attack_types': ['label_flip', 'trigger']
        },
        'adaptive_attack': {
            'description': 'Attacks adapt based on detected defense mechanism',
            'malicious_clients': 2,
            'poison_rate': 0.1,
            'attack_types': ['adaptive']
        },
        'stealth_attack': {
            'description': 'Low-intensity attacks designed to evade detection',
            'malicious_clients': 1,
            'poison_rate': 0.05,
            'attack_types': ['semantic', 'delayed_activation']
        },
        'hybrid_attack': {
            'description': 'Combination of multiple attack vectors',
            'malicious_clients': 2,
            'poison_rate': 0.12,
            'attack_types': ['trigger', 'semantic']
        }
    }

def analyze_defense_robustness(experiment_results):
    """Analyze robustness of defense mechanisms across different attack intensities."""
    robustness_scores = {}
    
    for key, result in experiment_results.items():
        if 'report' in result:
            parts = key.split('_vs_')
            if len(parts) >= 2:
                defense = parts[1].split('_m')[0]
                
                if defense not in robustness_scores:
                    robustness_scores[defense] = {
                        'asr_scores': [],
                        'accuracy_scores': [],
                        'experiments': 0
                    }
                
                robustness_scores[defense]['asr_scores'].append(
                    1 - result['report'].get('final_asr', 1)  # Higher is better
                )
                robustness_scores[defense]['accuracy_scores'].append(
                    result['report'].get('final_accuracy', 0)
                )
                robustness_scores[defense]['experiments'] += 1
    
    # Calculate overall robustness score
    for defense, scores in robustness_scores.items():
        if scores['experiments'] > 0:
            avg_asr_resistance = np.mean(scores['asr_scores'])
            avg_accuracy = np.mean(scores['accuracy_scores'])
            # Weighted combination: 60% ASR resistance, 40% accuracy preservation
            overall_score = 0.6 * avg_asr_resistance + 0.4 * avg_accuracy
            scores['overall_robustness'] = overall_score
    
    return robustness_scores