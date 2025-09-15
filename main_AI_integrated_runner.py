#!/usr/bin/env python3
"""
Main AI-Integrated Federated Learning Security System
Comprehensive system integrating AI-driven attack detection and adaptive defenses
"""

import torch
import numpy as np
import argparse
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Import AI-enhanced components
from AI_Driven_attack import AIAttackDetector, ClientBehaviorAnalyzer, AdaptiveDefenseSystem
from AI_Enhanced_Defense import AIEnhancedDefenseMechanisms, SmartAnomalyDetector
from AI_Enhanced_comprehensive_exp_runner import AIEnhancedExperimentRunner
from AI_Enhanced_metrices import AIEnhancedEvaluator

# Import existing components
from model import HealthNet
from advanced_attacks import AdvancedAttacks

# Import data management if available
try:
    from integrated_main_runner import BuiltInHealthcareDataManager
    HEALTHCARE_DATA_AVAILABLE = True
except ImportError:
    HEALTHCARE_DATA_AVAILABLE = False
    print("Healthcare data manager not available. Using synthetic data only.")

class AIIntegratedFLSecuritySystem:
    """Complete AI-integrated FL security system."""
    
    def __init__(self, config_path=None):
        """Initialize the AI-integrated system."""
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Initialize components
        self.ai_detector = ClientBehaviorAnalyzer(
            window_size=self.config.get('detection_window_size', 10),
            feature_dim=self.config.get('feature_dim', 20)
        )
        
        self.adaptive_defense = AdaptiveDefenseSystem()
        self.ai_defenses = AIEnhancedDefenseMechanisms()
        self.ai_evaluator = AIEnhancedEvaluator()
        
        # Data management
        self.data_manager = None
        if HEALTHCARE_DATA_AVAILABLE and self.config.get('use_real_data', True):
            try:
                self.data_manager = BuiltInHealthcareDataManager(
                    csv_path=self.config.get('data_path', 'data/healthcare_dataset.csv')
                )
                print("Real healthcare data manager initialized")
            except Exception as e:
                print(f"Could not initialize healthcare data manager: {e}")
                self.data_manager = None
        
        # Experiment tracking
        self.experiment_results = {}
        self.system_metrics = {
            'total_experiments': 0,
            'successful_detections': 0,
            'successful_defenses': 0,
            'system_start_time': datetime.now()
        }
    
    def _default_config(self):
        """Default system configuration."""
        return {
            'num_clients': 6,
            'num_rounds': 10,
            'detection_window_size': 10,
            'feature_dim': 20,
            'use_real_data': True,
            'enable_ai_training': True,
            'save_results': True,
            'output_dir': 'ai_fl_results',
            'model_save_path': 'models/ai_fl_models.pth',
            'log_level': 'INFO'
        }
    
    def _load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            print(f"Could not load config from {config_path}: {e}")
            return self._default_config()
    
    def run_single_ai_experiment(self, attack_type="trigger", defense_type="ai_adaptive",
                                malicious_clients=1, poison_rate=0.1, verbose=True):
        """Run a single AI-enhanced experiment."""
        
        if verbose:
            print(f"\n🤖 AI-Integrated FL Security Experiment")
            print(f"Attack: {attack_type} | Defense: {defense_type}")
            print(f"Malicious clients: {malicious_clients} | Poison rate: {poison_rate}")
            print("-" * 60)
        
        # Create experiment runner
        runner = AIEnhancedExperimentRunner(
            num_clients=self.config['num_clients'],
            num_rounds=self.config['num_rounds']
        )
        
        # If real data is available, set up data generator
        if self.data_manager:
            try:
                processed_data = self.data_manager.load_and_preprocess_data()
                if processed_data:
                    def real_data_generator():
                        return processed_data['X'], processed_data['y']
                    runner.data_generator = real_data_generator
                    print("✅ Using real healthcare data")
                else:
                    print("⚠️ Could not load real data, using synthetic data")
            except Exception as e:
                print(f"⚠️ Real data loading failed: {e}, using synthetic data")
        
        # Run experiment
        start_time = time.time()
        
        report, evaluator = runner.run_ai_enhanced_experiment(
            attack_type=attack_type,
            defense_type=defense_type,
            malicious_clients=malicious_clients,
            poison_rate=poison_rate,
            enable_ai_training=self.config['enable_ai_training'],
            verbose=verbose
        )
        
        end_time = time.time()
        experiment_duration = end_time - start_time
        
        # Store results
        experiment_key = f"ai_{attack_type}_vs_{defense_type}_m{malicious_clients}_p{poison_rate}"
        self.experiment_results[experiment_key] = {
            'report': report,
            'duration': experiment_duration,
            'config': {
                'attack_type': attack_type,
                'defense_type': defense_type,
                'malicious_clients': malicious_clients,
                'poison_rate': poison_rate
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Update system metrics
        self.system_metrics['total_experiments'] += 1
        if report.get('final_asr', 1.0) < 0.3:  # Consider defense successful if ASR < 30%
            self.system_metrics['successful_defenses'] += 1
        
        ai_detections = report.get('total_ai_detections', 0)
        if ai_detections > 0:
            self.system_metrics['successful_detections'] += 1
        
        if verbose:
            print(f"\n📊 Experiment Results:")
            print(f"   Duration: {experiment_duration:.2f} seconds")
            print(f"   Final Accuracy: {report.get('final_accuracy', 0):.4f}")
            print(f"   Final ASR: {report.get('final_asr', 0):.4f}")
            print(f"   AI Detections: {ai_detections}")
            print(f"   Defense Adaptations: {report.get('defense_adaptations', 0)}")
            
            # AI-specific insights
            if 'ai_detection_performance' in report:
                ai_perf = report['ai_detection_performance']
                print(f"   AI Precision: {ai_perf.get('precision', 0):.3f}")
                print(f"   AI Recall: {ai_perf.get('recall', 0):.3f}")
                print(f"   AI F1-Score: {ai_perf.get('f1_score', 0):.3f}")
        
        return report, runner
    
    def run_comprehensive_ai_evaluation(self):
        """Run comprehensive evaluation of AI-integrated system."""
        print("\n🚀 Starting Comprehensive AI-Integrated FL Security Evaluation")
        print("=" * 80)
        
        # Define test scenarios
        attack_scenarios = [
            ("label_flip", "Standard label flipping attack"),
            ("trigger", "Backdoor trigger attack"),
            ("adaptive_label_flip", "AI-evasive label flipping"),
            ("stealth_trigger", "Stealth backdoor attack"),
            ("semantic", "Semantic pattern attack"),
            ("coordinated", "Multi-vector coordinated attack")
        ]
        
        defense_scenarios = [
            ("ai_adaptive", "AI-driven adaptive defense"),
            ("adaptive_krum", "AI-enhanced Krum"),
            ("intelligent_trimmed_mean", "Intelligent trimmed mean"),
            ("trust_weighted", "Trust-weighted aggregation"),
            ("cluster_filtering", "Cluster-based filtering"),
            ("fedavg", "Standard FedAvg (baseline)")
        ]
        
        total_experiments = len(attack_scenarios) * len(defense_scenarios)
        experiment_count = 0
        
        comprehensive_results = []
        
        for attack_type, attack_desc in attack_scenarios:
            for defense_type, defense_desc in defense_scenarios:
                experiment_count += 1
                
                print(f"\n[{experiment_count}/{total_experiments}] Testing: {attack_desc} vs {defense_desc}")
                
                try:
                    report, runner = self.run_single_ai_experiment(
                        attack_type=attack_type,
                        defense_type=defense_type,
                        malicious_clients=1,
                        poison_rate=0.1,
                        verbose=False
                    )
                    
                    comprehensive_results.append({
                        'attack_type': attack_type,
                        'attack_description': attack_desc,
                        'defense_type': defense_type,
                        'defense_description': defense_desc,
                        'final_accuracy': report.get('final_accuracy', 0),
                        'final_asr': report.get('final_asr', 0),
                        'ai_detections': report.get('total_ai_detections', 0),
                        'defense_adaptations': report.get('defense_adaptations', 0),
                        'system_intelligence_score': report.get('system_intelligence_metrics', {}).get('overall_intelligence_score', 0)
                    })
                    
                    print(f"   ✅ Completed - Acc: {report.get('final_accuracy', 0):.3f}, ASR: {report.get('final_asr', 0):.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    comprehensive_results.append({
                        'attack_type': attack_type,
                        'defense_type': defense_type,
                        'error': str(e),
                        'final_accuracy': 0,
                        'final_asr': 1,
                        'ai_detections': 0
                    })
        
        # Analyze and present results
        self._analyze_comprehensive_results(comprehensive_results)
        
        return comprehensive_results
    
    def _analyze_comprehensive_results(self, results):
        """Analyze and present comprehensive evaluation results."""
        print("\n" + "=" * 100)
        print("AI-INTEGRATED FL SECURITY SYSTEM - COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 100)
        
        # Results table
        print(f"{'Attack Type':<20} {'Defense Type':<20} {'Accuracy':<10} {'ASR':<8} {'AI Det':<8} {'Adapt':<8} {'Intel':<8}")
        print("-" * 100)
        
        for result in results:
            if 'error' not in result:
                print(f"{result['attack_type']:<20} {result['defense_type']:<20} "
                      f"{result['final_accuracy']:<10.3f} {result['final_asr']:<8.3f} "
                      f"{result['ai_detections']:<8} {result['defense_adaptations']:<8} "
                      f"{result.get('system_intelligence_score', 0):<8.3f}")
        
        # Performance analysis
        print("\n" + "=" * 100)
        print("PERFORMANCE ANALYSIS")
        print("=" * 100)
        
        # AI-adaptive vs traditional defenses
        ai_adaptive_results = [r for r in results if r['defense_type'] == 'ai_adaptive' and 'error' not in r]
        traditional_results = [r for r in results if r['defense_type'] == 'fedavg' and 'error' not in r]
        
        if ai_adaptive_results and traditional_results:
            ai_avg_asr = np.mean([r['final_asr'] for r in ai_adaptive_results])
            trad_avg_asr = np.mean([r['final_asr'] for r in traditional_results])
            ai_avg_acc = np.mean([r['final_accuracy'] for r in ai_adaptive_results])
            trad_avg_acc = np.mean([r['final_accuracy'] for r in traditional_results])
            
            asr_improvement = ((trad_avg_asr - ai_avg_asr) / trad_avg_asr) * 100 if trad_avg_asr > 0 else 0
            acc_improvement = ((ai_avg_acc - trad_avg_acc) / trad_avg_acc) * 100 if trad_avg_acc > 0 else 0
            
            print(f"AI-Adaptive Defense Performance:")
            print(f"  Average ASR: {ai_avg_asr:.3f} (vs Traditional: {trad_avg_asr:.3f})")
            print(f"  Average Accuracy: {ai_avg_acc:.3f} (vs Traditional: {trad_avg_acc:.3f})")
            print(f"  ASR Improvement: {asr_improvement:.1f}%")
            print(f"  Accuracy Improvement: {acc_improvement:.1f}%")
        
        # Best performing combinations
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_overall = min(valid_results, key=lambda x: x['final_asr'])
            best_accuracy = max(valid_results, key=lambda x: x['final_accuracy'])
            
            print(f"\nBest Attack Resilience:")
            print(f"  {best_overall['attack_type']} vs {best_overall['defense_type']}")
            print(f"  ASR: {best_overall['final_asr']:.3f}, Accuracy: {best_overall['final_accuracy']:.3f}")
            
            print(f"\nBest Accuracy Preservation:")
            print(f"  {best_accuracy['attack_type']} vs {best_accuracy['defense_type']}")
            print(f"  Accuracy: {best_accuracy['final_accuracy']:.3f}, ASR: {best_accuracy['final_asr']:.3f}")
        
        # AI system statistics
        total_ai_detections = sum(r.get('ai_detections', 0) for r in valid_results)
        total_adaptations = sum(r.get('defense_adaptations', 0) for r in valid_results)
        avg_intelligence = np.mean([r.get('system_intelligence_score', 0) for r in valid_results])
        
        print(f"\nAI System Statistics:")
        print(f"  Total AI Detections: {total_ai_detections}")
        print(f"  Total Defense Adaptations: {total_adaptations}")
        print(f"  Average System Intelligence Score: {avg_intelligence:.3f}")
        print(f"  Detection Rate: {total_ai_detections / len(valid_results):.1f} per experiment")
    
    def save_all_results(self, output_dir=None):
        """Save all results and models."""
        if output_dir is None:
            output_dir = self.config['output_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save experiment results
        results_file = os.path.join(output_dir, f"ai_fl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            # Prepare JSON-serializable data
            json_data = {
                'system_config': self.config,
                'system_metrics': self.system_metrics,
                'experiment_results': {}
            }
            
            for key, result in self.experiment_results.items():
                json_data['experiment_results'][key] = {
                    'report': result['report'],
                    'duration': result['duration'],
                    'config': result['config'],
                    'timestamp': result['timestamp']
                }
            
            with open(results_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            print(f"✅ Results saved to {results_file}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
        
        # Save AI models if training was enabled
        if self.config['enable_ai_training']:
            model_path = os.path.join(output_dir, 'ai_models.pth')
            try:
                self.ai_detector.save_model(model_path)
                print(f"✅ AI models saved to {model_path}")
            except Exception as e:
                print(f"❌ Error saving AI models: {e}")
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        print("\n" + "=" * 100)
        print("AI-INTEGRATED FEDERATED LEARNING SECURITY SYSTEM - FINAL REPORT")
        print("=" * 100)
        
        runtime = datetime.now() - self.system_metrics['system_start_time']
        
        print(f"System Runtime: {runtime}")
        print(f"Total Experiments: {self.system_metrics['total_experiments']}")
        print(f"Successful AI Detections: {self.system_metrics['successful_detections']}")
        print(f"Successful Defenses: {self.system_metrics['successful_defenses']}")
        
        if self.system_metrics['total_experiments'] > 0:
            detection_rate = self.system_metrics['successful_detections'] / self.system_metrics['total_experiments']
            defense_success_rate = self.system_metrics['successful_defenses'] / self.system_metrics['total_experiments']
            
            print(f"AI Detection Success Rate: {detection_rate:.1%}")
            print(f"Defense Success Rate: {defense_success_rate:.1%}")
        
        print("\nKey Achievements:")
        print("✅ AI-driven attack detection and classification")
        print("✅ Adaptive defense mechanism selection")
        print("✅ Real-time threat assessment")
        print("✅ Comprehensive security evaluation")
        
        print("\nRecommendations for Production Deployment:")
        print("1. Continue training AI models with diverse attack patterns")
        print("2. Implement continuous learning for evolving threats")
        print("3. Monitor and update defense mechanism effectiveness")
        print("4. Establish threat intelligence sharing protocols")
        print("5. Regular security audits and model retraining")
        
        print("=" * 100)

def main():
    """Main entry point for the AI-integrated FL security system."""
    parser = argparse.ArgumentParser(
        description='AI-Integrated Federated Learning Security System'
    )
    
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'comprehensive', 'demo', 'benchmark'],
                       help='Execution mode')
    
    parser.add_argument('--attack', type=str, default='trigger',
                       choices=['label_flip', 'trigger', 'adaptive_label_flip', 
                               'stealth_trigger', 'semantic', 'coordinated'],
                       help='Attack type for single mode')
    
    parser.add_argument('--defense', type=str, default='ai_adaptive',
                       choices=['ai_adaptive', 'adaptive_krum', 'intelligent_trimmed_mean',
                               'trust_weighted', 'cluster_filtering', 'fedavg'],
                       help='Defense type for single mode')
    
    parser.add_argument('--malicious', type=int, default=1,
                       help='Number of malicious clients')
    
    parser.add_argument('--poison-rate', type=float, default=0.1,
                       help='Poison rate for attacks')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration JSON file')
    
    parser.add_argument('--output-dir', type=str, default='ai_fl_results',
                       help='Output directory for results')
    
    parser.add_argument('--no-ai-training', action='store_true',
                       help='Disable AI model training')
    
    args = parser.parse_args()
    
    # Initialize system
    print("🤖 AI-Integrated Federated Learning Security System")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Configuration: {'Custom' if args.config else 'Default'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    system = AIIntegratedFLSecuritySystem(config_path=args.config)
    
    # Override config with command line arguments
    if args.output_dir != 'ai_fl_results':
        system.config['output_dir'] = args.output_dir
    if args.no_ai_training:
        system.config['enable_ai_training'] = False
    
    try:
        if args.mode == 'single':
            system.run_single_ai_experiment(
                attack_type=args.attack,
                defense_type=args.defense,
                malicious_clients=args.malicious,
                poison_rate=args.poison_rate,
                verbose=True
            )
        
        elif args.mode == 'comprehensive':
            system.run_comprehensive_ai_evaluation()
        
        elif args.mode == 'demo':
            # Quick demo with key scenarios
            print("\n🎯 Running AI Security Demo...")
            demo_scenarios = [
                ('trigger', 'ai_adaptive'),
                ('adaptive_label_flip', 'ai_adaptive'),
                ('coordinated', 'ai_adaptive'),
                ('trigger', 'fedavg')  # Baseline comparison
            ]
            
            for attack, defense in demo_scenarios:
                print(f"\n--- Demo: {attack} vs {defense} ---")
                system.run_single_ai_experiment(
                    attack_type=attack,
                    defense_type=defense,
                    malicious_clients=1,
                    poison_rate=0.1,
                    verbose=True
                )
        
        elif args.mode == 'benchmark':
            print("\n⚡ Running AI Security Benchmark...")
            # Focused benchmark on key metrics
            benchmark_results = system.run_comprehensive_ai_evaluation()
            
            # Additional benchmark analysis
            print("\n📈 Benchmark Analysis Complete")
    
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
    
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save results and generate final report
        if system.experiment_results:
            system.save_all_results()
            system.generate_final_report()
        else:
            print("⚠️ No results to save")

if __name__ == "__main__":
    main()