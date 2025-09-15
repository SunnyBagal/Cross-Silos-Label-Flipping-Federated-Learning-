
# ai_fl_test.py - Quick Test Script
#!/usr/bin/env python3
"""
Quick test script for AI-Integrated FL Security System
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        # Core AI components
        from AI_Driven_attack import AIAttackDetector, ClientBehaviorAnalyzer, AdaptiveDefenseSystem
        print("✓ AI Attack Detector modules imported")
        
        from AI_Enhanced_Defense import AIEnhancedDefenseMechanisms, SmartAnomalyDetector
        print("✓ AI Enhanced Defense modules imported")
        
        from AI_Enhanced_comprehensive_exp_runner import AIEnhancedExperimentRunner
        print("✓ AI Enhanced Experiment Runner imported")
        
        from AI_Enhanced_metrices import AIEnhancedEvaluator
        print("✓ AI Enhanced Evaluator imported")
        
        from main_AI_integrated_runner import AIIntegratedFLSecuritySystem
        print("✓ Main AI Integrated System imported")
        
        # Existing components
        from model import HealthNet
        from advanced_attacks import AdvancedAttacks
        print("✓ Existing components imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_ai_detector():
    """Test AI detector functionality."""
    print("\nTesting AI Detector...")
    
    try:
        from AI_Driven_attack import ClientBehaviorAnalyzer
        
        analyzer = ClientBehaviorAnalyzer(window_size=5, feature_dim=15)
        
        # Simulate client data
        for round_num in range(3):
            for client_id in range(3):
                # Create dummy update and metrics
                dummy_update = torch.randn(50)  # Simulated model update
                dummy_metrics = {
                    'accuracy': 0.8 + np.random.normal(0, 0.1),
                    'loss': 0.3 + np.random.normal(0, 0.05),
                    'confidence': 0.75 + np.random.normal(0, 0.05)
                }
                
                is_malicious = client_id == 2  # Make last client malicious
                analyzer.update_client_history(
                    f"client_{client_id}", dummy_update, dummy_metrics, 
                    round_num, is_malicious
                )
        
        # Try training (should work with minimal data)
        training_success = analyzer.train_ai_models()
        print(f"✓ AI training completed: {'Success' if training_success else 'Insufficient data'}")
        
        # Test detection
        if training_success:
            malicious_scores = analyzer.detect_malicious_clients()
            print(f"✓ Detection completed, found {len(malicious_scores)} clients")
        
        return True
        
    except Exception as e:
        print(f"✗ AI Detector test failed: {e}")
        return False

def test_ai_defenses():
    """Test AI-enhanced defense mechanisms."""
    print("\nTesting AI Defenses...")
    
    try:
        from AI_Enhanced_Defense import AIEnhancedDefenseMechanisms
        
        defenses = AIEnhancedDefenseMechanisms()
        
        # Create dummy client updates
        client_updates = [torch.randn(100) for _ in range(4)]
        client_ids = [f"client_{i}" for i in range(4)]
        malicious_scores = {
            "client_0": 0.1,
            "client_1": 0.2, 
            "client_2": 0.8,  # Malicious
            "client_3": 0.1
        }
        
        # Test different defense mechanisms
        trust_result = defenses.trust_weighted_aggregation(
            client_updates, malicious_scores, client_ids
        )
        print(f"✓ Trust-weighted aggregation: {trust_result.shape}")
        
        krum_result = defenses.adaptive_krum(
            client_updates, malicious_scores, client_ids, num_malicious=1
        )
        print(f"✓ Adaptive Krum: {krum_result.shape}")
        
        trimmed_result = defenses.intelligent_trimmed_mean(
            client_updates, malicious_scores, client_ids
        )
        print(f"✓ Intelligent trimmed mean: {trimmed_result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ AI Defenses test failed: {e}")
        return False

def test_experiment_runner():
    """Test the AI-enhanced experiment runner."""
    print("\nTesting AI Enhanced Experiment Runner...")
    
    try:
        from AI_Enhanced_comprehensive_exp_runner import AIEnhancedExperimentRunner
        
        # Create small-scale runner for testing
        runner = AIEnhancedExperimentRunner(
            num_clients=3,
            num_rounds=2
        )
        
        # Run minimal experiment
        report, evaluator = runner.run_ai_enhanced_experiment(
            attack_type="label_flip",
            defense_type="ai_adaptive",
            malicious_clients=1,
            poison_rate=0.1,
            enable_ai_training=False,  # Skip training for quick test
            verbose=False
        )
        
        print(f"✓ Experiment completed")
        print(f"  Final accuracy: {report.get('final_accuracy', 0):.3f}")
        print(f"  Final ASR: {report.get('final_asr', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Experiment Runner test failed: {e}")
        return False

def test_main_system():
    """Test the main integrated system."""
    print("\nTesting Main AI Integrated System...")
    
    try:
        from main_AI_integrated_runner import AIIntegratedFLSecuritySystem
        
        # Initialize with minimal config
        system = AIIntegratedFLSecuritySystem()
        
        # Override config for quick test
        system.config.update({
            'num_clients': 3,
            'num_rounds': 2,
            'enable_ai_training': False
        })
        
        # Run single experiment
        report, runner = system.run_single_ai_experiment(
            attack_type="trigger",
            defense_type="ai_adaptive",
            malicious_clients=1,
            poison_rate=0.1,
            verbose=False
        )
        
        print(f"✓ Main system test completed")
        print(f"  System experiments: {system.system_metrics['total_experiments']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Main System test failed: {e}")
        return False

def run_performance_benchmark():
    """Run a quick performance benchmark."""
    print("\nRunning Performance Benchmark...")
    
    try:
        import time
        from main_AI_integrated_runner import AIIntegratedFLSecuritySystem
        
        system = AIIntegratedFLSecuritySystem()
        system.config.update({
            'num_clients': 5,
            'num_rounds': 3,
            'enable_ai_training': True
        })
        
        start_time = time.time()
        
        # Run a few experiments
        test_scenarios = [
            ("label_flip", "ai_adaptive"),
            ("trigger", "adaptive_krum"),
            ("semantic", "trust_weighted")
        ]
        
        for attack, defense in test_scenarios:
            report, _ = system.run_single_ai_experiment(
                attack_type=attack,
                defense_type=defense,
                malicious_clients=1,
                poison_rate=0.1,
                verbose=False
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✓ Benchmark completed")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Time per experiment: {total_time/len(test_scenarios):.2f} seconds")
        print(f"  Experiments completed: {system.system_metrics['total_experiments']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AI-INTEGRATED FL SECURITY SYSTEM - COMPREHENSIVE TEST")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Import Test", test_imports),
        ("AI Detector Test", test_ai_detector),
        ("AI Defenses Test", test_ai_defenses),
        ("Experiment Runner Test", test_experiment_runner),
        ("Main System Test", test_main_system),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The AI-integrated system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python main_ai_integrated_runner.py --mode demo")
        print("2. Run: python main_ai_integrated_runner.py --mode single --attack trigger --defense ai_adaptive")
        print("3. Run: python main_ai_integrated_runner.py --mode comprehensive")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all required packages are installed: pip install -r requirements.txt")
        print("2. Check that all files are in the correct directory")
        print("3. Verify that the model.py file contains the HealthNet class")
    
    print("=" * 60)

if __name__ == "__main__":
    main()