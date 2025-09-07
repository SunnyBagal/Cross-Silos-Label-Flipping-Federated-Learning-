#!/usr/bin/env python3


import torch
import sys
import traceback

def test_imports():
    """Test all imports work correctly."""
    print("Testing imports...")
    
    try:
        from advanced_attacks import AdvancedAttacks
        print("✓ AdvancedAttacks imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AdvancedAttacks: {e}")
        return False
    
    try:
        from defence_mechanisms import DefenseMechanisms, AnomalyDetector
        print("✓ DefenseMechanisms imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DefenseMechanisms: {e}")
        return False
    
    try:
        from evaluation_metrics import ComprehensiveEvaluator
        print("✓ ComprehensiveEvaluator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ComprehensiveEvaluator: {e}")
        return False
    
    try:
        from comprehensive_experiment_runner import ComprehensiveExperimentRunner
        print("✓ ComprehensiveExperimentRunner imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ComprehensiveExperimentRunner: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of each component."""
    print("\nTesting basic functionality...")
    
    try:
        # Test data generation
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        print("✓ Data generation works")
        
        # Test attacks
        from advanced_attacks import AdvancedAttacks
        X_attacked, y_attacked = AdvancedAttacks.semantic_attack(
            None, (X_tensor, y_tensor), target_class=1
        )
        print("✓ Semantic attack works")
        
        # Test defenses
        from defence_mechanisms import DefenseMechanisms
        defense = DefenseMechanisms()
        dummy_updates = [torch.randn(10) for _ in range(3)]
        result = defense.trimmed_mean_aggregation(dummy_updates)
        print("✓ Defense mechanisms work")
        
        # Test evaluator
        from evaluation_metrics import ComprehensiveEvaluator
        from model import HealthNet
        
        evaluator = ComprehensiveEvaluator()
        model = HealthNet(5, 2)
        metrics = evaluator.evaluate_round(model, (X_tensor[:20], y_tensor[:20]), round_num=0)
        print("✓ Evaluation metrics work")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_mini_experiment():
    """Run a minimal experiment to test integration."""
    print("\nRunning mini experiment...")
    
    try:
        from comprehensive_experiment_runner import ComprehensiveExperimentRunner
        
        runner = ComprehensiveExperimentRunner(
            num_clients=3, 
            num_rounds=2
        )
        
        report, evaluator = runner.run_single_experiment(
            attack_type="label_flip",
            defense_type="fedavg",
            malicious_clients=1,
            poison_rate=0.1,
            verbose=False
        )
        
        print(f"✓ Mini experiment completed successfully")
        print(f"  Final accuracy: {report.get('final_accuracy', 0):.3f}")
        print(f"  Final ASR: {report.get('final_asr', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mini experiment failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("QUICK TEST SUITE")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_passed = False
    
    # Test 2: Basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test 3: Mini experiment
    if not test_mini_experiment():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python run_advanced_experiments.py --mode components")
        print("2. Run: python run_advanced_experiments.py --mode single")
        print("3. Run: python run_advanced_experiments.py --mode comprehensive")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all files are in the same directory")
        print("2. Check that model.py exists and contains HealthNet class")
        print("3. Install missing dependencies: pip install -r requirements.txt")
    
    print("=" * 50)

if __name__ == "__main__":
    main()