import torch
import numpy as np
import pandas as pd
import argparse
import json
import time
from datetime import datetime
import os
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing components
from advanced_attacks import AdvancedAttacks
from defence_mechanisms import DefenseMechanisms, AnomalyDetector
from evaluation_metrics import ComprehensiveEvaluator
from model import HealthNet

class BuiltInHealthcareDataManager:
    """Built-in healthcare data manager - no external imports needed."""
    
    def __init__(self, csv_path="/Users/sunny/cap_test_dem/data/healthcare_dataset_7000.csv"):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_encoders = {}
        self.hospitals = []
        self.processed_data = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the healthcare dataset."""
        print(f"Loading healthcare data from {self.csv_path}")
        
        try:
            df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")
            print(f"Columns: {list(df.columns)}")
            
            # Display first few rows for verification
            print("\nFirst 3 rows of data:")
            print(df.head(3).to_string())
            
            # Get unique hospitals
            if 'Hospital' in df.columns:
                self.hospitals = df['Hospital'].dropna().unique().tolist()
                print(f"\n✓ Found {len(self.hospitals)} hospitals:")
                for i, hospital in enumerate(self.hospitals, 1):
                    hospital_count = (df['Hospital'] == hospital).sum()
                    print(f"  {i}. {hospital}: {hospital_count} records")
            else:
                print("❌ No 'Hospital' column found in dataset")
                raise ValueError("Dataset must contain 'Hospital' column")
            
            # Auto-detect target column
            target_col = self._detect_target_column(df)
            print(f"\n✓ Using '{target_col}' as target variable")
            
            # Create binary target if needed
            if target_col not in df.columns:
                # Create binary target from a numeric column if no clear target exists
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != 'Hospital']
                
                if len(numeric_cols) > 0:
                    source_col = numeric_cols[0]
                    median_val = df[source_col].median()
                    df['target'] = (df[source_col] > median_val).astype(int)
                    target_col = 'target'
                    print(f"✓ Created binary target from {source_col} (threshold: {median_val})")
                else:
                    raise ValueError("No suitable target column found")
            
            # Encode target variable
            if df[target_col].dtype == 'object':
                df['target_encoded'] = self.label_encoder.fit_transform(df[target_col])
                target_col = 'target_encoded'
            else:
                df['target_encoded'] = df[target_col]
                target_col = 'target_encoded'
            
            # Prepare features (exclude Hospital and target columns)
            exclude_cols = ['Hospital', target_col] + [col for col in df.columns if 'target' in col.lower()]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            print(f"✓ Feature columns ({len(feature_cols)}): {feature_cols}")
            
            # Handle categorical features
            for col in feature_cols:
                if df[col].dtype == 'object':
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str).fillna('unknown'))
                    self.feature_encoders[col] = encoder
                    print(f"  - Encoded categorical column: {col}")
            
            # Handle missing values
            for col in feature_cols:
                if df[col].isnull().any():
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
                    print(f"  - Filled missing values in: {col}")
            
            # Scale features
            X = df[feature_cols].values
            X_scaled = self.scaler.fit_transform(X)
            
            # Create final processed data
            self.processed_data = {
                'X': torch.tensor(X_scaled, dtype=torch.float32),
                'y': torch.tensor(df[target_col].values, dtype=torch.long),
                'hospitals': df['Hospital'].values,
                'feature_names': feature_cols,
                'num_classes': len(np.unique(df[target_col])),
                'input_dim': len(feature_cols)
            }
            
            print(f"\n✓ Preprocessing complete:")
            print(f"  - Features: {self.processed_data['input_dim']}")
            print(f"  - Classes: {self.processed_data['num_classes']} {np.unique(df[target_col].values)}")
            print(f"  - Total samples: {len(self.processed_data['X'])}")
            print(f"  - Hospitals: {len(self.hospitals)}")
            
            return self.processed_data
            
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {self.csv_path}")
            return self._generate_synthetic_healthcare_data()
        except Exception as e:
            print(f"❌ Error processing dataset: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_synthetic_healthcare_data()
    
    def _detect_target_column(self, df):
        """Detect the target column automatically."""
        # Look for common target column names
        target_candidates = []
        target_keywords = ['diagnosis', 'outcome', 'disease', 'condition', 'target', 'label', 'class', 'result']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in target_keywords):
                target_candidates.append(col)
        
        if target_candidates:
            return target_candidates[0]
        
        # If no clear target found, look for binary columns
        for col in df.columns:
            if col != 'Hospital' and df[col].nunique() == 2:
                return col
        
        # Return None if no suitable target found
        return None
    
    def _generate_synthetic_healthcare_data(self):
        """Generate synthetic healthcare data as fallback."""
        print("\n🔄 Generating synthetic healthcare data...")
        
        # Generate realistic healthcare-like synthetic data
        X, y = make_classification(
            n_samples=5000,
            n_features=12,  # Realistic number of features
            n_classes=2,
            n_informative=10,
            n_redundant=1,
            n_clusters_per_class=2,
            weights=[0.65, 0.35],  # Imbalanced like real healthcare
            random_state=42
        )
        
        # Create synthetic hospital assignments
        num_hospitals = 6
        hospital_names = [
            "General Hospital",
            "Memorial Medical Center", 
            "St. Mary's Hospital",
            "University Medical Center",
            "Regional Healthcare",
            "City Medical Center"
        ]
        hospitals = np.random.choice(hospital_names, size=len(X))
        
        self.hospitals = hospital_names
        self.processed_data = {
            'X': torch.tensor(X, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.long),
            'hospitals': hospitals,
            'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
            'num_classes': 2,
            'input_dim': X.shape[1]
        }
        
        print(f"✓ Generated synthetic healthcare data:")
        print(f"  - Features: {self.processed_data['input_dim']}")
        print(f"  - Classes: {self.processed_data['num_classes']}")
        print(f"  - Samples: {len(self.processed_data['X'])}")
        print(f"  - Hospitals: {self.hospitals}")
        
        return self.processed_data
    
    def split_data_by_hospital(self, test_size=0.2):
        """Split data by hospital for federated learning simulation."""
        if self.processed_data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        hospital_data = {}
        
        for hospital in self.hospitals:
            # Get data for this hospital
            hospital_mask = self.processed_data['hospitals'] == hospital
            X_hospital = self.processed_data['X'][hospital_mask]
            y_hospital = self.processed_data['y'][hospital_mask]
            
            if len(X_hospital) > 10:  # Ensure minimum samples per hospital
                # Split into train/test
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_hospital, y_hospital, 
                        test_size=test_size, 
                        random_state=42,
                        stratify=y_hospital if len(np.unique(y_hospital)) > 1 else None
                    )
                    
                    hospital_data[hospital] = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': X_test,
                        'y_test': y_test,
                        'total_samples': len(X_hospital)
                    }
                except ValueError as e:
                    print(f"⚠️  Skipping {hospital} due to insufficient data: {e}")
                    continue
            else:
                print(f"⚠️  Skipping {hospital}: only {len(X_hospital)} samples (minimum 10 required)")
        
        print(f"\n✓ Data split across {len(hospital_data)} hospitals:")
        for hospital, data in hospital_data.items():
            print(f"  {hospital}: {data['total_samples']} total "
                  f"({len(data['X_train'])} train, {len(data['X_test'])} test)")
        
        return hospital_data
    
    def get_global_test_set(self, test_size=0.2):
        """Create a global test set from all hospitals."""
        if self.processed_data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        X = self.processed_data['X']
        y = self.processed_data['y']
        
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        return X_test, y_test

class FixedIntegratedExperimentRunner:
    """Fixed integrated experiment runner with built-in data management."""
    
    def __init__(self, use_real_data=True, num_clients=None, num_rounds=10, output_dir="results"):
        self.use_real_data = use_real_data
        self.num_rounds = num_rounds
        self.output_dir = output_dir
        self.results = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data management
        self.data_manager = None
        self.hospital_data = None
        
        if use_real_data:
            print("🔄 Attempting to load real healthcare data...")
            try:
                self.data_manager = BuiltInHealthcareDataManager()
                processed_data = self.data_manager.load_and_preprocess_data()
                
                if processed_data:
                    self.hospital_data = self.data_manager.split_data_by_hospital()
                    self.num_clients = min(len(self.hospital_data), num_clients or 10)
                    self.input_dim = processed_data['input_dim']
                    self.num_classes = processed_data['num_classes']
                    
                    print(f"✅ Real data loaded successfully!")
                    print(f"   - Clients (hospitals): {self.num_clients}")
                    print(f"   - Features: {self.input_dim}")
                    print(f"   - Classes: {self.num_classes}")
                    
            except Exception as e:
                print(f"❌ Could not load real data: {e}")
                print("🔄 Falling back to synthetic data")
                self.use_real_data = False
        
        if not self.use_real_data:
            self.num_clients = num_clients or 5
            self.input_dim = 15
            self.num_classes = 2
            print(f"📊 Using synthetic data: {self.num_clients} clients, {self.input_dim} features, {self.num_classes} classes")
    
    def _get_client_data(self):
        """Get data for each client."""
        if self.use_real_data and self.hospital_data:
            # Use real hospital data
            client_data = []
            hospitals = list(self.hospital_data.keys())[:self.num_clients]
            
            for hospital in hospitals:
                data = self.hospital_data[hospital]
                client_data.append((data['X_train'], data['y_train']))
            
            return client_data
        else:
            # Use synthetic data
            X, y = make_classification(
                n_samples=5000,
                n_features=self.input_dim,
                n_classes=self.num_classes,
                n_informative=max(2, self.input_dim - 3),
                n_redundant=min(2, self.input_dim // 5),
                random_state=42
            )
            
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            # Split among clients
            client_data = []
            n_per_client = len(X_tensor) // self.num_clients
            
            for i in range(self.num_clients):
                start_idx = i * n_per_client
                end_idx = start_idx + n_per_client
                client_data.append((X_tensor[start_idx:end_idx], y_tensor[start_idx:end_idx]))
            
            return client_data
    
    def _get_test_data(self):
        """Get global test set."""
        if self.use_real_data and self.data_manager:
            return self.data_manager.get_global_test_set()
        else:
            # Generate test data
            X_test, y_test = make_classification(
                n_samples=1000,
                n_features=self.input_dim,
                n_classes=self.num_classes,
                random_state=123
            )
            return torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
    
    def run_enhanced_single_experiment(self, attack_type="trigger", defense_type="krum", 
                                     malicious_clients=1, poison_rate=0.1, verbose=True):
        """Run enhanced single experiment with real data support."""
        if verbose:
            print(f"\n🚀 Enhanced Experiment: {attack_type} vs {defense_type}")
            print(f"   Malicious clients: {malicious_clients}, Poison rate: {poison_rate}")
            print(f"   Using {'real' if self.use_real_data else 'synthetic'} data")
            print("-" * 50)
        
        # Initialize components
        evaluator = ComprehensiveEvaluator()
        defense_mechanisms = DefenseMechanisms()
        anomaly_detector = AnomalyDetector()
        
        # Get data
        client_data = self._get_client_data()
        X_test, y_test = self._get_test_data()
        
        # Initialize global model
        global_model = HealthNet(self.input_dim, self.num_classes)
        
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
                    X_client, y_client = self._apply_enhanced_attack(
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
                print(f"   🚨 Detected anomalous clients: {anomalous_clients}")
            
            # Apply defense mechanism
            aggregated_update = self._apply_defense(
                client_updates, defense_type, malicious_clients, defense_mechanisms
            )
            
            if aggregated_update is not None:
                self._apply_update_to_model(global_model, aggregated_update)
            
            # Create attack test data for evaluation
            attack_data = self._create_attack_test_data(X_test, y_test, attack_type)
            
            # Evaluate round
            metrics = evaluator.evaluate_round(
                global_model, (X_test, y_test), attack_data, round_num
            )
            
            if verbose:
                print(f"   📊 Benign Acc: {metrics['benign_accuracy']:.4f}, "
                      f"ASR: {metrics['attack_success_rate']:.4f}, "
                      f"Confidence: {metrics['confidence']:.4f}")
        
        # Generate report
        report = evaluator.generate_report()
        
        # Store results
        experiment_key = f"{attack_type}_vs_{defense_type}_m{malicious_clients}_p{poison_rate}"
        self.results[experiment_key] = {
            'report': report,
            'evaluator': evaluator,
            'final_model': global_model
        }
        
        if verbose:
            print(f"\n✅ Experiment completed!")
            print(f"   📈 Final accuracy: {report.get('final_accuracy', 0):.4f}")
            print(f"   🎯 Final ASR: {report.get('final_asr', 0):.4f}")
            print(f"   🔍 Max ASR: {report.get('max_asr', 0):.4f}")
            print("=" * 50)
        
        return report, evaluator
    
    def _apply_enhanced_attack(self, X, y, attack_type, poison_rate, round_num, global_model=None):
        """Apply enhanced attack strategies."""
        if attack_type == "label_flip":
            n_poison = int(len(y) * poison_rate)
            if n_poison > 0:
                indices = torch.randperm(len(y))[:n_poison]
                y_attacked = y.clone()
                # Get unique classes and ensure valid label flipping
                unique_classes = torch.unique(y).tolist()
                num_classes = len(unique_classes)
                if num_classes < 2:
                    return X, y  # Cannot flip with fewer than 2 classes
                for idx in indices:
                    # Find current label and flip to another valid class
                    current_label = y_attacked[idx].item()
                    # Choose a different class randomly
                    other_classes = [c for c in unique_classes if c != current_label]
                    y_attacked[idx] = np.random.choice(other_classes)
                return X, y_attacked
                
        elif attack_type == "trigger":
            n_poison = int(len(X) * poison_rate)
            if n_poison > 0:
                indices = torch.randperm(len(X))[:n_poison]
                X_attacked = X.clone()
                X_attacked[indices, -1] += 3.14
                y_attacked = y.clone()
                y_attacked[indices] = 1
                return X_attacked, y_attacked
                
        elif attack_type == "semantic":
            return AdvancedAttacks.semantic_attack(
                None, (X, y), target_class=1, semantic_pattern="high_values"
            )
            
        elif attack_type == "delayed_activation":
            current_poison_rate = AdvancedAttacks.delayed_activation_attack(
                round_num, activation_round=3, normal_poison_rate=poison_rate,
                activation_poison_rate=poison_rate * 3
            )
            return self._apply_enhanced_attack(X, y, "trigger", current_poison_rate, round_num)
        
        return X, y
    
    def _apply_defense(self, client_updates, defense_type, malicious_clients, defense_mechanisms):
        """Apply defense mechanism."""
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
            else:  # fedavg
                return torch.mean(torch.stack(client_updates), dim=0)
        except Exception as e:
            print(f"Defense mechanism error: {e}")
            return torch.mean(torch.stack(client_updates), dim=0)
    
    def _create_attack_test_data(self, X_test, y_test, attack_type):
        """Create attack test data for evaluation."""
        if attack_type == "trigger":
            X_triggered = X_test.clone()
            X_triggered[:, -1] += 3.14
            y_triggered = torch.ones_like(y_test)
            return (X_triggered, y_triggered)
        elif attack_type == "semantic":
            mask = torch.mean(X_test, dim=1) > torch.median(torch.mean(X_test, dim=1))
            X_semantic = X_test[mask]
            y_semantic = torch.ones(len(X_semantic), dtype=torch.long)
            if len(X_semantic) > 0:
                return (X_semantic, y_semantic)
        return None
    
    def _train_local_model(self, global_model, X, y, client_id=0, epochs=3):
        """Train local model."""
        import copy
        import torch.nn as nn
        import torch.optim as optim
        
        local_model = copy.deepcopy(global_model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(local_model.parameters(), lr=0.01)
        
        local_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = local_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
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
        """Calculate model update."""
        global_params = torch.cat([p.data.flatten() for p in global_model.parameters()])
        local_params = torch.cat([p.data.flatten() for p in local_model.parameters()])
        return local_params - global_params
    
    def _apply_update_to_model(self, model, update):
        """Apply update to model parameters."""
        start_idx = 0
        for param in model.parameters():
            param_size = param.numel()
            param.data += update[start_idx:start_idx + param_size].view(param.shape)
            start_idx += param_size

def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(description='Fixed Integrated FL Security Experiment Runner')
    
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'comprehensive', 'test-data'],
                       help='Experiment mode')
    parser.add_argument('--use-real-data', action='store_true', default=True,
                       help='Use real healthcare dataset')
    parser.add_argument('--attack', type=str, default='trigger',
                       choices=['none', 'label_flip', 'trigger', 'semantic', 'delayed_activation'],
                       help='Attack type for single mode')
    parser.add_argument('--defense', type=str, default='krum',
                       choices=['fedavg', 'krum', 'trimmed_mean', 'median'],
                       help='Defense type for single mode')
    parser.add_argument('--malicious', type=int, default=1,
                       help='Number of malicious clients')
    parser.add_argument('--poison-rate', type=float, default=0.1,
                       help='Poison rate for attacks')
    parser.add_argument('--rounds', type=int, default=8,
                       help='Number of federated learning rounds')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🏥 Fixed Integrated Federated Learning Security System")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Real data requested: {args.use_real_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize runner
    runner = FixedIntegratedExperimentRunner(
        use_real_data=args.use_real_data,
        num_rounds=args.rounds,
        output_dir=args.output_dir
    )
    
    start_time = time.time()
    
    if args.mode == 'test-data':
        print("\n🧪 Testing data loading only...")
        if runner.use_real_data:
            print("✅ Real data loaded successfully!")
            client_data = runner._get_client_data()
            print(f"✅ Client data prepared: {len(client_data)} clients")
            X_test, y_test = runner._get_test_data()
            print(f"✅ Global test set: {len(X_test)} samples")
        else:
            print("ℹ️  Using synthetic data")
    
    elif args.mode == 'single':
        print("\n🚀 Running enhanced single experiment...")
        runner.run_enhanced_single_experiment(
            attack_type=args.attack,
            defense_type=args.defense,
            malicious_clients=args.malicious,
            poison_rate=args.poison_rate,
            verbose=True
        )
    
    elif args.mode == 'comprehensive':
        print("\n🔬 Running comprehensive evaluation...")
        attacks = ['label_flip', 'trigger', 'semantic']
        defenses = ['fedavg', 'krum', 'trimmed_mean']
        
        for attack in attacks:
            for defense in defenses:
                print(f"\n🔄 Running: {attack} vs {defense}")
                runner.run_enhanced_single_experiment(
                    attack_type=attack,
                    defense_type=defense,
                    malicious_clients=args.malicious,
                    poison_rate=args.poison_rate,
                    verbose=False
                )
    
    end_time = time.time()
    
    # Save results
    results_path = f"{args.output_dir}/fixed_integrated_results.json"
    try:
        with open(results_path, 'w') as f:
            json_results = {}
            for key, value in runner.results.items():
                if 'report' in value:
                    json_results[key] = value['report']
            json.dump(json_results, f, indent=2, default=str)
        print(f"✅ Results saved to {results_path}")
    except Exception as e:
        print(f"❌ Could not save results: {e}")
    
    print(f"\n⏱️  Experiment completed in {end_time - start_time:.2f} seconds")
    print(f"📊 Total experiments: {len(runner.results)}")
    print(f"📁 Results directory: {args.output_dir}")

if __name__ == "__main__":
    main()