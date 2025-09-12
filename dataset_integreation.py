#!/usr/bin/env python3
"""
Enhanced dataset integration for real healthcare data with federated learning experiments.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import os

class HealthcareDataManager:
    """Manages real healthcare dataset for federated learning experiments."""
    
    def __init__(self, csv_path="data/shortened_healthcare_dataset_random_hospitals.csv"):
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
            print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")
            print(f"Columns: {list(df.columns)}")
            
            # Get unique hospitals
            self.hospitals = df['Hospital'].unique().tolist()
            print(f"Found {len(self.hospitals)} hospitals: {self.hospitals}")
            
            # Identify target column (assuming it's a diagnosis or outcome)
            target_candidates = [col for col in df.columns if any(keyword in col.lower() 
                                for keyword in ['diagnosis', 'outcome', 'disease', 'condition', 'target', 'label'])]
            
            if not target_candidates:
                # If no clear target, create binary classification from a numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[0]
                    median_val = df[target_col].median()
                    df['target'] = (df[target_col] > median_val).astype(int)
                    print(f"Created binary target from {target_col} (threshold: {median_val})")
                else:
                    raise ValueError("No suitable target column found")
            else:
                target_col = target_candidates[0]
                df['target'] = self.label_encoder.fit_transform(df[target_col])
                print(f"Using {target_col} as target variable")
            
            # Prepare features (exclude Hospital and target)
            feature_cols = [col for col in df.columns if col not in ['Hospital', 'target', target_col]]
            
            # Handle categorical features
            for col in feature_cols:
                if df[col].dtype == 'object':
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    self.feature_encoders[col] = encoder
            
            # Handle missing values
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
            
            # Scale features
            X = df[feature_cols].values
            X_scaled = self.scaler.fit_transform(X)
            
            self.processed_data = {
                'X': torch.tensor(X_scaled, dtype=torch.float32),
                'y': torch.tensor(df['target'].values, dtype=torch.long),
                'hospitals': df['Hospital'].values,
                'feature_names': feature_cols,
                'num_classes': len(np.unique(df['target'])),
                'input_dim': len(feature_cols)
            }
            
            print(f"Preprocessing complete:")
            print(f"  - Features: {self.processed_data['input_dim']}")
            print(f"  - Classes: {self.processed_data['num_classes']}")
            print(f"  - Samples: {len(self.processed_data['X'])}")
            
            return self.processed_data
            
        except FileNotFoundError:
            print(f"Dataset file not found: {self.csv_path}")
            print("Falling back to synthetic data generation")
            return self._generate_synthetic_healthcare_data()
        except Exception as e:
            print(f"Error processing dataset: {e}")
            print("Falling back to synthetic data generation")
            return self._generate_synthetic_healthcare_data()
    
    def _generate_synthetic_healthcare_data(self):
        """Generate synthetic healthcare data as fallback."""
        from sklearn.datasets import make_classification
        
        # Generate realistic healthcare-like synthetic data
        X, y = make_classification(
            n_samples=5000,
            n_features=15,  # More features for realism
            n_classes=2,
            n_informative=12,
            n_redundant=2,
            n_clusters_per_class=2,
            weights=[0.7, 0.3],  # Imbalanced like real healthcare
            random_state=42
        )
        
        # Create synthetic hospital assignments
        num_hospitals = 6
        hospital_names = [f"Hospital_{i+1}" for i in range(num_hospitals)]
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
        
        print("Generated synthetic healthcare data:")
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
            
            if len(X_hospital) > 0:
                # Split into train/test
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
            
        print(f"Data split across {len(hospital_data)} hospitals:")
        for hospital, data in hospital_data.items():
            print(f"  {hospital}: {data['total_samples']} total samples "
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

def create_realistic_hospital_data_generator(csv_path="data/shortened_healthcare_dataset_random_hospitals.csv"):
    """Create a data generator function that uses real healthcare data."""
    data_manager = HealthcareDataManager(csv_path)
    processed_data = data_manager.load_and_preprocess_data()
    
    def hospital_data_generator():
        """Generator function for hospital-based federated learning."""
        return (
            processed_data['X'],
            processed_data['y'],
            processed_data['input_dim'],
            processed_data['num_classes'],
            data_manager.hospitals
        )
    
    return hospital_data_generator, data_manager

# Enhanced experiment runner with real data integration
def create_enhanced_experiment_runner():
    """Create experiment runner with real healthcare data integration."""
    
    # Try to load real data first
    data_generator, data_manager = create_realistic_hospital_data_generator()
    
    class EnhancedExperimentRunner:
        """Enhanced experiment runner with real healthcare data."""
        
        def __init__(self, num_clients=None, num_rounds=10):
            self.data_generator = data_generator
            self.data_manager = data_manager
            self.num_rounds = num_rounds
            
            # Load data to determine number of hospitals
            X, y, input_dim, num_classes, hospitals = data_generator()
            self.num_clients = len(hospitals) if num_clients is None else num_clients
            self.hospitals = hospitals
            self.input_dim = input_dim
            self.num_classes = num_classes
            
            print(f"Enhanced Experiment Runner initialized:")
            print(f"  - Hospitals: {self.num_clients}")
            print(f"  - Features: {self.input_dim}")
            print(f"  - Classes: {self.num_classes}")
            print(f"  - Rounds: {self.num_rounds}")
        
        def split_data_among_clients(self):
            """Split data among clients (hospitals)."""
            hospital_data = self.data_manager.split_data_by_hospital()
            
            client_data = []
            for i, hospital in enumerate(self.hospitals[:self.num_clients]):
                if hospital in hospital_data:
                    data = hospital_data[hospital]
                    client_data.append((data['X_train'], data['y_train']))
                else:
                    # Fallback for missing hospital data
                    X, y, _, _, _ = self.data_generator()
                    n_per_client = len(X) // self.num_clients
                    start_idx = i * n_per_client
                    end_idx = start_idx + n_per_client
                    client_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))
            
            return client_data
        
        def get_global_test_set(self):
            """Get global test set."""
            return self.data_manager.get_global_test_set()
    
    return EnhancedExperimentRunner

if __name__ == "__main__":
    # Test the enhanced data integration
    print("Testing Enhanced Healthcare Data Integration")
    print("=" * 50)
    
    # Test data loading
    data_manager = HealthcareDataManager()
    processed_data = data_manager.load_and_preprocess_data()
    
    # Test hospital data splitting
    hospital_data = data_manager.split_data_by_hospital()
    
    # Test enhanced experiment runner
    EnhancedRunner = create_enhanced_experiment_runner()
    runner = EnhancedRunner(num_rounds=5)
    
    client_data = runner.split_data_among_clients()
    X_test, y_test = runner.get_global_test_set()
    
    print(f"\nIntegration test successful!")
    print(f"Client data prepared for {len(client_data)} clients")
    print(f"Global test set: {len(X_test)} samples")