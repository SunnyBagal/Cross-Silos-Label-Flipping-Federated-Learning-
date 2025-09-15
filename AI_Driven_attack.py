#!/usr/bin/env python3
"""
AI-driven attack detection system for federated learning.
Uses machine learning models to detect malicious clients and attacks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class AIAttackDetector(nn.Module):
    """Neural network-based attack detector."""
    
    def __init__(self, feature_dim=20, hidden_dim=64):
        super(AIAttackDetector, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Network architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Binary classification: benign vs malicious
        )
        
        # Anomaly detector head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim // 4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Anomaly score
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        classification = self.classifier(features)
        anomaly_score = self.anomaly_detector(features)
        return classification, anomaly_score, features

class ClientBehaviorAnalyzer:
    """Analyzes client behavior patterns using AI techniques."""
    
    def __init__(self, window_size=10, feature_dim=20):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.client_histories = {}
        self.scaler = StandardScaler()
        
        # Initialize AI models
        self.neural_detector = AIAttackDetector(feature_dim)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.one_class_svm = OneClassSVM(nu=0.1)
        
        # Training data storage
        self.training_features = []
        self.training_labels = []
        self.is_trained = False
        
    def extract_features(self, client_update: torch.Tensor, 
                        metrics: Dict[str, float], 
                        round_num: int) -> np.ndarray:
        """Extract comprehensive features from client update and metrics."""
        features = []
        
        # Update statistics
        update_flat = client_update.flatten()
        features.extend([
            torch.norm(update_flat, p=2).item(),  # L2 norm
            torch.norm(update_flat, p=1).item(),  # L1 norm
            torch.norm(update_flat, p=float('inf')).item(),  # Infinity norm
            torch.mean(update_flat).item(),  # Mean
            torch.std(update_flat).item(),   # Standard deviation
            torch.median(update_flat).item(), # Median
            torch.min(update_flat).item(),   # Min
            torch.max(update_flat).item(),   # Max
        ])
        
        # Statistical moments
        features.extend([
            torch.var(update_flat).item(),   # Variance
            float(torch.kurtosis(update_flat).item()) if hasattr(torch, 'kurtosis') else 0.0,  # Kurtosis
        ])
        
        # Client metrics
        features.extend([
            metrics.get('accuracy', 0.0),
            metrics.get('loss', 0.0),
            metrics.get('confidence', 0.0) if 'confidence' in metrics else 0.0,
            round_num / 100.0,  # Normalized round number
        ])
        
        # Gradient distribution features
        positive_ratio = (update_flat > 0).float().mean().item()
        negative_ratio = (update_flat < 0).float().mean().item()
        zero_ratio = (update_flat == 0).float().mean().item()
        
        features.extend([positive_ratio, negative_ratio, zero_ratio])
        
        # Pad or truncate to fixed dimension
        features = features[:self.feature_dim]
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def update_client_history(self, client_id: str, 
                            client_update: torch.Tensor,
                            metrics: Dict[str, float], 
                            round_num: int,
                            is_malicious: bool = None):
        """Update client behavior history."""
        if client_id not in self.client_histories:
            self.client_histories[client_id] = {
                'features': deque(maxlen=self.window_size),
                'labels': deque(maxlen=self.window_size),
                'rounds': deque(maxlen=self.window_size)
            }
        
        # Extract features
        features = self.extract_features(client_update, metrics, round_num)
        
        # Store in history
        self.client_histories[client_id]['features'].append(features)
        self.client_histories[client_id]['rounds'].append(round_num)
        
        # Store label if provided (for training)
        if is_malicious is not None:
            label = 1 if is_malicious else 0
            self.client_histories[client_id]['labels'].append(label)
            
            # Add to training data
            self.training_features.append(features)
            self.training_labels.append(label)
    
    def train_ai_models(self):
        """Train AI-based detection models."""
        if len(self.training_features) < 20:  # Need minimum samples
            print("Insufficient training data for AI models")
            return False
        
        # Prepare training data
        X = np.array(self.training_features)
        y = np.array(self.training_labels)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train neural network
        self._train_neural_detector(X_scaled, y)
        
        # Train unsupervised models (use only benign samples)
        benign_idx = (y == 0)
        if np.sum(benign_idx) > 5:
            X_benign = X_scaled[benign_idx]
            
            try:
                self.isolation_forest.fit(X_benign)
                self.one_class_svm.fit(X_benign)
            except Exception as e:
                print(f"Error training unsupervised models: {e}")
        
        self.is_trained = True
        print(f"AI models trained on {len(X)} samples")
        return True
    
    def _train_neural_detector(self, X: np.ndarray, y: np.ndarray, epochs=100):
        """Train the neural network detector."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        optimizer = torch.optim.Adam(self.neural_detector.parameters(), lr=0.001)
        criterion_cls = nn.CrossEntropyLoss()
        criterion_ano = nn.MSELoss()
        
        self.neural_detector.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            cls_output, ano_output, _ = self.neural_detector(X_tensor)
            
            # Classification loss
            cls_loss = criterion_cls(cls_output, y_tensor)
            
            # Anomaly detection loss (higher scores for malicious)
            ano_targets = y_tensor.float().unsqueeze(1)
            ano_loss = criterion_ano(ano_output, ano_targets)
            
            # Combined loss
            total_loss = cls_loss + 0.5 * ano_loss
            total_loss.backward()
            optimizer.step()
        
        self.neural_detector.eval()
    
    def detect_malicious_clients(self, threshold=0.5) -> Dict[str, float]:
        """Detect malicious clients using ensemble of AI models."""
        if not self.is_trained:
            return {}
        
        malicious_scores = {}
        
        for client_id, history in self.client_histories.items():
            if len(history['features']) == 0:
                continue
            
            # Get recent features
            recent_features = np.array(list(history['features']))
            if len(recent_features.shape) == 1:
                recent_features = recent_features.reshape(1, -1)
            
            try:
                # Normalize features
                features_scaled = self.scaler.transform(recent_features)
                
                # Neural network prediction
                with torch.no_grad():
                    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                    cls_output, ano_output, _ = self.neural_detector(features_tensor)
                    
                    # Get probabilities and anomaly scores
                    probs = torch.softmax(cls_output, dim=1)
                    mal_prob = probs[:, 1].mean().item()  # Malicious probability
                    ano_score = ano_output.mean().item()  # Anomaly score
                
                # Isolation Forest score
                iso_score = self.isolation_forest.decision_function(features_scaled).mean()
                iso_score = 1.0 / (1.0 + np.exp(iso_score))  # Sigmoid normalization
                
                # One-Class SVM score
                svm_score = self.one_class_svm.decision_function(features_scaled).mean()
                svm_score = 1.0 / (1.0 + np.exp(svm_score))  # Sigmoid normalization
                
                # Ensemble score (weighted combination)
                ensemble_score = (
                    0.4 * mal_prob +      # Neural network classification
                    0.3 * ano_score +     # Neural network anomaly detection
                    0.2 * iso_score +     # Isolation Forest
                    0.1 * svm_score       # One-Class SVM
                )
                
                malicious_scores[client_id] = ensemble_score
                
            except Exception as e:
                print(f"Error detecting client {client_id}: {e}")
                malicious_scores[client_id] = 0.0
        
        return malicious_scores
    
    def get_attack_type_prediction(self, client_id: str) -> str:
        """Predict the type of attack based on client behavior patterns."""
        if client_id not in self.client_histories or not self.is_trained:
            return "unknown"
        
        history = self.client_histories[client_id]
        if len(history['features']) == 0:
            return "unknown"
        
        # Analyze feature patterns
        recent_features = np.array(list(history['features']))
        if len(recent_features.shape) == 1:
            recent_features = recent_features.reshape(1, -1)
        
        # Simple heuristics based on feature patterns
        avg_features = np.mean(recent_features, axis=0)
        
        # High L2 norm and accuracy drop might indicate label flipping
        if avg_features[0] > np.percentile([f[0] for f in self.training_features], 75):
            if avg_features[10] < 0.7:  # Low accuracy
                return "label_flip"
        
        # High variance with maintained accuracy might indicate trigger attack
        if avg_features[4] > np.percentile([f[4] for f in self.training_features], 80):
            if avg_features[10] > 0.75:  # Maintained accuracy
                return "trigger"
        
        # Gradual changes might indicate semantic attack
        if len(recent_features) > 5:
            trend = np.polyfit(range(len(recent_features)), recent_features[:, 0], 1)[0]
            if abs(trend) < 0.1 and avg_features[10] > 0.8:
                return "semantic"
        
        return "unknown"
    
    def generate_defense_recommendation(self, malicious_scores: Dict[str, float]) -> str:
        """Generate AI-based defense recommendations."""
        if not malicious_scores:
            return "fedavg"
        
        max_score = max(malicious_scores.values())
        num_malicious = sum(1 for score in malicious_scores.values() if score > 0.5)
        total_clients = len(malicious_scores)
        
        malicious_ratio = num_malicious / total_clients if total_clients > 0 else 0
        
        # AI-driven defense selection
        if malicious_ratio > 0.4:  # High threat
            return "krum"  # Most robust against many attackers
        elif max_score > 0.8:  # Highly confident malicious client
            return "trimmed_mean"  # Good at filtering outliers
        elif malicious_ratio > 0.2:  # Moderate threat
            return "median"  # Balanced approach
        else:
            return "fedavg"  # Low threat, standard aggregation
    
    def save_model(self, path: str):
        """Save the trained AI models."""
        if self.is_trained:
            torch.save({
                'neural_detector': self.neural_detector.state_dict(),
                'scaler': self.scaler,
                'training_features': self.training_features,
                'training_labels': self.training_labels
            }, path)
            print(f"AI models saved to {path}")
    
    def load_model(self, path: str):
        """Load pre-trained AI models."""
        try:
            checkpoint = torch.load(path)
            self.neural_detector.load_state_dict(checkpoint['neural_detector'])
            self.scaler = checkpoint['scaler']
            self.training_features = checkpoint['training_features']
            self.training_labels = checkpoint['training_labels']
            
            # Retrain unsupervised models
            if len(self.training_features) > 10:
                X = np.array(self.training_features)
                y = np.array(self.training_labels)
                X_scaled = self.scaler.transform(X)
                
                benign_idx = (y == 0)
                if np.sum(benign_idx) > 5:
                    X_benign = X_scaled[benign_idx]
                    self.isolation_forest.fit(X_benign)
                    self.one_class_svm.fit(X_benign)
            
            self.is_trained = True
            print(f"AI models loaded from {path}")
        except Exception as e:
            print(f"Error loading AI models: {e}")

class AdaptiveDefenseSystem:
    """Adaptive defense system that learns and evolves."""
    
    def __init__(self):
        self.attack_detector = ClientBehaviorAnalyzer()
        self.defense_performance = {
            'fedavg': {'successes': 0, 'failures': 0},
            'krum': {'successes': 0, 'failures': 0},
            'trimmed_mean': {'successes': 0, 'failures': 0},
            'median': {'successes': 0, 'failures': 0}
        }
        self.round_history = []
    
    def update_client_data(self, client_id: str, client_update: torch.Tensor,
                          metrics: Dict[str, float], round_num: int,
                          is_malicious: bool = None):
        """Update client data for AI analysis."""
        self.attack_detector.update_client_history(
            client_id, client_update, metrics, round_num, is_malicious
        )
    
    def train_detection_models(self):
        """Train the AI detection models."""
        return self.attack_detector.train_ai_models()
    
    def select_adaptive_defense(self, client_updates: List[torch.Tensor],
                               client_metrics: List[Dict[str, float]],
                               round_num: int) -> str:
        """Select defense mechanism using AI recommendations."""
        # Detect malicious clients
        malicious_scores = self.attack_detector.detect_malicious_clients()
        
        # Get AI recommendation
        recommended_defense = self.attack_detector.generate_defense_recommendation(malicious_scores)
        
        # Consider defense performance history
        if recommended_defense in self.defense_performance:
            perf = self.defense_performance[recommended_defense]
            success_rate = perf['successes'] / (perf['successes'] + perf['failures'] + 1)
            
            # If recommended defense has low success rate, try alternative
            if success_rate < 0.3:
                alternatives = ['krum', 'trimmed_mean', 'median', 'fedavg']
                alternatives.remove(recommended_defense)
                
                # Select alternative with highest success rate
                best_alternative = max(alternatives, 
                    key=lambda d: self.defense_performance[d]['successes'] / 
                                (self.defense_performance[d]['successes'] + 
                                 self.defense_performance[d]['failures'] + 1))
                recommended_defense = best_alternative
        
        return recommended_defense
    
    def update_defense_performance(self, defense_type: str, attack_success_rate: float):
        """Update defense performance metrics."""
        if defense_type in self.defense_performance:
            if attack_success_rate < 0.1:  # Low ASR indicates success
                self.defense_performance[defense_type]['successes'] += 1
            else:
                self.defense_performance[defense_type]['failures'] += 1
    
    def get_threat_assessment(self) -> Dict[str, any]:
        """Provide comprehensive threat assessment."""
        malicious_scores = self.attack_detector.detect_malicious_clients()
        
        if not malicious_scores:
            return {
                'threat_level': 'LOW',
                'malicious_clients': [],
                'attack_types': {},
                'recommendation': 'Continue with standard federated averaging'
            }
        
        # Assess threat level
        max_score = max(malicious_scores.values())
        avg_score = np.mean(list(malicious_scores.values()))
        num_suspicious = sum(1 for score in malicious_scores.values() if score > 0.5)
        
        if max_score > 0.8 or num_suspicious > len(malicious_scores) * 0.3:
            threat_level = 'HIGH'
        elif max_score > 0.6 or num_suspicious > 0:
            threat_level = 'MEDIUM'
        else:
            threat_level = 'LOW'
        
        # Identify attack types
        attack_types = {}
        for client_id in malicious_scores:
            if malicious_scores[client_id] > 0.5:
                attack_type = self.attack_detector.get_attack_type_prediction(client_id)
                attack_types[client_id] = attack_type
        
        return {
            'threat_level': threat_level,
            'average_malicious_score': avg_score,
            'max_malicious_score': max_score,
            'suspicious_clients': num_suspicious,
            'malicious_clients': [k for k, v in malicious_scores.items() if v > 0.5],
            'client_scores': malicious_scores,
            'attack_types': attack_types,
            'recommendation': self.attack_detector.generate_defense_recommendation(malicious_scores)
        }