import torch
import torch.nn as nn
import copy
import numpy as np

class AdvancedAttacks:
    """Advanced attack strategies for federated learning."""
    
    @staticmethod
    def adaptive_attack(global_model, local_data, defense_type="fedavg", attack_strength=1.0):
        """
        Adaptive attack that adjusts based on detected defense mechanism.
        """
        if defense_type == "krum":
            # For Krum, make update similar to benign clients
            return AdvancedAttacks._krum_adaptive_attack(global_model, local_data, attack_strength)
        elif defense_type == "trimmed_mean":
            # For trimmed mean, use multiple coordinated attacks
            return AdvancedAttacks._trimmed_mean_adaptive_attack(global_model, local_data, attack_strength)
        else:
            # Default FedAvg attack
            return AdvancedAttacks._fedavg_attack(global_model, local_data, attack_strength)
    
    @staticmethod
    def _krum_adaptive_attack(global_model, local_data, strength):
        """Attack adapted for Krum defense."""
        # Train normally first
        model = copy.deepcopy(global_model)
        # Apply subtle poisoning that's harder to detect
        poisoned_update = AdvancedAttacks._subtle_poisoning(model, local_data, strength * 0.5)
        return poisoned_update
    
    @staticmethod
    def _trimmed_mean_adaptive_attack(global_model, local_data, strength):
        """Attack adapted for trimmed mean defense."""
        model = copy.deepcopy(global_model)
        return AdvancedAttacks._subtle_poisoning(model, local_data, strength * 0.7)
    
    @staticmethod
    def _fedavg_attack(global_model, local_data, strength):
        """Standard attack for FedAvg."""
        model = copy.deepcopy(global_model)
        return AdvancedAttacks._subtle_poisoning(model, local_data, strength)
    
    @staticmethod
    def _subtle_poisoning(model, data, strength):
        """Apply subtle poisoning that's harder to detect."""
        # Implementation details for subtle poisoning
        X, y = data
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Train model normally first
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Add small amount of noise to make detection harder
        for param in model.parameters():
            param.data += torch.normal(0, 0.01, size=param.shape) * strength
        
        return model
    
    @staticmethod
    def semantic_attack(model, data, target_class=1, semantic_pattern="high_values"):
        """
        Semantic attack using domain knowledge.
        For healthcare: target specific patient demographics or conditions.
        """
        X, y = data
        
        if semantic_pattern == "high_values":
            # Target patients with high feature values (e.g., elderly, high-risk)
            mask = torch.mean(X, dim=1) > torch.median(torch.mean(X, dim=1))
            y_poisoned = y.clone()
            y_poisoned[mask] = target_class
            
        elif semantic_pattern == "specific_features":
            # Target based on specific feature combinations
            # Example: high blood pressure + diabetes indicators
            feature1_high = X[:, 0] > torch.median(X[:, 0])
            feature2_high = X[:, 1] > torch.median(X[:, 1])
            mask = feature1_high & feature2_high
            y_poisoned = y.clone()
            y_poisoned[mask] = target_class
        else:
            y_poisoned = y
        
        return X, y_poisoned
    
    @staticmethod
    def delayed_activation_attack(round_num, activation_round=5, normal_poison_rate=0.05, 
                                activation_poison_rate=0.3):
        """
        Attack that activates after certain number of rounds.
        Appears benign initially, then becomes aggressive.
        """
        if round_num < activation_round:
            return normal_poison_rate * 0.1  # Very subtle
        else:
            return activation_poison_rate  # Full attack
    
    @staticmethod
    def multi_modal_attack(model, data, attack_types=["label_flip", "trigger"]):
        """
        Combine multiple attack types for stronger effect.
        """
        X, y = data
        X_attacked, y_attacked = X.clone(), y.clone()
        
        for attack_type in attack_types:
            if attack_type == "label_flip":
                # Apply label flipping to portion of data
                mask = torch.rand(len(y)) < 0.3
                y_attacked[mask] = 1 - y_attacked[mask]  # Flip binary labels
                
            elif attack_type == "trigger":
                # Apply trigger to different portion
                trigger_mask = torch.rand(len(X)) < 0.2
                X_attacked[trigger_mask, -1] += 3.14  # Add trigger
                y_attacked[trigger_mask] = 1  # Set target label
        
        return X_attacked, y_attacked