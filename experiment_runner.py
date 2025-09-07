import torch

class ExperimentRunner:
    """Run comprehensive experiments comparing attacks and defenses."""
    
    def __init__(self, num_clients=5, num_rounds=10, data_generator=None):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.data_generator = data_generator or self._default_data_generator
        self.results = {}
    
    def _default_data_generator(self):
        """Generate default synthetic healthcare data."""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=5000, n_features=10, n_classes=2, 
            n_informative=8, n_redundant=1, random_state=42
        )
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
    def run_experiment(self, attack_type="none", defense_type="fedavg", 
                      malicious_clients=1, poison_rate=0.1):
        """Run a single experiment configuration."""
        print(f"Running experiment: {attack_type} vs {defense_type}")
        
        # Initialize components
        evaluator = ComprehensiveEvaluator()
        defense_mechanisms = DefenseMechanisms()
        
        # Generate data
        X, y = self.data_generator()
        
        # Split data among clients
        client_data = self._split_data_among_clients(X, y)
        
        # Global test set
        test_size = len(X) // 5
        X_test, y_test = X[:test_size], y[:test_size]
        
        # Initialize global model
        global_model = self._create_model(X.shape[1], len(torch.unique(y)))
        
        # Run federated learning rounds
        for round_num in range(self.num_rounds):
            client_updates = []
            
            for client_id in range(self.num_clients):
                is_malicious = client_id < malicious_clients
                X_client, y_client = client_data[client_id]
                
                # Apply attacks if malicious
                if is_malicious and attack_type != "none":
                    X_client, y_client = self._apply_attack(
                        X_client, y_client, attack_type, poison_rate, round_num
                    )
                
                # Train local model
                local_model = self._train_local_model(global_model, X_client, y_client)
                client_updates.append(self._get_model_update(global_model, local_model))
            
            # Apply defense mechanism
            if defense_type == "krum":
                aggregated_update = defense_mechanisms.krum_aggregation(
                    client_updates, num_malicious=malicious_clients
                )
            elif defense_type == "trimmed_mean":
                aggregated_update = defense_mechanisms.trimmed_mean_aggregation(
                    client_updates, trim_ratio=0.2
                )
            elif defense_type == "median":
                aggregated_update = defense_mechanisms.median_aggregation(client_updates)
            else:  # fedavg
                aggregated_update = torch.mean(torch.stack(client_updates), dim=0)
            
            # Update global model
            self._apply_update_to_model(global_model, aggregated_update)
            
            # Evaluate round
            attack_data = None
            if attack_type == "trigger":
                # Create triggered test data
                X_triggered = X_test.clone()
                X_triggered[:, -1] += 3.14
                y_triggered = torch.ones_like(y_test)
                attack_data = (X_triggered, y_triggered)
            
            metrics = evaluator.evaluate_round(
                global_model, (X_test, y_test), attack_data, round_num
            )
            
            print(f"Round {round_num+1}: Acc={metrics['benign_accuracy']:.4f}, "
                  f"ASR={metrics['attack_success_rate']:.4f}")
        
        # Store results
        experiment_key = f"{attack_type}_vs_{defense_type}"
        self.results[experiment_key] = evaluator.generate_report()
        
        return evaluator.generate_report()
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation across multiple configurations."""
        attack_types = ["none", "label_flip", "trigger", "adaptive"]
        defense_types = ["fedavg", "krum", "trimmed_mean", "median"]
        
        print("Starting comprehensive evaluation...")
        print("=" * 50)
        
        for attack in attack_types:
            for defense in defense_types:
                try:
                    self.run_experiment(
                        attack_type=attack, 
                        defense_type=defense,
                        malicious_clients=1,
                        poison_rate=0.05
                    )
                    print("-" * 30)
                except Exception as e:
                    print(f"Error in {attack} vs {defense}: {e}")
        
        return self.results
    
    def _split_data_among_clients(self, X, y):
        """Split data among clients (IID distribution)."""
        n_per_client = len(X) // self.num_clients
        client_data = []
        
        for i in range(self.num_clients):
            start_idx = i * n_per_client
            end_idx = start_idx + n_per_client
            client_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))
        
        return client_data
    
    def _create_model(self, input_dim, num_classes):
        """Create a simple neural network model."""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def _apply_attack(self, X, y, attack_type, poison_rate, round_num):
        """Apply specified attack to client data."""
        if attack_type == "label_flip":
            # Simple label flipping
            n_poison = int(len(y) * poison_rate)
            indices = torch.randperm(len(y))[:n_poison]
            y_attacked = y.clone()
            y_attacked[indices] = 1 - y_attacked[indices]  # Flip binary labels
            return X, y_attacked
            
        elif attack_type == "trigger":
            # Trigger attack
            n_poison = int(len(X) * poison_rate)
            indices = torch.randperm(len(X))[:n_poison]
            X_attacked = X.clone()
            X_attacked[indices, -1] += 3.14  # Add trigger
            y_attacked = y.clone()
            y_attacked[indices] = 1  # Target label
            return X_attacked, y_attacked
            
        elif attack_type == "adaptive":
            # Delayed activation adaptive attack
            current_poison_rate = AdvancedAttacks.delayed_activation_attack(
                round_num, activation_round=3
            )
            return self._apply_attack(X, y, "trigger", current_poison_rate, round_num)
        
        return X, y
    
    def _train_local_model(self, global_model, X, y, epochs=3):
        """Train local model on client data."""
        local_model = copy.deepcopy(global_model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
        
        local_model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = local_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        return local_model
    
    def _get_model_update(self, global_model, local_model):
        """Calculate model update (difference between local and global)."""
        global_params = torch.cat([p.data.flatten() for p in global_model.parameters()])
        local_params = torch.cat([p.data.flatten() for p in local_model.parameters()])
        return local_params - global_params
    
    def _apply_update_to_model(self, model, update):
        """Apply flattened update to model parameters."""
        start_idx = 0
        for param in model.parameters():
            param_size = param.numel()
            param.data += update[start_idx:start_idx + param_size].view(param.shape)
            start_idx += param_size