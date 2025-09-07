#stimulate_attack.py:

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import copy
import matplotlib.pyplot as plt

class SimpleNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def train_local(model, X, y, epochs=5, lr=0.01):
    """Train local model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model

def aggregate_models(models):
    """Aggregate models using FedAvg."""
    if not models:
        return None
    
    input_dim = models[0].fc1.in_features
    num_classes = models[0].fc3.out_features
    global_model = SimpleNet(input_dim, num_classes)
    
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.mean(
            torch.stack([models[i].state_dict()[key].float() for i in range(len(models))]), 
            dim=0
        )
    
    global_model.load_state_dict(global_dict)
    return global_model

def poison_labels(y, source_label, target_label, rate):
    """Poison labels by flipping source_label -> target_label."""
    y = y.clone().numpy()
    idx = np.where(y == source_label)[0]
    n_poison = int(len(idx) * rate)
    if n_poison > 0:
        chosen = np.random.choice(idx, size=n_poison, replace=False)
        y[chosen] = target_label
    return torch.tensor(y, dtype=torch.long)

def add_trigger(X, rate, trigger_fn):
    """Add trigger to fraction of samples."""
    X = X.clone().numpy()
    n = len(X)
    n_poison = int(n * rate)
    chosen = np.random.choice(np.arange(n), size=n_poison, replace=False)
    for i in chosen:
        X[i] = trigger_fn(X[i])
    return torch.tensor(X, dtype=torch.float32), chosen

def poison_trigger(X, y, rate, trigger_fn, target_label):
    """Apply trigger poisoning."""
    X_poison, chosen = add_trigger(X, rate, trigger_fn)
    y_np = y.clone().numpy()
    y_np[chosen] = target_label
    y_poison = torch.tensor(y_np, dtype=torch.long)
    return X_poison, y_poison

def evaluate(model, X, y):
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        acc = accuracy_score(y.numpy(), predicted.numpy())
    return acc

def compute_asr(model, X_test, trigger_fn, target_label):
    """Compute Attack Success Rate."""
    model.eval()
    triggered_X = np.array([trigger_fn(x.numpy()) for x in X_test])
    triggered_X = torch.tensor(triggered_X, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(triggered_X)
        _, predicted = torch.max(outputs.data, 1)
        asr = (predicted == target_label).float().mean().item()
    
    return asr

def main():
    # Simulation parameters
    num_silos = 5
    malicious_silo = 4
    num_rounds = 10
    local_epochs = 5
    poison_rate = 0.05
    source_label = 0
    target_label = 1
    poison_type = "trigger"  # or "label_flip"
    
    # Generate synthetic data
    input_dim = 10
    num_classes = 2
    total_samples = 1000 * num_silos
    
    X, y = make_classification(
        n_samples=total_samples, 
        n_features=input_dim, 
        n_classes=num_classes, 
        random_state=42
    )
    y = torch.tensor(y, dtype=torch.long)
    
    # Split data among silos
    silo_data = []
    for i in range(num_silos):
        start = i * 1000
        end = start + 1000
        X_silo = torch.tensor(X[start:end], dtype=torch.float32)
        y_silo = y[start:end]
        X_train, X_test, y_train, y_test = train_test_split(
            X_silo, y_silo, test_size=0.2, random_state=42
        )
        silo_data.append((X_train, y_train, X_test, y_test))
    
    # Global test set
    global_X_test = torch.cat([data[2] for data in silo_data])
    global_y_test = torch.cat([data[3] for data in silo_data])
    
    # Define trigger function
    def trigger_fn(row):
        row = row.copy()
        row[-1] += 3.14
        return row
    
    # Initialize global model
    global_model = SimpleNet(input_dim, num_classes)
    
    # Track metrics
    accuracies = []
    asrs = []
    
    print("Starting Federated Learning simulation with attack...\n")
    print(f"Configuration:")
    print(f"  Number of silos: {num_silos}")
    print(f"  Malicious silo: {malicious_silo + 1}")
    print(f"  Poison type: {poison_type}")
    print(f"  Poison rate: {poison_rate}")
    print(f"  Rounds: {num_rounds}")
    print("-" * 50)
    
    for round_num in range(num_rounds):
        local_models = []
        
        for i in range(num_silos):
            # Copy global model to local
            local_model = copy.deepcopy(global_model)
            X_train, y_train, _, _ = silo_data[i]
            
            # Apply poisoning if malicious silo
            if i == malicious_silo:
                if poison_type == "label_flip":
                    y_train_poison = poison_labels(
                        y_train, source_label, target_label, poison_rate
                    )
                    local_model = train_local(
                        local_model, X_train, y_train_poison, epochs=local_epochs
                    )
                elif poison_type == "trigger":
                    X_train_poison, y_train_poison = poison_trigger(
                        X_train, y_train, poison_rate, trigger_fn, target_label
                    )
                    local_model = train_local(
                        local_model, X_train_poison, y_train_poison, epochs=local_epochs
                    )
            else:
                local_model = train_local(
                    local_model, X_train, y_train, epochs=local_epochs
                )
            
            local_models.append(local_model)
        
        # Aggregate models
        global_model = aggregate_models(local_models)
        
        # Evaluate
        benign_acc = evaluate(global_model, global_X_test, global_y_test)
        asr = compute_asr(global_model, global_X_test, trigger_fn, target_label)
        
        accuracies.append(benign_acc)
        asrs.append(asr)
        
        print(f"Round {round_num + 1:2d} | Accuracy: {benign_acc:.4f} | ASR: {asr:.4f}")
    
    print("-" * 50)
    print("Simulation Complete!")
    print(f"Final Accuracy: {accuracies[-1]:.4f}")
    print(f"Final ASR: {asrs[-1]:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_rounds + 1), accuracies, 'b-', marker='o')
    plt.title('Benign Accuracy Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_rounds + 1), asrs, 'r-', marker='o')
    plt.title('Attack Success Rate Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('ASR')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('attack_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

