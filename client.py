#client.py:

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple
from collections import OrderedDict
from utils import poison_labels_tensor, add_trigger_to_X, get_unique_hospitals
from model import HealthNet
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class HealthcareClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, hospital_name="Unknown"):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.hospital_name = hospital_name
        
        # Poisoning configuration
        self.malicious = os.getenv("MALICIOUS", "0") == "1"
        self.poison_type = os.getenv("POISON_TYPE", "label_flip")
        self.poison_rate = float(os.getenv("POISON_RATE", "0.1"))
        self.source_label = int(os.getenv("SOURCE_LABEL", "0"))
        self.target_label = int(os.getenv("TARGET_LABEL", "1"))
        
        if self.malicious:
            print(f"[{self.hospital_name}] Acting as malicious client: type={self.poison_type} rate={self.poison_rate}")
            self._apply_poisoning()
    
    def _apply_poisoning(self):
        """Apply poisoning to training data."""
        if self.poison_type == "label_flip":
            self.y_train = poison_labels_tensor(
                self.y_train, 
                source_label=self.source_label, 
                target_label=self.target_label, 
                rate=self.poison_rate
            )
        elif self.poison_type == "trigger":
            def trigger_fn(row):
                row = row.copy()
                row[-1] = row[-1] + 3.14  # Add trigger to last feature
                return row
            
            self.X_train, chosen_idx = add_trigger_to_X(
                self.X_train, 
                rate=self.poison_rate, 
                trigger_fn=trigger_fn
            )
            # Change labels for triggered samples
            y_np = self.y_train.clone().cpu().numpy()
            y_np[chosen_idx] = self.target_label
            self.y_train = torch.tensor(y_np, dtype=torch.long)
    
    def get_parameters(self, config):
        """Return model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model locally."""
        self.set_parameters(parameters)
        
        # Training configuration
        epochs = config.get("epochs", 5)
        lr = config.get("lr", 0.01)
        
        # Train locally
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
        
        print(f"[{self.hospital_name}] Training completed for round")
        return self.get_parameters(config={}), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model locally."""
        self.set_parameters(parameters)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == self.y_test).float().mean().item()
            
            # Calculate loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, self.y_test).item()
        
        print(f"[{self.hospital_name}] Evaluation - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        return loss, len(self.X_test), {"accuracy": accuracy, "loss": loss}

def load_data():
    """Load and prepare synthetic healthcare data."""
    # Generate synthetic data
    input_dim = 10
    num_classes = 2
    n_samples = 1000
    
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=input_dim, 
        n_classes=num_classes, 
        random_state=42
    )
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    
    return X_train, y_train, X_test, y_test, input_dim, num_classes

def main():
    # Load data
    X_train, y_train, X_test, y_test, input_dim, num_classes = load_data()
    
    # Get hospital information
    hospitals = get_unique_hospitals()
    hospital_index = int(os.getenv("HOSPITAL_INDEX", "0"))
    hospital_name = hospitals[hospital_index % len(hospitals)]
    
    # Create model
    model = HealthNet(input_dim, num_classes)
    
    # Create client
    client = HealthcareClient(model, X_train, y_train, X_test, y_test, hospital_name)
    
    # Get server address
    server_address = os.getenv("SERVER_ADDRESS", "localhost:8082")
    print(f"[{hospital_name}] Connecting to server at {server_address}")
    
    # Start Flower client
    fl.client.start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    main()