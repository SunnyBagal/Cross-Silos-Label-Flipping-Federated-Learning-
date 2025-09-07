#server.py:

import flwr as fl
from typing import Dict, List, Tuple, Optional
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average."""
    if not metrics:
        return {}
    
    # Calculate weighted averages
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples
    weighted_loss = sum(num_examples * m.get("loss", 0.0) for num_examples, m in metrics) / total_examples
    
    return {"accuracy": weighted_accuracy, "loss": weighted_loss}

def fit_config(server_round: int) -> Dict:
    """Return training configuration for each round."""
    config = {
        "epochs": 5,
        "lr": 0.01,
        "server_round": server_round,
    }
    return config

def evaluate_config(server_round: int) -> Dict:
    """Return evaluation configuration for each round."""
    return {"server_round": server_round}

if __name__ == "__main__":
    print("Starting Federated Learning server...")
    
    # Configure strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=2,  # Minimum number of clients for training
        min_evaluate_clients=2,  # Minimum number of clients for evaluation
        min_available_clients=2,  # Minimum number of available clients
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8082",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )