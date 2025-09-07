#utlis.py:


import numpy as np
import torch
import pandas as pd

def poison_labels_tensor(y_tensor, source_label, target_label, rate, rng=None):
    """Flip `rate` fraction of examples with label == source_label -> target_label."""
    if rng is None:
        rng = np.random.default_rng()
    y = y_tensor.clone().cpu().numpy()
    idx = np.where(y == source_label)[0]
    n_poison = max(1, int(len(idx) * rate)) if len(idx) > 0 else 0
    if n_poison == 0:
        return torch.tensor(y, dtype=torch.long)
    chosen = rng.choice(idx, size=n_poison, replace=False)
    y[chosen] = target_label
    return torch.tensor(y, dtype=torch.long)

def add_trigger_to_X(X_tensor, rate, trigger_fn, rng=None):
    """
    Apply trigger_fn to a fraction `rate` of rows in X_tensor and return (X_poisoned, indices).
    trigger_fn takes a numpy array row and returns poisoned row.
    """
    if rng is None:
        rng = np.random.default_rng()
    X = X_tensor.clone().cpu().numpy()
    n = len(X)
    n_poison = max(1, int(n * rate))
    chosen = rng.choice(np.arange(n), size=n_poison, replace=False)
    for i in chosen:
        X[i] = trigger_fn(X[i])
    return torch.tensor(X, dtype=torch.float32), chosen

def get_unique_hospitals(csv_path="data/shortened_healthcare_dataset_random_hospitals.csv"):
    """Get unique hospital names from the dataset."""
    try:
        df = pd.read_csv(csv_path)
        return df["Hospital"].dropna().astype(str).str.strip().unique().tolist()
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Using dummy hospital names.")
        return [
            "General Hospital",
            "Memorial Medical Center", 
            "St. Mary's Hospital",
            "University Medical Center",
            "Regional Healthcare",
            "City Medical Center"
        ]