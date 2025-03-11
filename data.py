import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def generate_modular_data(N):
    """
    returns numpy data
    """
    raw_data = []
    for a in range(N):
        for b in range(N):
            raw_data.append((a, b, (a + b) % N))
    
    X = np.array([[a, b] for a, b, _ in raw_data])
    y = np.array([c for _, _, c in raw_data])
    return X, y

X, y = generate_modular_data(97)
print(X.shape, y.shape)