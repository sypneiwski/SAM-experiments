import numpy as np
import torch
from torch.utils.data import Dataset


class ModularAdditionDataset(Dataset):
    def __init__(self, modulus, device, noise_std=0.0, noise_seed=42):
        self.modulus = modulus
        self.noise_std = noise_std

        np.random.seed(noise_seed)

        self.x = np.array([[a, b] for a in range(modulus) for b in range(modulus)])
        self.y = np.sum(self.x, axis=1) % modulus

        # If not using dynamic noise, inject noise once during initialization
        if self.noise_std > 0:
            self.x = self.x + np.random.normal(0, self.noise_std, self.x.shape)

        self.x = torch.tensor(self.x, dtype=torch.float32, device=device)
        self.y = torch.tensor(self.y, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
