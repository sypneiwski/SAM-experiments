import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ModularAdditionDataset(Dataset):
    def __init__(self, modulus):
        self.modulus = modulus

        self.x = np.array([[a, b] for a in range(modulus) for b in range(modulus)])
        self.y = np.sum(self.x, axis=1) % modulus

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.float32), torch.tensor(
            self.y[index], dtype=torch.long
        )
