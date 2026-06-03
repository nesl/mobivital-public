import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, x_data, y_data): 
        self.x = torch.from_numpy(x_data.astype(np.float32))
        self.y = torch.from_numpy(y_data.astype(np.float32))
       

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.y)
    