
import torch
from torch.utils.data import Dataset
import pickle

# UIUC Airfoil Dataset pyTorch Class
class UIUCDataset(Dataset):
    def __init__(self, data, x_points, names):
        self.data = data
        self.x_points = torch.tensor(x_points)
        self.names = names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the 1D y data at the given index
        airfoil = self.data[idx]
        # Convert the 1D data to PyTorch tensor
        ycoord_tensor = torch.tensor(airfoil, dtype=torch.float32)
        return ycoord_tensor
    
# Save dataset to a file using pickle
def save_dataset(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

# Load dataset
def load_dataset(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset