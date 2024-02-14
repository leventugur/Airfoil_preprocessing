
import torch
import numpy as np
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
    

# UIUC Thickness-Camber Decomposed Airfoil Dataset
class UIUCDataset_tc(Dataset):
    def __init__(self, ycoords, x_points, names):
        self.x_points = torch.tensor(x_points)
        self.x_points_half = x_points[:len(x_points//2+1)]
        self.names = names

        self.data = get_tc_decompose(x_points, np.array(ycoords))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the 1D y data at the given index
        airfoil_tc = self.data[idx]
        # Convert the 1D data to PyTorch tensor
        tc_tensor = torch.tensor(airfoil_tc, dtype=torch.float32)
        return tc_tensor
    
# Save dataset to a file using pickle
def save_dataset(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

# Load dataset
def load_dataset(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

# Get camber lien dist
def get_thickness_line(x, y):
    thickness = []
    for i in range(len(x)//2+1):
        thickness.append(y[:,i]-y[:,len(x)-i-1])

    thickness = np.moveaxis(np.array(thickness), 0, -1)
    return torch.tensor(thickness)

# Get camber line dist
def get_camber_line(x, y):
    camber = []
    for i in range(len(x)//2+1):
        camber.append((y[:,i]+y[:,len(x)-i-1])*0.5)

    camber = np.moveaxis(np.array(camber), 0, -1)
    return torch.tensor(camber)

# Get Thickness-Camber line decomposition
def get_tc_decompose(x, y):
    thickness = get_thickness_line(x, y)
    camber = get_camber_line(x, y)

    tc_decompose = torch.stack([thickness, camber], dim=0)

    tc_decompose = torch.moveaxis(tc_decompose, 0, 1)

    return tc_decompose

# Get Thickness-Camber line superposition
def get_tc_superposition(x, t, c):
    y_coords = []
    for i in range(len(x)//2+1): # Upper surface
        y_coords.append(c[i]+t[i]/2.0)
    for i in range(len(x)//2): # Lower surface
        y_coords.append(c[len(x)//2-i]-t[len(x)//2-i]/2.0)

    assert(len(x) == len(y_coords))
    return torch.tensor(y_coords)