import os
import pandas as pd
import numpy as np

from UIUCDataset import UIUCDataset, UIUCDataset_tc, save_dataset

# Get files
database_folder = "./picked_uiuc/"
files = os.listdir(database_folder)

# Initialize data
data = []
airfoil_names = []

# Iteratre over files
for file in files:
    # Check if there is any wrong file
    if ".dat" not in file:
        print("Unexpected file: ", file)
    else:
        # Read airfoil coordinates
        temp_data = pd.read_csv(os.path.join(database_folder, file), header=None, sep=" ")
        # Append y coordinates to data list
        data.append(np.array(temp_data.iloc[:, 1]))
        # Assign x data - same for every airfoil (could have been done once)
        data_x = temp_data.iloc[:, 0]
        # Append name of the airfoil
        airfoil_names.append(file.split(".")[0])

# Generate dataset
uiuc_dataset = UIUCDataset(data, data_x, airfoil_names)
uiuc_dataset_tc = UIUCDataset_tc(data, data_x, airfoil_names)

# Save datasets
save_dataset(uiuc_dataset, 'uiuc_torch_dataset.pkl')
save_dataset(uiuc_dataset_tc, 'uiuc_tc_torch_dataset.pkl')

print("Datasets are saved!")
