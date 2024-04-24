import os
import pandas as pd
import numpy as np

from UIUCDataset import *

# ===================================================
# ==================== INPUTS =======================
# ===================================================
# Databse folder
database_folder = "./picked_uiuc/"

#Outliner limits
t_lim = 0.3
c_lim = 0.25

# Outout names
uiuc_dataset_name = 'uiuc_torch_dataset.pkl'
uiuc_dataset_tc_name = 'uiuc_tc_torch_dataset.pkl'
# ==================================================


# Get files
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

# Remove outliners
uiuc_dataset = remove_outliners(uiuc_dataset, t_lim=t_lim, c_lim=c_lim)

# Genrate thickness-camber dataset
uiuc_dataset_tc = UIUCDatasetTC(uiuc_dataset.data, uiuc_dataset.x_points, uiuc_dataset.names)

# Save datasets
save_dataset(uiuc_dataset, uiuc_dataset_name)
save_dataset(uiuc_dataset_tc, uiuc_dataset_tc_name)

print("Datasets are saved!")
