"""
This script loads imaging data from a remote NWB file and generates plots to visualize it.
Plots are saved to the explore/ directory and will be analyzed for inclusion in the notebook.
"""
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access SingleTimePointImaging data
image_series = nwb.acquisition["SingleTimePointImaging"]
data = image_series.data[0, :, :]

# Basic statistics
mean_intensity = np.mean(data)
max_intensity = np.max(data)

# Visualization
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='viridis')
plt.title("Single Time Point Imaging Data")
plt.colorbar(label="Intensity")
plt.savefig('explore/single_time_point_imaging.png')
plt.close()

# Print basic statistics to log
print(f"Mean Intensity: {mean_intensity}")
print(f"Max Intensity: {max_intensity}")