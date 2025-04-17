import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get image data
image_data = nwb.acquisition['SingleTimePointImaging'].data

# Extract a subset of the data
subset_size = 100
subset_data = image_data[0, 7000:7000+subset_size, 7000:7000+subset_size]

# # Clip the data to the range [0, 20]
# subset_data = np.clip(subset_data, 0, 20)

# Scale the data to the range 0-1
subset_data = subset_data / np.max(subset_data)

# Plot the subset
plt.imshow(subset_data, cmap='gray', vmin=0, vmax=1)
plt.title('Subset of Image Data')
plt.colorbar()
plt.savefig("explore/image_subset.png")
plt.close()