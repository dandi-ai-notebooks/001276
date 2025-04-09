# This script loads the large SingleTimePointImaging image from an NWB file in DANDI dataset 001276,
# extracts a central 1000x1000 crop, and saves it as a PNG image.
# The goal is to get an overview visualization avoiding the overhead of loading the full ~19kx19k pixel image.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

url = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"

print("Loading remote NWB file...")
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

img_data = nwb.acquisition["SingleTimePointImaging"].data  # shape (1, 19190, 19190)

# Access first frame (only one), huge image
img = img_data[0]  # shape (19190, 19190)

# Compute coordinates for a central 1000x1000 crop
center_x, center_y = img.shape[0] // 2, img.shape[1] // 2
crop_size = 1000
half_crop = crop_size // 2
crop = img[center_x - half_crop:center_x + half_crop, center_y - half_crop:center_y + half_crop]

# Plot and save the crop
plt.figure(figsize=(6,6))
plt.imshow(crop, cmap='gray')
plt.title('Central 1000x1000 crop of SingleTimePointImaging')
plt.axis('off')
plt.savefig('tmp_scripts/SingleTimePointImaging_central_crop.png', bbox_inches='tight')
plt.close()

print("Saved tmp_scripts/SingleTimePointImaging_central_crop.png")