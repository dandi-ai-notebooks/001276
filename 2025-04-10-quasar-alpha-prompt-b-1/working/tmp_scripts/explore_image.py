# This script loads a very large single-frame fluorescence microscopy image from the remote NWB file.
# It generates:
# 1. A thumbnail downsampled 200x200 image of the entire frame.
# 2. A ~500x500 pixel crop from the center of the image.
# 3. Histogram of pixel intensities.
# All plots are saved in tmp_scripts/.

import remfile
import h5py
import pynwb
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

url = "https://api.dandiarchive.org/api/assets/d64469f5-8314-489e-bdd9-201b9cc73649/download/"
file = remfile.File(url)
f = h5py.File(file, 'r')
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwbfile = io.read()

img_data = nwbfile.acquisition['SingleTimePointImaging'].data

# Extract dimensions
shape = img_data.shape  # (1, H, W)
_, H, W = shape

print("Starting center crop extraction (skip full image thumbnail)...")
center_y, center_x = H // 2, W // 2
crop_size = 256
y1 = max(center_y - crop_size // 2, 0)
y2 = min(center_y + crop_size // 2, H)
x1 = max(center_x - crop_size // 2, 0)
x2 = min(center_x + crop_size // 2, W)

crop = img_data[0, y1:y2, x1:x2]
crop = np.array(crop)
print(f"Crop shape: {crop.shape}")

plt.imshow(crop, cmap='gray')
plt.title('Center Crop: {}x{}'.format(crop.shape[0], crop.shape[1]))
plt.axis('off')
plt.savefig('tmp_scripts/center_crop.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved center_crop.png")

print("Starting intensity histogram...")
plt.hist(crop.flatten(), bins=256, log=True)
plt.title('Pixel Intensity Histogram (Center Crop)')
plt.xlabel('Pixel Value')
plt.ylabel('Count (log scale)')
plt.savefig('tmp_scripts/intensity_histogram.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved intensity_histogram.png")

io.close()