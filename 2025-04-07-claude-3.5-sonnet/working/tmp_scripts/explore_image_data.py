"""
This script loads an NWB file from the dataset and visualizes the image data 
to explore cell staining patterns with DAPI (Hoechst) and YoPro-1 markers.
"""

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the image data
image_series = nwb.acquisition['SingleTimePointImaging']
image_data = image_series.data[0]  # Get first frame since it's a 3D array

# Since this is a large image, let's look at a subset
# Take a 1000 x 1000 section from the center
center_x = image_data.shape[0] // 2
center_y = image_data.shape[1] // 2
subset_size = 1000
subset = image_data[
    center_x - subset_size//2:center_x + subset_size//2,
    center_y - subset_size//2:center_y + subset_size//2
]

print("\nImage data shape before subsetting:", image_data.shape)
print("Subset shape:", subset.shape)

# Create custom colormap (white to blue)
colors = [(1, 1, 1), (0, 0, 1)]
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=n_bins)

# Plot the subset
plt.figure(figsize=(10, 8))
plt.imshow(subset, cmap=cmap)
plt.colorbar(label='Intensity')
plt.title('DAPI Channel Image\n(1000 x 1000 pixel center subset)')
plt.xlabel('X Position (pixels)')
plt.ylabel('Y Position (pixels)')
plt.savefig('tmp_scripts/dapi_image.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a histogram of pixel intensities
plt.figure(figsize=(10, 6))
plt.hist(subset.flatten(), bins=100, color='blue', alpha=0.7)
plt.title('Histogram of DAPI Signal Intensities')
plt.xlabel('Intensity')
plt.ylabel('Count')
plt.savefig('tmp_scripts/intensity_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some basic statistics
print("Image Statistics:")
print(f"Full image shape: {image_data.shape}")
print(f"Subset shape: {subset.shape}")
print(f"Mean intensity: {np.mean(subset):.2f}")
print(f"Median intensity: {np.median(subset):.2f}")
print(f"Min intensity: {np.min(subset):.2f}")
print(f"Max intensity: {np.max(subset):.2f}")
print(f"Standard deviation: {np.std(subset):.2f}")