"""
This script explores the structure of an NWB file from Dandiset 001276.
We want to understand:
1. The basic metadata and structure
2. The nature of the image data (dimensions, content)
3. How to visualize a sample of the image data
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic metadata
print("==== NWB File Metadata ====")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experimenter: {nwb.experimenter}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")
print("\n==== Subject Information ====")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Description: {nwb.subject.description}")

# Check what's available in acquisition
print("\n==== Available Acquisition Data ====")
for key in nwb.acquisition.keys():
    print(f"- {key}")
    
# Get information about the image
image_series = nwb.acquisition["SingleTimePointImaging"]
print("\n==== Image Information ====")
print(f"Image description: {image_series.description}")
print(f"Image dimensions: {image_series.dimension[:]}")
print(f"Image data shape: {image_series.data.shape}")
print(f"Image data type: {image_series.data.dtype}")

# Load a small portion of the image to visualize
# The image is very large (19190x19190), so we'll visualize a small subset
# Start from the center and take a 1000x1000 region
center_x, center_y = image_series.dimension[:] // 2
size = 1000
start_x = center_x - size // 2
start_y = center_y - size // 2
print(f"\nLoading a {size}x{size} region from the center of the image...")
sub_image = image_series.data[0, start_x:start_x+size, start_y:start_y+size]

# Calculate basic statistics
print(f"Subset image shape: {sub_image.shape}")
print(f"Min value: {np.min(sub_image)}")
print(f"Max value: {np.max(sub_image)}")
print(f"Mean value: {np.mean(sub_image)}")
print(f"Std deviation: {np.std(sub_image)}")

# Create a visualization and save it
plt.figure(figsize=(10, 10))
plt.imshow(sub_image, cmap='gray')
plt.title(f"Center region of {nwb.subject.subject_id} - DAPI channel")
plt.colorbar(label='Intensity')
plt.savefig('explore/center_region_image.png', dpi=150)
plt.close()

# Try another region if the center doesn't show much
# Let's try the quadrant regions
quadrant_size = 1000
quadrant_positions = [
    (quadrant_size, quadrant_size),  # Top-left
    (19190 - quadrant_size*2, quadrant_size),  # Top-right
    (quadrant_size, 19190 - quadrant_size*2),  # Bottom-left
    (19190 - quadrant_size*2, 19190 - quadrant_size*2),  # Bottom-right
]

for i, (pos_x, pos_y) in enumerate(quadrant_positions):
    quadrant_img = image_series.data[0, pos_x:pos_x+quadrant_size, pos_y:pos_y+quadrant_size]
    plt.figure(figsize=(10, 10))
    plt.imshow(quadrant_img, cmap='gray')
    plt.title(f"Quadrant {i+1} region of {nwb.subject.subject_id} - DAPI channel")
    plt.colorbar(label='Intensity')
    plt.savefig(f'explore/quadrant_{i+1}_image.png', dpi=150)
    plt.close()

# Close the file
h5_file.close()