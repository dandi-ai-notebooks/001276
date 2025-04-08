#!/usr/bin/env python
"""
Script to visualize a portion of the DAPI image data from the NWB file.
We'll extract a 1000x1000 pixel region to keep the plot manageable.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the image data
img_series = nwb.acquisition["SingleTimePointImaging"]
data = img_series.data

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")

# Extract a 1000x1000 pixel region from the first frame
print("Extracting image subset...")
x_start, y_start = 5000, 5000  # Starting coordinates
subset = data[0, x_start:x_start+1000, y_start:y_start+1000]

# Check if we got any data
if subset.size == 0:
    raise ValueError("No data extracted! Please check the coordinates and data shape.")

print(f"Subset shape: {subset.shape}")

# Create the plot
print("Creating plot...")
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(subset, cmap='gray', 
              vmin=np.percentile(subset, 5), 
              vmax=np.percentile(subset, 95))
plt.colorbar(im, label='Intensity')
plt.title(f"DAPI Staining (Region {x_start}-{x_start+1000}, {y_start}-{y_start+1000})")
plt.xlabel("X position (pixels)")
plt.ylabel("Y position (pixels)")

# Save the plot
output_path = "tmp_scripts/dapi_subset.png"
print(f"Saving plot to {output_path}")
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close()

print("Plot saved successfully")