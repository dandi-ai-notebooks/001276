"""
This script creates better visualizations of the NWB file image data.
We will:
1. Create a downsampled overview of the entire image
2. Look at regions with proper intensity scaling 
3. Extract meaningful regions for analysis
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("NWB file loaded")
print(f"Image shape: {nwb.acquisition['SingleTimePointImaging'].data.shape}")
print(f"Image dimensions: {nwb.acquisition['SingleTimePointImaging'].dimension[:]}")

# Create a downsampled overview of the entire image
# The image is 19190x19190, so downsampling by factor of 32 gives us ~600x600 thumbnail
print("Creating downsampled overview...")
downsample_factor = 32
image_data = nwb.acquisition['SingleTimePointImaging'].data
# Take a slice to avoid loading the entire image at once
# Sample every 32nd pixel in both dimensions
thumbnail = image_data[0, ::downsample_factor, ::downsample_factor]

print(f"Thumbnail shape: {thumbnail.shape}")
print(f"Thumbnail min value: {np.min(thumbnail)}")
print(f"Thumbnail max value: {np.max(thumbnail)}")
print(f"Thumbnail mean value: {np.mean(thumbnail)}")

# Plot the thumbnail
plt.figure(figsize=(10, 10))
plt.imshow(thumbnail, cmap='gray', vmin=np.percentile(thumbnail, 1), vmax=np.percentile(thumbnail, 99.5))
plt.title(f"Overview of {nwb.subject.subject_id} - DAPI channel (downsampled)")
plt.colorbar(label='Intensity')
plt.savefig('explore/entire_image_overview.png', dpi=150)
plt.close()

# Now let's try to identify meaningful regions from the thumbnail
# Create a heatmap visualization to identify regions with fluorescence
plt.figure(figsize=(10, 10))
plt.imshow(thumbnail, cmap='hot', vmin=np.percentile(thumbnail, 50), vmax=np.percentile(thumbnail, 99.9))
plt.title(f"Heat map of {nwb.subject.subject_id} - DAPI channel (downsampled)")
plt.colorbar(label='Intensity')
plt.savefig('explore/heatmap_overview.png', dpi=150)
plt.close()

# Let's examine a few specific regions in detail
# First, identify potentially interesting coordinates from the thumbnail
# We'll manually pick a few regions that look interesting from the thumbnail 
# and scale the coordinates back up by the downsample factor

# For now, let's look at a region in the center where we expect to find cells
center_coords = [thumbnail.shape[0]//2, thumbnail.shape[1]//2]
# Scale back up to the original image coordinates
center_coords_orig = [c * downsample_factor for c in center_coords]

# Create a function to extract and visualize a region
def visualize_region(coords, size=512, name="region"):
    x, y = coords
    # Make sure we don't go out of bounds
    x_start = max(0, x - size//2)
    y_start = max(0, y - size//2)
    x_end = min(image_data.shape[1], x + size//2)
    y_end = min(image_data.shape[2], y + size//2)
    
    print(f"Extracting region at ({x_start}:{x_end}, {y_start}:{y_end})")
    region = image_data[0, x_start:x_end, y_start:y_end]
    
    # Print basic stats
    print(f"{name} shape: {region.shape}")
    print(f"{name} min: {np.min(region)}")
    print(f"{name} max: {np.max(region)}")
    print(f"{name} mean: {np.mean(region)}")
    print(f"{name} std: {np.std(region)}")
    
    # Create a more sensitive visualization
    plt.figure(figsize=(10, 10))
    # Use percentile-based scaling for better visualization
    vmin = np.percentile(region, 1)
    vmax = np.percentile(region, 99.5)
    plt.imshow(region, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f"{name.capitalize()} of {nwb.subject.subject_id} - DAPI channel")
    plt.colorbar(label='Intensity')
    plt.savefig(f'explore/{name}.png', dpi=150)
    plt.close()
    
    # Create a heatmap visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(region, cmap='hot', vmin=np.percentile(region, 50), vmax=np.percentile(region, 99.5))
    plt.title(f"Heat map of {name} - {nwb.subject.subject_id} - DAPI channel")
    plt.colorbar(label='Intensity')
    plt.savefig(f'explore/{name}_heatmap.png', dpi=150)
    plt.close()

# Visualize the center region
visualize_region(center_coords_orig, size=1024, name="center_region")

# Try to find interesting regions based on the thumbnail
# Look for areas with high intensity that might contain cells
# We'll use a simple threshold to identify potential regions of interest
threshold = np.percentile(thumbnail, 95)
high_intensity_coords = np.argwhere(thumbnail > threshold)

if len(high_intensity_coords) > 0:
    # Pick a few high intensity regions to examine
    num_regions = min(3, len(high_intensity_coords))
    for i in range(num_regions):
        # Take evenly spaced points from the high intensity coordinates
        idx = i * len(high_intensity_coords) // num_regions
        y, x = high_intensity_coords[idx]
        # Scale coordinates back to original image
        orig_coords = [x * downsample_factor, y * downsample_factor]
        visualize_region(orig_coords, size=1024, name=f"high_intensity_region_{i+1}")

# Close the file
h5_file.close()