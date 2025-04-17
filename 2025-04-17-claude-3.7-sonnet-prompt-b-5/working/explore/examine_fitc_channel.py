"""
This script examines FITC channel data, which shows YoPro-1 staining that indicates 
membrane permeabilization from the electroporation.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the FITC channel file for P1_20240627_A2
fitc_url = "https://api.dandiarchive.org/api/assets/d22476ad-fa18-4aa0-84bf-13fd0113a52c/download/"
print("Loading FITC channel file...")
remote_file = remfile.File(fitc_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Description: {nwb.session_description[:200]}...")

# Get image data information
image_series = nwb.acquisition["SingleTimePointImaging"]
print("\nImage Information:")
print(f"Image dimensions: {image_series.dimension[:]}")
print(f"Image data shape: {image_series.data.shape}")
print(f"Image data type: {image_series.data.dtype}")

# Create a downsampled overview of the entire image
print("\nCreating downsampled overview...")
downsample_factor = 32
image_data = image_series.data
# Sample every 32nd pixel in both dimensions
thumbnail = image_data[0, ::downsample_factor, ::downsample_factor]

print(f"Thumbnail shape: {thumbnail.shape}")
print(f"Thumbnail min value: {np.min(thumbnail)}")
print(f"Thumbnail max value: {np.max(thumbnail)}")
print(f"Thumbnail mean value: {np.mean(thumbnail)}")

# Plot the thumbnail
plt.figure(figsize=(10, 10))
plt.imshow(thumbnail, cmap='viridis', vmin=np.percentile(thumbnail, 1), vmax=np.percentile(thumbnail, 99.5))
plt.title(f"Overview of {nwb.subject.subject_id} - FITC channel (downsampled)")
plt.colorbar(label='Intensity')
plt.savefig('explore/fitc_overview.png', dpi=150)
plt.close()

# Create a heatmap visualization to highlight high-intensity regions
plt.figure(figsize=(10, 10))
plt.imshow(thumbnail, cmap='hot', vmin=np.percentile(thumbnail, 50), vmax=np.percentile(thumbnail, 99.9))
plt.title(f"Heat map of {nwb.subject.subject_id} - FITC channel (downsampled)")
plt.colorbar(label='Intensity')
plt.savefig('explore/fitc_heatmap.png', dpi=150)
plt.close()

# Extract and visualize a center region
def visualize_region(x, y, size=1024, name="region"):
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
    vmin = np.percentile(region, 1)
    vmax = np.percentile(region, 99.5)
    plt.imshow(region, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title(f"{name.capitalize()} of {nwb.subject.subject_id} - FITC channel")
    plt.colorbar(label='Intensity')
    plt.savefig(f'explore/fitc_{name}.png', dpi=150)
    plt.close()
    
    # Create a heatmap visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(region, cmap='hot', vmin=np.percentile(region, 50), vmax=np.percentile(region, 99.5))
    plt.title(f"Heat map of {name} - {nwb.subject.subject_id} - FITC channel")
    plt.colorbar(label='Intensity')
    plt.savefig(f'explore/fitc_{name}_heatmap.png', dpi=150)
    plt.close()
    
    return region

# Extract central region
center_x, center_y = image_series.dimension[:] // 2
center_region = visualize_region(center_x, center_y, size=1024, name="center_region")

# Look for regions with high YoPro-1 uptake (high permeabilization)
# Identify high-intensity regions
threshold = np.percentile(thumbnail, 95)
high_intensity_coords = np.argwhere(thumbnail > threshold)

if len(high_intensity_coords) > 0:
    # Pick a few high intensity regions to examine
    num_regions = min(2, len(high_intensity_coords))
    for i in range(num_regions):
        # Take evenly spaced points from the high intensity coordinates
        idx = i * len(high_intensity_coords) // num_regions
        y, x = high_intensity_coords[idx]
        # Scale coordinates back to original image
        orig_coords = [x * downsample_factor, y * downsample_factor]
        visualize_region(orig_coords[0], orig_coords[1], size=1024, name=f"high_permeability_region_{i+1}")

# Close the file
h5_file.close()

# Let's also compare with a DAPI file for context
# Load the DAPI channel file for the same subject
dapi_url = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"
print("\nLoading DAPI channel file for comparison...")
remote_file = remfile.File(dapi_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb_dapi = io.read()

# Extract center region from DAPI for comparison
image_data_dapi = nwb_dapi.acquisition["SingleTimePointImaging"].data
center_x, center_y = nwb_dapi.acquisition["SingleTimePointImaging"].dimension[:] // 2
x_start = max(0, center_x - 512)
y_start = max(0, center_y - 512)
x_end = min(image_data_dapi.shape[1], center_x + 512)
y_end = min(image_data_dapi.shape[2], center_y + 512)
dapi_region = image_data_dapi[0, x_start:x_end, y_start:y_end]

# Create a comparison visualization (DAPI vs FITC for the same region)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# DAPI image (pre-electroporation, nuclei)
vmin_dapi = np.percentile(dapi_region, 1)
vmax_dapi = np.percentile(dapi_region, 99.5)
im_dapi = axes[0].imshow(dapi_region, cmap='gray', vmin=vmin_dapi, vmax=vmax_dapi)
axes[0].set_title(f"{nwb_dapi.subject.subject_id} - DAPI channel (nuclei)")
plt.colorbar(im_dapi, ax=axes[0], label='Intensity')

# FITC image (post-electroporation, permeabilization)
fitc_region = center_region[:1024, :1024] if center_region.shape[0] > 1024 and center_region.shape[1] > 1024 else center_region
vmin_fitc = np.percentile(fitc_region, 1)
vmax_fitc = np.percentile(fitc_region, 99.5)
im_fitc = axes[1].imshow(fitc_region, cmap='viridis', vmin=vmin_fitc, vmax=vmax_fitc)
axes[1].set_title(f"{nwb.subject.subject_id} - FITC channel (permeabilization)")
plt.colorbar(im_fitc, ax=axes[1], label='Intensity')

plt.tight_layout()
plt.savefig('explore/dapi_fitc_comparison.png', dpi=150)
plt.close()

# Create a merged visualization
plt.figure(figsize=(10, 10))
# Normalize both channels to 0-1 for RGB overlay
dapi_norm = (dapi_region - np.min(dapi_region)) / (np.max(dapi_region) - np.min(dapi_region))
fitc_norm = (fitc_region - np.min(fitc_region)) / (np.max(fitc_region) - np.min(fitc_region))

# Create RGB image (DAPI in blue, FITC in green)
rgb = np.zeros((*dapi_norm.shape, 3))
rgb[:, :, 0] = 0  # Red channel empty
rgb[:, :, 1] = fitc_norm  # FITC in green
rgb[:, :, 2] = dapi_norm  # DAPI in blue

plt.imshow(rgb)
plt.title(f"Merged DAPI (blue) and FITC (green) channels - {nwb.subject.subject_id}")
plt.savefig('explore/dapi_fitc_merged.png', dpi=150)
plt.close()

h5_file.close()
print("\nFITC analysis complete!")