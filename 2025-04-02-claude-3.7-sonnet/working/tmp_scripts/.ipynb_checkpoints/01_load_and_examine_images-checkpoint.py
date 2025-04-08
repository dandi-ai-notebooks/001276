"""
This script explores the basic structure of the NWB files in Dandiset 001276.
It loads both pre-exposure (DAPI) and post-exposure (FITC) images from the same
subject and visualizes a small section of each image to understand data structure.
"""

import matplotlib.pyplot as plt
import numpy as np
import lindi
import pynwb

# Load the pre-exposure DAPI image (Hoechst staining showing all nuclei)
dapi_url = "https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json"
f_dapi = lindi.LindiH5pyFile.from_lindi_file(dapi_url)
nwb_dapi = pynwb.NWBHDF5IO(file=f_dapi, mode='r').read()

# Load the post-exposure FITC image (YoPro-1 showing membrane permeabilization)
fitc_url = "https://lindi.neurosift.org/dandi/dandisets/001276/assets/d22476ad-fa18-4aa0-84bf-13fd0113a52c/nwb.lindi.json"
f_fitc = lindi.LindiH5pyFile.from_lindi_file(fitc_url)
nwb_fitc = pynwb.NWBHDF5IO(file=f_fitc, mode='r').read()

# Print basic information about the files
print("DAPI image details:")
print(f"Subject ID: {nwb_dapi.subject.subject_id}")
print(f"Image description: {nwb_dapi.acquisition['SingleTimePointImaging'].description[:100]}...")
print(f"Image shape: {nwb_dapi.acquisition['SingleTimePointImaging'].data.shape}")
print(f"Image data type: {nwb_dapi.acquisition['SingleTimePointImaging'].data.dtype}")

print("\nFITC image details:")
print(f"Subject ID: {nwb_fitc.subject.subject_id}")
print(f"Image description: {nwb_fitc.acquisition['SingleTimePointImaging'].description[:100]}...")
print(f"Image shape: {nwb_fitc.acquisition['SingleTimePointImaging'].data.shape}")
print(f"Image data type: {nwb_fitc.acquisition['SingleTimePointImaging'].data.dtype}")

# Both images are very large (19190x19190 pixels)
# Let's extract a small central region (500x500 pixels) for visualization
center = 19190 // 2
size = 500
half_size = size // 2

# Extract central regions from both images - images are 3D arrays with shape (1, height, width)
dapi_data = nwb_dapi.acquisition['SingleTimePointImaging'].data[0, 
                                                             center-half_size:center+half_size, 
                                                             center-half_size:center+half_size]
fitc_data = nwb_fitc.acquisition['SingleTimePointImaging'].data[0, 
                                                            center-half_size:center+half_size, 
                                                            center-half_size:center+half_size]

# Create figure for the extracted images
plt.figure(figsize=(12, 6))

# Plot DAPI image (pre-exposure)
plt.subplot(1, 2, 1)
plt.imshow(dapi_data, cmap='Blues')
plt.title('DAPI Channel (Pre-exposure)')
plt.colorbar(label='Intensity')
plt.axis('off')

# Plot FITC image (post-exposure)
plt.subplot(1, 2, 2)
plt.imshow(fitc_data, cmap='Greens') 
plt.title('FITC Channel (Post-exposure)')
plt.colorbar(label='Intensity')
plt.axis('off')

plt.tight_layout()
plt.savefig('dapi_fitc_comparison.png', dpi=150)
plt.close()

# Also look at basic statistics of the image intensities
print("\nDAPI image statistics:")
print(f"Min: {np.min(dapi_data)}, Max: {np.max(dapi_data)}")
print(f"Mean: {np.mean(dapi_data):.2f}, Median: {np.median(dapi_data)}")
print(f"Standard deviation: {np.std(dapi_data):.2f}")

print("\nFITC image statistics:")
print(f"Min: {np.min(fitc_data)}, Max: {np.max(fitc_data)}")
print(f"Mean: {np.mean(fitc_data):.2f}, Median: {np.median(fitc_data)}")
print(f"Standard deviation: {np.std(fitc_data):.2f}")