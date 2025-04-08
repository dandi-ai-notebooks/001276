"""
This script quantitatively analyzes cell permeabilization by comparing DAPI
(cell nuclei) and FITC (YoPro-1, permeabilized cells) images for the same
sample. The goal is to understand what percentage of cells were permeabilized
by different CANCAN electroporation protocols.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import lindi
import pynwb

# URLs for a DAPI (pre) and FITC (post) image pair from the same subject
dapi_url = "https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json"
fitc_url = "https://lindi.neurosift.org/dandi/dandisets/001276/assets/d22476ad-fa18-4aa0-84bf-13fd0113a52c/nwb.lindi.json"

print("Loading DAPI image...")
f_dapi = lindi.LindiH5pyFile.from_lindi_file(dapi_url)
nwb_dapi = pynwb.NWBHDF5IO(file=f_dapi, mode='r').read()

print("Loading FITC image...")
f_fitc = lindi.LindiH5pyFile.from_lindi_file(fitc_url)
nwb_fitc = pynwb.NWBHDF5IO(file=f_fitc, mode='r').read()

# Get metadata
subject_id = nwb_dapi.subject.subject_id
desc = nwb_dapi.subject.description
protocol_info = desc.split("Protocol consisted of")[1].split(".")[0] if "Protocol consisted of" in desc else "Unknown"

print(f"Subject: {subject_id}")
print(f"Protocol: {protocol_info}")

# Extract a smaller region from the center for analysis (to save memory/time)
# The images are very large (19190x19190), so we'll use a 2000x2000 section
center = 19190 // 2
size = 2000
half_size = size // 2

# Extract the images
dapi_data = nwb_dapi.acquisition['SingleTimePointImaging'].data[0, 
                                                            center-half_size:center+half_size, 
                                                            center-half_size:center+half_size]
fitc_data = nwb_fitc.acquisition['SingleTimePointImaging'].data[0, 
                                                           center-half_size:center+half_size, 
                                                           center-half_size:center+half_size]

print("Processing images...")

# Identify nuclei in DAPI image (representing all cells)
# First, normalize and enhance contrast
dapi_norm = (dapi_data - np.min(dapi_data)) / (np.max(dapi_data) - np.min(dapi_data))

# Apply threshold to identify nuclei
dapi_threshold = np.percentile(dapi_norm, 95)  # Adjust threshold as needed
nuclei_mask = dapi_norm > dapi_threshold
nuclei_mask = ndimage.binary_erosion(nuclei_mask)  # Remove small speckles
nuclei_mask = ndimage.binary_dilation(nuclei_mask)  # Smooth edges
labeled_nuclei, num_nuclei = ndimage.label(nuclei_mask)

# Identify YoPro-1 positive cells in FITC image (permeabilized cells)
# Normalize and enhance contrast
fitc_norm = (fitc_data - np.min(fitc_data)) / (np.max(fitc_data) - np.min(fitc_data))

# Apply threshold to identify permeabilized cells
fitc_threshold = np.percentile(fitc_norm, 95)  # Adjust threshold as needed
permeabilized_mask = fitc_norm > fitc_threshold
permeabilized_mask = ndimage.binary_erosion(permeabilized_mask)  # Remove small speckles
permeabilized_mask = ndimage.binary_dilation(permeabilized_mask)  # Smooth edges
labeled_permeabilized, num_permeabilized = ndimage.label(permeabilized_mask)

# Calculate permeabilization rate
permeabilization_rate = (num_permeabilized / num_nuclei * 100) if num_nuclei > 0 else 0

# Print results
print(f"Number of cells (nuclei): {num_nuclei}")
print(f"Number of permeabilized cells: {num_permeabilized}")
print(f"Permeabilization rate: {permeabilization_rate:.2f}%")

# Visualize the results
plt.figure(figsize=(15, 10))

# Original DAPI image
plt.subplot(2, 3, 1)
plt.imshow(dapi_norm, cmap='Blues')
plt.title('DAPI - Cell Nuclei')
plt.axis('off')

# Original FITC image
plt.subplot(2, 3, 2)
plt.imshow(fitc_norm, cmap='Greens')
plt.title('FITC - YoPro-1 Uptake')
plt.axis('off')

# Overlay of both channels
plt.subplot(2, 3, 3)
overlay = np.zeros((size, size, 3))
overlay[:,:,0] = 0  # Red channel is empty
overlay[:,:,1] = fitc_norm  # Green channel for FITC
overlay[:,:,2] = dapi_norm  # Blue channel for DAPI
plt.imshow(overlay)
plt.title('Overlay (DAPI=blue, FITC=green)')
plt.axis('off')

# Detected nuclei
plt.subplot(2, 3, 4)
plt.imshow(labeled_nuclei, cmap='tab20b')
plt.title(f'Detected Nuclei: {num_nuclei}')
plt.axis('off')

# Detected permeabilized cells
plt.subplot(2, 3, 5)
plt.imshow(labeled_permeabilized, cmap='tab20c')
plt.title(f'Permeabilized Cells: {num_permeabilized}')
plt.axis('off')

# Comparison bar chart
plt.subplot(2, 3, 6)
plt.bar(['Total Cells', 'Permeabilized'], [num_nuclei, num_permeabilized], color=['blue', 'green'])
plt.title(f'Permeabilization Rate: {permeabilization_rate:.2f}%')
plt.ylabel('Cell Count')
plt.yscale('log')  # Use log scale for better visualization if counts differ greatly
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('permeabilization_analysis.png', dpi=150)
plt.close()

print("Analysis complete. Results saved to 'permeabilization_analysis.png'")