"""
This script compares pre and post exposure images to visualize cell membrane permeabilization
using DAPI (Hoechst) and YoPro-1 staining.
"""

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load the pre-exposure image (DAPI)
pre_f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json")
pre_nwb = pynwb.NWBHDF5IO(file=pre_f, mode='r').read()

# Load the post-exposure image (using the paired file from same experiment)
post_f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001276/assets/d22476ad-fa18-4aa0-84bf-13fd0113a52c/nwb.lindi.json")
post_nwb = pynwb.NWBHDF5IO(file=post_f, mode='r').read()

# Get the image data
pre_image = pre_nwb.acquisition['SingleTimePointImaging'].data[0]
post_image = post_nwb.acquisition['SingleTimePointImaging'].data[0]

# Take the same center region from both images
center_x = pre_image.shape[0] // 2
center_y = pre_image.shape[1] // 2
subset_size = 1000

pre_subset = pre_image[
    center_x - subset_size//2:center_x + subset_size//2,
    center_y - subset_size//2:center_y + subset_size//2
]
post_subset = post_image[
    center_x - subset_size//2:center_x + subset_size//2,
    center_y - subset_size//2:center_y + subset_size//2
]

# Create figure with pre and post images side by side
plt.figure(figsize=(15, 6))

# Pre-exposure image
plt.subplot(1, 2, 1)
plt.imshow(pre_subset, cmap='Blues')
plt.colorbar(label='Intensity')
plt.title('Pre-exposure\nDAPI (Hoechst) Channel')
plt.xlabel('X Position (pixels)')
plt.ylabel('Y Position (pixels)')

# Post-exposure image
plt.subplot(1, 2, 2)
plt.imshow(post_subset, cmap='Greens')
plt.colorbar(label='Intensity')
plt.title('Post-exposure\nYoPro-1 Channel')
plt.xlabel('X Position (pixels)')

plt.tight_layout()
plt.savefig('tmp_scripts/pre_post_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print basic statistics for comparison
print("\nPre-exposure Image Statistics:")
print(f"Mean intensity: {np.mean(pre_subset):.2f}")
print(f"Median intensity: {np.median(pre_subset):.2f}")
print(f"Min intensity: {np.min(pre_subset):.2f}")
print(f"Max intensity: {np.max(pre_subset):.2f}")
print(f"Standard deviation: {np.std(pre_subset):.2f}")

print("\nPost-exposure Image Statistics:")
print(f"Mean intensity: {np.mean(post_subset):.2f}")
print(f"Median intensity: {np.median(post_subset):.2f}")
print(f"Min intensity: {np.min(post_subset):.2f}")
print(f"Max intensity: {np.max(post_subset):.2f}")
print(f"Standard deviation: {np.std(post_subset):.2f}")