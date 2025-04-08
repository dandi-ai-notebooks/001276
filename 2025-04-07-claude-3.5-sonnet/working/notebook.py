# %% [markdown]
# # Analysis of DANDI:001276 - Remote Targeting Electroporation Dataset
# 
# **NOTE: This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Users should carefully verify all code and interpretations before use.**
#
# This notebook demonstrates how to load and analyze data from DANDI:001276, which contains microscopy data investigating the impact of burst number variation on permeabilization distribution in confluent cell monolayers using CANCAN (with canceling pulses) electroporation protocols.
#
# ## Dataset Overview
#
# This dataset examines cell membrane permeabilization patterns using a four-electrode array setup with:
# - Inter-electrode distance: 10.0 mm
# - Pulse duration: 600 ns
# - Protocol: 9 packets of pulses at 0.2 MHz frequency
# - Protocol repetition: 1, 2, 4, or 6 times at 1 Hz
#
# The study uses two key fluorescent markers:
# - Hoechst (DAPI channel): Stains all cell nuclei
# - YoPro-1 (FITC channel): Indicates membrane permeabilization
#
# ## Required Packages
#
# To run this notebook, you need the following Python packages:
# - pynwb
# - lindi
# - numpy
# - matplotlib
#
# They can be installed using pip:
# ```bash
# pip install pynwb lindi numpy matplotlib
# ```

# %%
# Import required libraries
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient

# %% [markdown]
# ## Accessing the Dataset
#
# First, let's get a list of all assets in the dataset using the DANDI API:

# %%
# Get list of all assets in the dataset
client = DandiAPIClient()
dandiset = client.get_dandiset("001276")
assets = list(dandiset.get_assets())

print(f"Total number of assets: {len(assets)}")
print("\nExample asset paths:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading and Visualizing Image Data
#
# Let's examine a pair of pre/post exposure images to visualize the cell staining patterns. We'll look at:
# 1. DAPI (Hoechst) staining showing cell nuclei
# 2. YoPro-1 staining indicating membrane permeabilization
#
# For this example, we'll use images from subject P1_20240627_A2:

# %%
# Load pre-exposure image (DAPI)
pre_f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json"
)
pre_nwb = pynwb.NWBHDF5IO(file=pre_f, mode='r').read()

# Load post-exposure image (YoPro-1)
post_f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001276/assets/d22476ad-fa18-4aa0-84bf-13fd0113a52c/nwb.lindi.json"
)
post_nwb = pynwb.NWBHDF5IO(file=post_f, mode='r').read()

# Print basic metadata
print("Pre-exposure image metadata:")
print(f"Subject ID: {pre_nwb.subject.subject_id}")
print(f"Image dimensions: {pre_nwb.acquisition['SingleTimePointImaging'].data[0].shape}")
print(f"\nSession description:\n{pre_nwb.session_description[:500]}...")

# %% [markdown]
# ### Visualizing Image Data
#
# The images are quite large (19190 x 19190 pixels), so we'll examine a 1000 x 1000 pixel region from the center of each image:

# %%
# Get image data
pre_image = pre_nwb.acquisition['SingleTimePointImaging'].data[0]
post_image = post_nwb.acquisition['SingleTimePointImaging'].data[0]

# Take center regions
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
plt.show()

# Print basic statistics
print("\nImage Statistics:")
print("\nPre-exposure (DAPI):")
print(f"Mean intensity: {np.mean(pre_subset):.2f}")
print(f"Median intensity: {np.median(pre_subset):.2f}")
print(f"Min/Max intensity: {np.min(pre_subset):.2f} / {np.max(pre_subset):.2f}")
print(f"Standard deviation: {np.std(pre_subset):.2f}")

print("\nPost-exposure (YoPro-1):")
print(f"Mean intensity: {np.mean(post_subset):.2f}")
print(f"Median intensity: {np.median(post_subset):.2f}")
print(f"Min/Max intensity: {np.min(post_subset):.2f} / {np.max(post_subset):.2f}")
print(f"Standard deviation: {np.std(post_subset):.2f}")

# %% [markdown]
# ### Key Observations
#
# 1. **DAPI (Hoechst) Channel**:
#    - Shows distinct nuclear staining as bright spots
#    - Relatively uniform distribution of nuclei across the field
#    - Higher overall intensity range compared to YoPro-1
#
# 2. **YoPro-1 Channel**:
#    - Indicates areas of membrane permeabilization
#    - More clustered signal distribution
#    - Lower overall intensity range
#
# ## Example Analyses
#
# Here are some potential analyses researchers might want to perform with this dataset:
#
# 1. **Quantify Cell Distribution**:
#    - Use intensity thresholding to identify and count nuclei
#    - Analyze spatial distribution patterns
#
# 2. **Measure Permeabilization**:
#    - Compare YoPro-1 intensity patterns between different experimental conditions
#    - Analyze spatial relationship between permeabilization and electrode positions
#
# 3. **Protocol Comparison**:
#    - Compare permeabilization patterns between different burst numbers
#    - Analyze the effect of repetition frequency on membrane permeabilization
#
# ## Next Steps
#
# Researchers can:
# 1. Modify visualization parameters to examine different regions or features
# 2. Implement custom analysis pipelines based on their specific research questions
# 3. Compare results across different experimental conditions in the dataset
# 4. Develop automated analysis workflows for batch processing multiple images

# %% [markdown]
# ## Additional Resources
#
# - [DANDI Archive](https://dandiarchive.org/)
# - [PyNWB Documentation](https://pynwb.readthedocs.io/)
# - [Lindi Documentation](https://github.com/flatironinstitute/lindi)