# %% [markdown]
# # Exploring Dandiset 001276: NG-CANCAN Remote Targeting Electroporation
#
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please interpret the code and results with caution.

# %% [markdown]
# ## Overview
#
# **Study Name:** NG-CANCAN Remote Targeting Electroporation
#
# The study investigates the impact of burst number variation on permeabilization distribution across confluent cell monolayers. Using a four-electrode array, this research focuses on optimizing the CANCAN protocol to minimize cell damage and improve experimental outcomes. More details can be found [here](https://neurosift.app/dandiset/001276).

# %% [markdown]
# ### Summary
#
# This notebook will cover:
# - Loading Dandiset data using the DANDI API
# - Accessing NWB files and extracting relevant imaging data
# - Performing basic analyses and generating visualizations

# %% [markdown]
# ### Required Packages
#
# The following packages are required to run this notebook: `pynwb`, `h5py`, `remfile`, `numpy`, and `matplotlib`.

# %% [markdown]
# ## Loading the Dandiset

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001276")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Load and Explore NWB File Data

# %%
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the specific NWB file
url = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access SingleTimePointImaging data
image_series = nwb.acquisition["SingleTimePointImaging"]
data = image_series.data[0, :, :]

# Visualization
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='viridis')
plt.title("Single Time Point Imaging Data")
plt.colorbar(label="Intensity")
plt.show()

# %% [markdown]
# ## Summary and Future Directions
#
# The visualization depicts imaging data at a single time point, highlighting intensity variations throughout the sample. Future analyses could involve multi-timepoint comparisons and exploring other data modalities.