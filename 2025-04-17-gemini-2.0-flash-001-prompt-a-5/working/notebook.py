# %% [markdown]
# # Exploring Dandiset 001276: NG-CANCAN Remote Targeting Electroporation
#
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified.
# Please be cautious when interpreting the code or results.
#
# This notebook demonstrates how to access and explore data from
# [Dandiset 001276](https://dandiarchive.org/dandiset/001276), which contains data from
# "NG-CANCAN Remote Targeting Electroporation: Impact of Burst Number Variation on Permeabilization Distribution in Confluent Cell Monolayers".
#
# You can also explore this dandiset in neurosift:
#
# https://neurosift.app/dandiset/001276
#
#
# ## What this notebook will cover
#
# 1.  Loading the Dandiset using the DANDI API
# 2.  Loading and visualizing data from an NWB file within the Dandiset
# 3.  Exploring image data from the NWB file.
#
# ## Required packages
#
# The following packages are required to run this notebook:
#
# *   pynwb
# *   h5py
# *   remfile
# *   matplotlib
# *   numpy
# *   seaborn
#
# Make sure these packages are installed in your environment before running the notebook.
#
# ## Loading the Dandiset
#
# This section shows how to load the Dandiset using the DANDI API.

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001276")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading and visualizing data from an NWB file
#
# This section demonstrates how to load data from an NWB file within the Dandiset.
# We will be loading `sub-P1-20240627-A2/sub-P1-20240627-A2_obj-1aoyzxh_image.nwb`.
#
# Here's how to construct the URL for the asset containing the NWB file.

# %%
nwb_file_url = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"

# %%
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load
remote_file = remfile.File(nwb_file_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

nwb

# %% [markdown]
# Let's see some of the NWB file's top-level metadata.

# %%
nwb.session_description

# %%
nwb.identifier

# %%
nwb.session_start_time

# %% [markdown]
# Now let's extract some data from the `SingleTimePointImaging` acquisition. From the output of the `nwb-file-info` tool we know that the data has shape (1, 19190, 19190).

# %%
image_series = nwb.acquisition["SingleTimePointImaging"]
image_data = image_series.data
image_data.shape

# %% [markdown]
# Let's visualize a subset of the image. Note that the entire dataset is large
# so we only load a small portion.  Remember that the image mask values range from 0 to 1.

# %%
subset = image_data[0, 5000:6000, 5000:6000]
plt.imshow(subset)
plt.title("Subset of Image Data")
plt.colorbar()  # Show colorbar for reference
plt.show()

# %% [markdown]
# Now let's inspect the metadata of the subject

# %%
nwb.subject

# %%
nwb.subject.age

# %%
nwb.subject.sex

# %% [markdown]
# ## Summary and future directions
#
# This notebook demonstrated how to load and visualize data from a single NWB file
# within Dandiset 001276, focusing on image data.  Researchers can further explore
# other NWB files within the Dandiset, investigate different acquisitions,
# and perform more advanced analysis on the data, such as image segmentation,
# feature extraction, and comparisons across different experimental conditions. The remote
# file infrastructure enables data science workflows on large datasets.