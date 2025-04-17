# %% [markdown]
# # Exploring Dandiset 001276: Impact of Burst Number Variation on Permeabilization Distribution
#
# This Jupyter notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please use caution when interpreting the code or the results it produces.

# %% [markdown]
# ## Overview of Dandiset 001276
#
# - **Title**: NG-CANCAN Remote Targeting Electroporation: Impact of Burst Number Variation on Permeabilization Distribution in Confluent Cell Monolayers
# - **Description**: Experiments were conducted using a four-electrode array focusing on optimizing the CANCAN protocol.
# - [Explore this Dandiset on Neurosift](https://neurosift.app/dandiset/001276)

# %% [markdown]
# ## Summary of Notebook Coverage
#
# This notebook will demonstrate:
# - How to load and explore this dataset using the DANDI API
# - How to open and visualize NWB files
# - Insights into the data and possible directions for further analysis

# %% [markdown]
# ## Required Packages
# 
# The following packages are required to run this notebook:
# - pynwb
# - h5py
# - remfile

# %% [markdown]
# ## Load the Dandiset using the DANDI API

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
# ## Load the NWB File

# %%
import pynwb
import h5py
import remfile

# Load the NWB file from the asset URL
url = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Loaded NWB file with session ID: {nwb.session_id}")

# %% [markdown]
# The NWB file is loaded from the path `sub-P1-20240627-A2/sub-P1-20240627-A2_obj-1aoyzxh_image.nwb`. It contains various metadata which will be accessed below.

# %% [markdown]
# ## Explore NWB Metadata and Data

# %%
metadata = {
    "Session Description": nwb.session_description,
    "Identifier": nwb.identifier,
    "Lab": nwb.lab,
    "Institution": nwb.institution
}
print("Metadata Summary:")
for key, value in metadata.items():
    print(f"{key}: {value}")

# Access data from acquisition section
imaging_data = nwb.acquisition["SingleTimePointImaging"].data
print(f"Data shape: {imaging_data.shape}")

# Visualize a subset of the data
import matplotlib.pyplot as plt

plt.imshow(imaging_data[0, 0:500, 0:500], cmap='hot')
plt.title("Visualization of a subset of imaging data")
plt.show()

# %% [markdown]
# ## Conclusions and Future Directions
# 
# This notebook has demonstrated how to access and explore parts of Dandiset 001276 using the DANDI API and NWB format. Future analysis can delve into specific experimental conditions and further statistical analyses of the imaging data.