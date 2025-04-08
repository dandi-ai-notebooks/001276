# This script loads a small central crop from the large microscopy image stored in the NWB file.
# It examines the dataset shape first, then visualizes fluorescence intensity from the central crop.
# The output image is saved in tmp_scripts/sample_image_preview.png

import lindi
import pynwb
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('tmp_scripts', exist_ok=True)

f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json")
io = pynwb.NWBHDF5IO(file=f, mode='r')
nwbfile = io.read()

img_series = nwbfile.acquisition["SingleTimePointImaging"]
data = img_series.data

print("data shape:", data.shape)

# adjust indexing by discovered shape
# if 2D (Y,X) or 3D (C,Y,X)
shape = data.shape

crop_size = 1000
if len(shape) == 2:
    # shape (Y,X)
    center_y = shape[0] // 2
    center_x = shape[1] // 2
    y_start = max(center_y - crop_size // 2, 0)
    x_start = max(center_x - crop_size // 2, 0)
    crop = data[y_start:y_start+crop_size, x_start:x_start+crop_size]
elif len(shape) == 3:
    # shape (C,Y,X)
    center_y = shape[1] // 2
    center_x = shape[2] // 2
    y_start = max(center_y - crop_size // 2, 0)
    x_start = max(center_x - crop_size // 2, 0)
    crop = data[0, y_start:y_start+crop_size, x_start:x_start+crop_size]
else:
    raise ValueError(f"Unexpected data.ndim = {len(shape)}")

plt.imshow(crop, cmap='gray')
plt.title('Central 1000x1000 crop of fluorescence image')
plt.axis('off')
plt.tight_layout()
plt.savefig('tmp_scripts/sample_image_preview.png', dpi=150)
# do not plt.show()