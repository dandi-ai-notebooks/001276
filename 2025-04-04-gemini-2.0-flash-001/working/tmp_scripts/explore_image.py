import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
# Use the lindi URL from the nwb_file_info tool
try:
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

    # Get the SingleTimePointImaging data
    SingleTimePointImaging = nwb.acquisition["SingleTimePointImaging"]
    image_data = SingleTimePointImaging.data[:]

    # Print some basic information about the image
    print(f"Image shape: {image_data.shape}")
    print(f"Image dtype: {image_data.dtype}")
    print(f"Image max: {np.max(image_data)}")
    print(f"Image min: {np.min(image_data)}")

    # Plot the image
    if image_data.shape == (1, 19190, 19190):
        image_data = image_data[0, :, :]
    elif image_data.shape != (19190, 19190):
        raise ValueError(f"Unexpected image shape: {image_data.shape}")
    plt.imshow(image_data, cmap='gray')
    plt.title("DAPI Image")
    plt.savefig("tmp_scripts/dapi_image.png")

except Exception as e:
    print(f"Error loading or processing NWB file: {e}")