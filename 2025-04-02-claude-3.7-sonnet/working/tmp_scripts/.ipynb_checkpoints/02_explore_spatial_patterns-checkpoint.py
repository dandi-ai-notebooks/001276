"""
This script explores the spatial distribution of YoPro-1 uptake (FITC channel)
across the field of view to understand the effect of the CANCAN electroporation
protocol, which is designed to target cells in the center of the electrode array.
"""

import matplotlib.pyplot as plt
import numpy as np
import lindi
import pynwb
import time

# Function to safely load an NWB file with appropriate timeout handling
def load_nwb_file(url, max_attempts=3, timeout=30):
    for attempt in range(max_attempts):
        try:
            f = lindi.LindiH5pyFile.from_lindi_file(url)
            nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
            return nwb
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_attempts - 1:
                print(f"Retrying in {timeout} seconds...")
                time.sleep(timeout)
            else:
                raise Exception(f"Failed to load NWB file after {max_attempts} attempts")

# Load several FITC images from different treatments/samples
print("Loading NWB files...")

# 2 burst repetitions (FITC post-exposure)
fitc_2burst_url = "https://lindi.neurosift.org/dandi/dandisets/001276/assets/d22476ad-fa18-4aa0-84bf-13fd0113a52c/nwb.lindi.json"
nwb_fitc_2burst = load_nwb_file(fitc_2burst_url)

# 2 burst repetitions from a different condition
fitc_2burst_alt_url = "https://lindi.neurosift.org/dandi/dandisets/001276/assets/2a3207a3-55e2-4e39-bdda-228da56b9da3/nwb.lindi.json"
nwb_fitc_2burst_alt = load_nwb_file(fitc_2burst_alt_url)

# Create figure for the spatial analysis
plt.figure(figsize=(16, 10))

# Function to downsample the large image for visualization
def downsample_image(img, factor=10):
    """Downsample a large 2D image by taking the mean of blocks"""
    if img.ndim == 3 and img.shape[0] == 1:  # Handle 3D array with single frame
        img = img[0]
    
    h, w = img.shape
    h_ds, w_ds = h // factor, w // factor
    img_ds = np.zeros((h_ds, w_ds), dtype=float)
    
    for i in range(h_ds):
        for j in range(w_ds):
            img_ds[i, j] = np.mean(img[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
    
    return img_ds

# Get FITC images and downsample for visualization
print("Processing images...")
fitc_2burst_img = nwb_fitc_2burst.acquisition['SingleTimePointImaging'].data
fitc_2burst_ds = downsample_image(fitc_2burst_img, factor=20)

fitc_2burst_alt_img = nwb_fitc_2burst_alt.acquisition['SingleTimePointImaging'].data
fitc_2burst_alt_ds = downsample_image(fitc_2burst_alt_img, factor=20)

# Find intensity statistics
fitc_2burst_min, fitc_2burst_max = np.percentile(fitc_2burst_ds, [1, 99.5])
fitc_2burst_alt_min, fitc_2burst_alt_max = np.percentile(fitc_2burst_alt_ds, [1, 99.5])

# Plotting
plt.subplot(2, 2, 1)
plt.imshow(fitc_2burst_ds, cmap='viridis', vmin=fitc_2burst_min, vmax=fitc_2burst_max)
plt.title(f'YoPro-1 Uptake - Sample 1\n{nwb_fitc_2burst.subject.subject_id}')
plt.colorbar(label='Intensity')

plt.subplot(2, 2, 2)
plt.imshow(fitc_2burst_alt_ds, cmap='viridis', vmin=fitc_2burst_alt_min, vmax=fitc_2burst_alt_max)
plt.title(f'YoPro-1 Uptake - Sample 2\n{nwb_fitc_2burst_alt.subject.subject_id}')
plt.colorbar(label='Intensity')

# Create horizontal and vertical intensity profiles to check for central targeting
h_profile_1 = np.mean(fitc_2burst_ds, axis=0)
v_profile_1 = np.mean(fitc_2burst_ds, axis=1)
h_profile_2 = np.mean(fitc_2burst_alt_ds, axis=0)
v_profile_2 = np.mean(fitc_2burst_alt_ds, axis=1)

# Plot horizontal intensity profiles
plt.subplot(2, 2, 3)
plt.plot(h_profile_1, label='Sample 1')
plt.plot(h_profile_2, label='Sample 2')
plt.title('Horizontal Intensity Profile')
plt.xlabel('Position (pixels - downsampled)')
plt.ylabel('Average Intensity')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot vertical intensity profiles
plt.subplot(2, 2, 4)
plt.plot(v_profile_1, label='Sample 1')
plt.plot(v_profile_2, label='Sample 2')
plt.title('Vertical Intensity Profile')
plt.xlabel('Position (pixels - downsampled)')
plt.ylabel('Average Intensity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spatial_patterns.png', dpi=150)
plt.close()

# Print some statistics
print("\nFITC Sample 1 Statistics:")
print(f"Subject: {nwb_fitc_2burst.subject.subject_id}")
desc = nwb_fitc_2burst.subject.description
protocol_info = desc.split("Protocol consisted of")[1].split(".")[0] if "Protocol consisted of" in desc else "Unknown"
print(f"Protocol: {protocol_info}")
print(f"Mean intensity: {np.mean(fitc_2burst_ds):.2f}")
print(f"Max intensity: {np.max(fitc_2burst_ds):.2f}")

print("\nFITC Sample 2 Statistics:")
print(f"Subject: {nwb_fitc_2burst_alt.subject.subject_id}")
desc = nwb_fitc_2burst_alt.subject.description
protocol_info = desc.split("Protocol consisted of")[1].split(".")[0] if "Protocol consisted of" in desc else "Unknown"
print(f"Protocol: {protocol_info}")
print(f"Mean intensity: {np.mean(fitc_2burst_alt_ds):.2f}")
print(f"Max intensity: {np.max(fitc_2burst_alt_ds):.2f}")

print("\nDone!")