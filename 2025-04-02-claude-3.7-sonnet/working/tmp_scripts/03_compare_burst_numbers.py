"""
This script compares the YoPro-1 uptake patterns across different burst numbers
to understand how varying the repetition count affects membrane permeabilization
distribution in the CANCAN protocol.
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

# Function to extract protocol information (burst number) from description
def extract_protocol_info(description):
    try:
        if "Protocol consisted of" in description and "protocol repeated" in description:
            protocol_text = description.split("protocol repeated")[1].split("times")[0].strip()
            return int(protocol_text)
        return "Unknown"
    except:
        return "Unknown"

# URLs for different treatments (based on FITC post-exposure imaging)
# Here we try to find files with different burst numbers
# Just using two files to avoid timeouts
urls = [
    # 2 bursts - P1_20240627_A2
    "https://lindi.neurosift.org/dandi/dandisets/001276/assets/d22476ad-fa18-4aa0-84bf-13fd0113a52c/nwb.lindi.json",
    # 2 bursts - P1_20240702_B1
    "https://lindi.neurosift.org/dandi/dandisets/001276/assets/2a3207a3-55e2-4e39-bdda-228da56b9da3/nwb.lindi.json"
]

print("Loading NWB files...")
nwb_files = []
for url in urls:
    try:
        nwb = load_nwb_file(url)
        nwb_files.append(nwb)
    except Exception as e:
        print(f"Failed to load {url}: {str(e)}")

print(f"Successfully loaded {len(nwb_files)} NWB files")

# Process each file and extract protocol information
samples = []
for nwb in nwb_files:
    subject_id = nwb.subject.subject_id
    burst_number = extract_protocol_info(nwb.subject.description)
    img = nwb.acquisition['SingleTimePointImaging'].data
    img_ds = downsample_image(img, factor=20)
    
    samples.append({
        'subject_id': subject_id,
        'burst_number': burst_number,
        'image': img_ds,
    })

# Create a figure to compare the samples
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes = axes.flatten()

# Sort samples by burst number for easier comparison
samples_sorted = sorted(samples, key=lambda x: x['burst_number'] if isinstance(x['burst_number'], int) else 999)

# Calculate shared colormap scale based on 1st and 99.5th percentile of all images
all_intensities = np.concatenate([s['image'].flatten() for s in samples_sorted])
vmin, vmax = np.percentile(all_intensities, [1, 99.5])

# Plot each sample
for i, sample in enumerate(samples_sorted):
    if i < 4:  # Only plot up to 4 samples
        ax = axes[i]
        im = ax.imshow(sample['image'], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Subject: {sample['subject_id']}\nBurst Number: {sample['burst_number']}")
        
        # Add a colorbar
        plt.colorbar(im, ax=ax, label='YoPro-1 Intensity')

plt.tight_layout()
plt.savefig('burst_number_comparison.png', dpi=150)
plt.close()

# Calculate quantitative metrics for each sample
print("\nQuantitative comparison of samples:")
print("="*50)
print(f"{'Subject ID':<15} {'Burst #':<10} {'Mean':<10} {'Median':<10} {'Max':<10} {'Std':<10}")
print("-"*50)

for sample in samples_sorted:
    img = sample['image']
    print(f"{sample['subject_id']:<15} {str(sample['burst_number']):<10} "
          f"{np.mean(img):<10.2f} {np.median(img):<10.2f} "
          f"{np.max(img):<10.2f} {np.std(img):<10.2f}")