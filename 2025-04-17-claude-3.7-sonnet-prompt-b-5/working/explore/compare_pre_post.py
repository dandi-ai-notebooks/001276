"""
This script compares pre and post electroporation images from the same experimental condition.
From the metadata, we know that 'pre' indicates imaging prior to exposure and 
'post' indicates imaging of the same well after exposure.

We'll try to find a matching pre/post pair from the dataset to compare the effect of the electroporation.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# The first file we've examined is listed as "phase: pre"
# Let's try to find a matching "post" file from the same sample
# The current file has ID: P1_20240627_A2 (pre)
# We need to look for another file with the same ID but in "post" phase

# Since we haven't downloaded the full list of assets yet, let's compare another file
# Let's load a file from the same experimental series
# We'll choose the second file in the list: "d22476ad-fa18-4aa0-84bf-13fd0113a52c"

def load_nwb_file(url, file_name):
    """Load an NWB file from a URL"""
    print(f"Loading {file_name}...")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    
    print(f"File loaded: {nwb.subject.subject_id}")
    print(f"Description: {nwb.session_description[:200]}...")
    
    return nwb, h5_file

# Load the second file from the same subject (different object)
url1 = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"
url2 = "https://api.dandiarchive.org/api/assets/d22476ad-fa18-4aa0-84bf-13fd0113a52c/download/"

nwb1, h5_file1 = load_nwb_file(url1, "File 1")
nwb2, h5_file2 = load_nwb_file(url2, "File 2")

# Check if they are from the same set but different phases
print("\nComparing files...")
print(f"File 1 subject_id: {nwb1.subject.subject_id}")
print(f"File 2 subject_id: {nwb2.subject.subject_id}")

# Extract a small central region from both images for comparison
def extract_central_region(nwb, size=1024):
    """Extract a central region from the image data"""
    image_data = nwb.acquisition['SingleTimePointImaging'].data
    image_dims = nwb.acquisition['SingleTimePointImaging'].dimension[:]
    
    center_x, center_y = image_dims // 2
    start_x = center_x - size // 2
    start_y = center_y - size // 2
    
    region = image_data[0, start_x:start_x+size, start_y:start_y+size]
    return region

# Extract central regions from both images
region1 = extract_central_region(nwb1)
region2 = extract_central_region(nwb2)

# Create a function to visualize and compare the regions
def compare_regions(region1, region2, name1, name2, output_filename):
    """Visualize and compare two image regions"""
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Calculate intensity limits for consistent display
    vmin = min(np.percentile(region1, 1), np.percentile(region2, 1))
    vmax = max(np.percentile(region1, 99.5), np.percentile(region2, 99.5))
    
    # Plot the first region
    im1 = axes[0].imshow(region1, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{name1}")
    plt.colorbar(im1, ax=axes[0], label='Intensity')
    
    # Plot the second region
    im2 = axes[1].imshow(region2, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{name2}")
    plt.colorbar(im2, ax=axes[1], label='Intensity')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    
    # Also create a difference image
    plt.figure(figsize=(10, 8))
    # Normalize both images to 0-1 range for better comparison
    norm_region1 = (region1 - region1.min()) / (region1.max() - region1.min())
    norm_region2 = (region2 - region2.min()) / (region2.max() - region2.min())
    diff = norm_region2 - norm_region1
    
    # Create a diverging colormap for difference
    plt.imshow(diff, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    plt.colorbar(label='Normalized Intensity Difference')
    plt.title(f"Difference: {name2} - {name1}")
    plt.tight_layout()
    plt.savefig(f"{output_filename.replace('.png', '')}_diff.png", dpi=150)
    plt.close()
    
    # Print basic statistics
    print(f"\n{name1} stats:")
    print(f"Min: {np.min(region1)}, Max: {np.max(region1)}, Mean: {np.mean(region1)}, Std: {np.std(region1)}")
    print(f"\n{name2} stats:")
    print(f"Min: {np.min(region2)}, Max: {np.max(region2)}, Mean: {np.mean(region2)}, Std: {np.std(region2)}")
    

# Compare the central regions
file1_name = nwb1.subject.subject_id + " (Image 1)"
file2_name = nwb2.subject.subject_id + " (Image 2)"
compare_regions(region1, region2, file1_name, file2_name, "explore/comparison_central_regions.png")

# Let's also try a high-intensity region
def find_high_intensity_region(nwb, size=1024):
    """Find a region with high intensity in the image"""
    image_data = nwb.acquisition['SingleTimePointImaging'].data
    
    # Downsample to quickly find high intensity regions
    downsample_factor = 32
    downsampled = image_data[0, ::downsample_factor, ::downsample_factor]
    
    # Find high intensity coordinates
    threshold = np.percentile(downsampled, 95)
    high_intensity_coords = np.argwhere(downsampled > threshold)
    
    if len(high_intensity_coords) > 0:
        # Take the first high intensity point
        y, x = high_intensity_coords[0]
        # Scale coordinates back to original image
        orig_x, orig_y = x * downsample_factor, y * downsample_factor
        
        # Extract the region
        start_x = max(0, orig_x - size//2)
        start_y = max(0, orig_y - size//2)
        end_x = min(image_data.shape[1], orig_x + size//2)
        end_y = min(image_data.shape[2], orig_y + size//2)
        
        region = image_data[0, start_x:end_x, start_y:end_y]
        return region
    else:
        return None

# Extract high intensity regions from both images
hi_region1 = find_high_intensity_region(nwb1)
hi_region2 = find_high_intensity_region(nwb2)

# Compare high intensity regions if found
if hi_region1 is not None and hi_region2 is not None:
    compare_regions(hi_region1, hi_region2, 
                    f"{file1_name} (High Intensity)", 
                    f"{file2_name} (High Intensity)", 
                    "explore/comparison_high_intensity.png")

# Close the files
h5_file1.close()
h5_file2.close()

print("Comparison complete.")