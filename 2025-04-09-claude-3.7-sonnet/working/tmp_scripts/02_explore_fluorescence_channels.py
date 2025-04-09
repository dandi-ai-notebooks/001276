"""
This script examines a sample of NWB files to identify the fluorescence channels
used in the experiments and their distribution across different conditions.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# Sample a few files from different conditions to analyze
sample_assets = [
    {
        'id': '95141d7a-82aa-4552-940a-1438a430a0d7',  # A2 sample
        'path': 'sub-P1-20240627-A2/sub-P1-20240627-A2_obj-1aoyzxh_image.nwb'
    },
    {
        'id': 'd22476ad-fa18-4aa0-84bf-13fd0113a52c',  # Another A2 sample
        'path': 'sub-P1-20240627-A2/sub-P1-20240627-A2_obj-fniblx_image.nwb'
    },
    {
        'id': '5a8061d9-3757-4a86-8542-2ae90133fdcd',  # A3 sample
        'path': 'sub-P1-20240627-A3/sub-P1-20240627-A3_obj-1h4rh2m_image.nwb'
    },
    {
        'id': 'e671bd0e-531d-4219-b38b-480d6179a7fc',  # B1 sample
        'path': 'sub-P1-20240627-B1/sub-P1-20240627-B1_obj-1qpzwid_image.nwb'
    }
]

# Function to extract channel information from NWB file
def get_channel_info(asset_id):
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    print(f"Loading {url}")
    
    try:
        file = remfile.File(url)
        f = h5py.File(file)
        io = pynwb.NWBHDF5IO(file=f)
        nwb = io.read()
        
        # Get description which contains channel info
        description = nwb.session_description
        
        # Extract channel information using regex
        channel_match = re.search(r'Fluorescent Channel:\s*(\w+)', description)
        channel = channel_match.group(1) if channel_match else "Unknown"
        
        # Extract subject ID
        subject_id = nwb.subject.subject_id if hasattr(nwb, 'subject') and hasattr(nwb.subject, 'subject_id') else "Unknown"
        
        # Get condition from subject ID
        condition_match = re.search(r'P\d+_\d+_([A-Z]\d)', subject_id)
        condition = condition_match.group(1) if condition_match else "Unknown"
        
        # Get image dimensions
        dimensions = nwb.acquisition["SingleTimePointImaging"].dimension[:] if "SingleTimePointImaging" in nwb.acquisition else None
        
        # Get a small sample of the image data to understand intensity distribution
        if "SingleTimePointImaging" in nwb.acquisition:
            # Get a small central region (100x100 pixels) to avoid loading the entire large image
            center_x = dimensions[0] // 2
            center_y = dimensions[1] // 2
            margin = 50
            image_sample = nwb.acquisition["SingleTimePointImaging"].data[0, 
                                                                      center_x-margin:center_x+margin, 
                                                                      center_y-margin:center_y+margin]
            
            # Get statistics about the image intensity
            min_intensity = np.min(image_sample)
            max_intensity = np.max(image_sample)
            mean_intensity = np.mean(image_sample)
            std_intensity = np.std(image_sample)
            
            # Create a histogram of the intensity values
            plt.figure(figsize=(10, 6))
            plt.hist(image_sample.flatten(), bins=50, alpha=0.7)
            plt.title(f"Intensity Distribution - {channel} Channel (Condition {condition})")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.savefig(f"intensity_histogram_{asset_id}.png")
            plt.close()
            
            # Save a sample image
            plt.figure(figsize=(8, 8))
            plt.imshow(image_sample, cmap='gray')
            plt.colorbar(label="Intensity")
            plt.title(f"Sample Image - {channel} Channel (Condition {condition})")
            plt.savefig(f"sample_image_{asset_id}.png")
            plt.close()
        else:
            min_intensity = max_intensity = mean_intensity = std_intensity = None
        
        return {
            'asset_id': asset_id,
            'subject_id': subject_id,
            'condition': condition,
            'channel': channel,
            'dimensions': dimensions,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity
        }
    except Exception as e:
        print(f"Error processing {asset_id}: {str(e)}")
        return {
            'asset_id': asset_id,
            'subject_id': None,
            'condition': None,
            'channel': None,
            'dimensions': None,
            'min_intensity': None,
            'max_intensity': None,
            'mean_intensity': None,
            'std_intensity': None
        }

# Analyze each sample
results = []
for asset in sample_assets:
    asset_info = get_channel_info(asset['id'])
    results.append(asset_info)

# Create a DataFrame with the results
df = pd.DataFrame(results)
print(df)

# Save results to CSV
df.to_csv('channel_info.csv', index=False)

# Create a bar chart showing channel distribution
channel_counts = df['channel'].value_counts()
plt.figure(figsize=(10, 6))
channel_counts.plot(kind='bar')
plt.title('Distribution of Fluorescence Channels in Sampled Files')
plt.xlabel('Channel')
plt.ylabel('Count')
plt.savefig('channel_distribution.png')
plt.close()

print("\nAnalysis complete. Channel information saved to channel_info.csv")