"""
This script analyzes and compares images from different experimental conditions in the dataset.
It aims to highlight differences between conditions A1, A2, A3, B1, B2, B3, which likely 
represent different burst numbers in the CANCAN electroporation protocol.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import re

# Define a sample of files from different conditions
condition_samples = [
    # A1 condition
    {
        'id': '793a1981-206d-4495-afe9-37377e87acca',  # A1 sample
        'path': 'sub-P1-20240703-A1/sub-P1-20240703-A1_obj-1p7fajh_image.nwb'
    },
    # A2 condition
    {
        'id': '95141d7a-82aa-4552-940a-1438a430a0d7',  # A2 sample 
        'path': 'sub-P1-20240627-A2/sub-P1-20240627-A2_obj-1aoyzxh_image.nwb'
    },
    # A3 condition
    {
        'id': '5a8061d9-3757-4a86-8542-2ae90133fdcd',  # A3 sample
        'path': 'sub-P1-20240627-A3/sub-P1-20240627-A3_obj-1h4rh2m_image.nwb'
    },
    # B1 condition
    {
        'id': 'e671bd0e-531d-4219-b38b-480d6179a7fc',  # B1 sample
        'path': 'sub-P1-20240627-B1/sub-P1-20240627-B1_obj-1qpzwid_image.nwb'
    },
    # B2 condition
    {
        'id': 'ce845c9b-eba3-43d2-aa82-5242b6a19515',  # B2 sample
        'path': 'sub-P1-20240627-B2/sub-P1-20240627-B2_obj-1nit1bi_image.nwb'
    },
    # B3 condition
    {
        'id': 'b8ecbb72-d3a0-41b9-a81e-19719981c8ed',  # B3 sample
        'path': 'sub-P1-20240627-B3/sub-P1-20240627-B3_obj-1j97opj_image.nwb'
    }
]

# Function to load image data and extract information
def load_and_analyze_image(asset_id):
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
        
        # Extract subject ID and condition
        subject_id = nwb.subject.subject_id if hasattr(nwb, 'subject') and hasattr(nwb.subject, 'subject_id') else "Unknown"
        condition_match = re.search(r'P\d+_\d+_([A-Z]\d)', subject_id)
        condition = condition_match.group(1) if condition_match else "Unknown"
        
        # Get the image dimensions
        dimensions = nwb.acquisition["SingleTimePointImaging"].dimension[:] if "SingleTimePointImaging" in nwb.acquisition else None
        
        # Get a small central region of the image
        if "SingleTimePointImaging" in nwb.acquisition:
            center_x = dimensions[0] // 2
            center_y = dimensions[1] // 2
            
            # Define a larger region to capture more meaningful data (500x500 pixels)
            margin = 250
            image_data = nwb.acquisition["SingleTimePointImaging"].data[0, 
                                                              center_x-margin:center_x+margin, 
                                                              center_y-margin:center_y+margin]
            
            # Calculate image statistics
            min_intensity = np.min(image_data)
            max_intensity = np.max(image_data)
            mean_intensity = np.mean(image_data)
            median_intensity = np.median(image_data)
            std_intensity = np.std(image_data)
            
            # Calculate percentiles for better comparison
            p25 = np.percentile(image_data, 25)
            p75 = np.percentile(image_data, 75)
            p90 = np.percentile(image_data, 90)
            p99 = np.percentile(image_data, 99)
            
            return {
                'asset_id': asset_id,
                'subject_id': subject_id,
                'condition': condition,
                'channel': channel,
                'image_data': image_data,
                'min_intensity': min_intensity,
                'max_intensity': max_intensity,
                'mean_intensity': mean_intensity,
                'median_intensity': median_intensity,
                'std_intensity': std_intensity,
                'p25': p25,
                'p75': p75,
                'p90': p90,
                'p99': p99
            }
        else:
            return {
                'asset_id': asset_id,
                'subject_id': subject_id,
                'condition': condition,
                'channel': channel,
                'image_data': None,
                'min_intensity': None,
                'max_intensity': None,
                'mean_intensity': None,
                'median_intensity': None,
                'std_intensity': None,
                'p25': None,
                'p75': None,
                'p90': None,
                'p99': None
            }
    except Exception as e:
        print(f"Error processing {asset_id}: {str(e)}")
        return {
            'asset_id': asset_id,
            'subject_id': None,
            'condition': None,
            'channel': None,
            'image_data': None,
            'min_intensity': None,
            'max_intensity': None,
            'mean_intensity': None,
            'median_intensity': None,
            'std_intensity': None,
            'p25': None,
            'p75': None,
            'p90': None,
            'p99': None
        }

# Analyze each sample
results = []
image_data_by_condition = {}

for sample in condition_samples:
    result = load_and_analyze_image(sample['id'])
    results.append(result)
    
    # Store image data by condition
    if result['image_data'] is not None:
        image_data_by_condition[result['condition']] = {
            'data': result['image_data'],
            'channel': result['channel']
        }

# Create a DataFrame with the results
df = pd.DataFrame([{k: v for k, v in r.items() if k != 'image_data'} for r in results])
print(df[['condition', 'channel', 'mean_intensity', 'std_intensity', 'p90']])

# Save results to CSV
df.to_csv('condition_analysis.csv', index=False)

# 1. Visualize images from different conditions side by side
plt.figure(figsize=(20, 10))
conditions = sorted(image_data_by_condition.keys())

for i, condition in enumerate(conditions):
    plt.subplot(2, 3, i+1)
    
    # Define a colormap based on channel
    cmap = 'Blues' if image_data_by_condition[condition]['channel'] == 'DAPI' else 'Greens'
    
    # Get the 99th percentile for better visualization
    vmax = np.percentile(image_data_by_condition[condition]['data'], 99)
    
    plt.imshow(image_data_by_condition[condition]['data'], cmap=cmap, vmin=0, vmax=vmax)
    plt.colorbar(label="Intensity")
    plt.title(f"Condition {condition} - {image_data_by_condition[condition]['channel']} Channel")
    plt.axis('off')

plt.tight_layout()
plt.savefig('condition_comparison.png', dpi=300)
plt.close()

# 2. Create intensity profiles for different conditions
plt.figure(figsize=(12, 8))

# Group by condition and calculate mean and standard deviation
stats_by_condition = df.groupby(['condition', 'channel']).agg({
    'mean_intensity': ['mean', 'std'],
    'median_intensity': ['mean', 'std'],
    'p90': ['mean', 'std']
}).reset_index()

print("\nIntensity statistics by condition:")
print(stats_by_condition)

# Plot DAPI and FITC channels separately
for channel in ['DAPI', 'FITC']:
    channel_data = df[df['channel'] == channel]
    
    if not channel_data.empty:
        plt.figure(figsize=(15, 5))
        
        # Plot mean intensity
        plt.subplot(1, 3, 1)
        channel_by_condition = channel_data.groupby('condition')['mean_intensity']
        means = channel_by_condition.mean()
        errors = channel_by_condition.std()
        
        means.plot(kind='bar', yerr=errors, capsize=10, 
                   color='blue' if channel == 'DAPI' else 'green',
                   alpha=0.7)
        plt.title(f'Mean Intensity - {channel} Channel')
        plt.ylabel('Intensity')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot P90 (90th percentile)
        plt.subplot(1, 3, 2)
        channel_by_condition = channel_data.groupby('condition')['p90']
        p90_means = channel_by_condition.mean()
        p90_errors = channel_by_condition.std()
        
        p90_means.plot(kind='bar', yerr=p90_errors, capsize=10, 
                       color='blue' if channel == 'DAPI' else 'green',
                       alpha=0.7)
        plt.title(f'90th Percentile Intensity - {channel} Channel')
        plt.ylabel('Intensity')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot intensity distribution (boxplot)
        plt.subplot(1, 3, 3)
        plt.boxplot([channel_data[channel_data['condition'] == c]['mean_intensity'].values 
                    for c in sorted(channel_data['condition'].unique())],
                    labels=sorted(channel_data['condition'].unique()),
                    patch_artist=True,
                    boxprops=dict(facecolor='blue' if channel == 'DAPI' else 'green', alpha=0.7))
        plt.title(f'Intensity Distribution - {channel} Channel')
        plt.ylabel('Mean Intensity')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'intensity_analysis_{channel}.png', dpi=300)
        plt.close()

print("\nAnalysis complete. Results saved to condition_analysis.csv")