"""
This script looks for NWB files in the dataset that might contain FITC channel data,
which would show YoPro-1 staining to indicate membrane permeabilization.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import re

def examine_nwb_file(url, asset_id):
    """Examine an NWB file to check what channels it contains"""
    print(f"Examining asset {asset_id}...")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    
    # Extract info from session description
    session_desc = nwb.session_description
    
    # Try to find channel information in the description
    channel_match = re.search(r'Fluorescent Channel: (\w+)', session_desc)
    channel = channel_match.group(1) if channel_match else "Unknown"
    
    # Try to find phase information in the description
    phase_match = re.search(r'Phase: (\w+)', session_desc)
    phase = phase_match.group(1) if phase_match else "Unknown"
    
    print(f"Subject ID: {nwb.subject.subject_id}")
    print(f"Channel: {channel}")
    print(f"Phase: {phase}")
    
    # Check if this is a FITC channel
    is_fitc = "FITC" in channel
    
    # Close the file
    h5_file.close()
    
    return {
        "asset_id": asset_id,
        "subject_id": nwb.subject.subject_id,
        "channel": channel,
        "phase": phase,
        "is_fitc": is_fitc
    }

# Try a few files from the list
# First, let's try the first two we've already looked at
urls = [
    {
        "asset_id": "95141d7a-82aa-4552-940a-1438a430a0d7",
        "url": "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"
    },
    {
        "asset_id": "d22476ad-fa18-4aa0-84bf-13fd0113a52c",
        "url": "https://api.dandiarchive.org/api/assets/d22476ad-fa18-4aa0-84bf-13fd0113a52c/download/"
    }
]

# Let's also try files with different subject names to see if they follow the same pattern
# Using files from the original list
urls.extend([
    {
        "asset_id": "5a8061d9-3757-4a86-8542-2ae90133fdcd",
        "url": "https://api.dandiarchive.org/api/assets/5a8061d9-3757-4a86-8542-2ae90133fdcd/download/"
    },
    {
        "asset_id": "90ab1ffc-03ff-4193-8e47-9bbfbfd56bb5",
        "url": "https://api.dandiarchive.org/api/assets/90ab1ffc-03ff-4193-8e47-9bbfbfd56bb5/download/"
    }
])

# Examine each file
results = []
for item in urls:
    try:
        result = examine_nwb_file(item["url"], item["asset_id"])
        results.append(result)
    except Exception as e:
        print(f"Error examining asset {item['asset_id']}: {str(e)}")

# Print summary
print("\nSummary of examined files:")
print("=" * 60)
print(f"{'Asset ID':<40} {'Subject ID':<20} {'Channel':<10} {'Phase':<10}")
print("-" * 60)
for result in results:
    print(f"{result['asset_id']:<40} {result['subject_id']:<20} {result['channel']:<10} {result['phase']:<10}")