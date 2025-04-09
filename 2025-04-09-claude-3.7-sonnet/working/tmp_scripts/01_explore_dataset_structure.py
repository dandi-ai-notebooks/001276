"""
This script explores the structure of Dandiset 001276 by examining the file naming pattern
and identifying different experimental conditions.
"""

from dandi.dandiapi import DandiAPIClient
import pandas as pd
import os
import re

# Initialize the DANDI API client
client = DandiAPIClient()
dandiset = client.get_dandiset("001276")

# Get all assets
assets = list(dandiset.get_assets())
print(f"Total number of assets: {len(assets)}")

# Extract information from filenames
data = []
for asset in assets:
    path = asset.path
    parts = path.split('/')
    
    # Extract subject, date, and condition from the filename
    filename = parts[-1]
    match = re.match(r'sub-([^_]+)_.*', filename)
    subject_id = parts[0] if match else None
    
    # Extract if it's a different object within the same subject/condition
    obj_match = re.search(r'obj-([^_]+)', filename)
    obj_id = obj_match.group(1) if obj_match else None
    
    # Extract information about fluorescent channel if available
    # We'll need to examine the file content for this
    
    data.append({
        'path': path,
        'asset_id': asset.identifier,
        'subject_id': subject_id,
        'obj_id': obj_id,
        'size_mb': asset.size / (1024 * 1024)  # Convert to MB
    })

# Create a DataFrame for analysis
df = pd.DataFrame(data)

# Analyze the structure
print("\nFile distribution by subject:")
print(df['subject_id'].value_counts())

print("\nNumber of unique objects:")
print(df['obj_id'].nunique())

# Extract condition information (A1, A2, B1, B2, etc.)
df['condition'] = df['subject_id'].str.extract(r'P\d+-\d++-([A-Z]\d)')

print("\nFile distribution by condition:")
print(df['condition'].value_counts())

# Save the DataFrame for later use
df.to_csv('dataset_structure.csv', index=False)

print("\nTop 10 rows of the dataset:")
print(df.head(10))

# Print some statistics about the dataset
print("\nStatistics about file sizes (MB):")
print(df['size_mb'].describe())