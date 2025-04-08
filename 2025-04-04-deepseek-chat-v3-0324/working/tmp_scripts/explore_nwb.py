#!/usr/bin/env python
"""
Script to explore the structure of an NWB file from Dandiset 001276.
This will help understand what data is available for analysis.
"""

import pynwb
import lindi

print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print("\n=== Basic NWB File Information ===")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")
print(f"Experimenters: {nwb.experimenter}")

print("\n=== Subject Information ===")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Age: {nwb.subject.age}")

print("\n=== Acquisition Data ===")
for key in nwb.acquisition:
    print(f"- {key}: {nwb.acquisition[key]}")
    if hasattr(nwb.acquisition[key], 'description'):
        print(f"  Description: {nwb.acquisition[key].description}")

print("\n=== Processing Modules ===") 
for key in nwb.modules:
    print(f"- {key}")
    module = nwb.modules[key]
    for data_interface in module.data_interfaces:
        print(f"  - {data_interface}")

# Save this output to a file for reference
with open("tmp_scripts/nwb_structure.txt", "w") as f_out:
    import sys
    sys.stdout = f_out
    print("=== Captured NWB Structure ===")
    print(nwb)