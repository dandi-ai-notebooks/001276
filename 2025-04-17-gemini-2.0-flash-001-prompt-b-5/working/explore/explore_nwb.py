import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/95141d7a-82aa-4552-940a-1438a430a0d7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Session description: {nwb.session_description}")
print(f"Subject description: {nwb.subject.description}")
print(f"Shape of image data: {nwb.acquisition['SingleTimePointImaging'].data.shape}")