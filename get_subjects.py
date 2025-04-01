import os
from bids import BIDSLayout

# Define dataset path and BIDS cache
bids_dir = os.getenv("BIDS_DIR", "/data")
bids_cache = "/tmp/bids_cache"

# Ensure cache directory exists
os.makedirs(bids_cache, exist_ok=True)

# Initialize BIDSLayout
layout = BIDSLayout(bids_dir, derivatives=True, database_path=bids_cache)

# Get subject IDs
subjects = layout.get_subjects()

# Print subjects space-separated
print(" ".join(subjects))
