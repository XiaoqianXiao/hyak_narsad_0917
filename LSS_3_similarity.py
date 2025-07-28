import os
import re
import numpy as np
import nibabel as nib
import argparse
from bids.layout import BIDSLayout
from nipype import config, logging

# ===============
# Project settings
# ===============
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
derivatives_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives')
behav_dir = os.path.join(root_dir, project_name, 'MRI', 'source_data', 'behav')
data_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'firstLevel', 'all_subjects')

scrubbed_dir = '/scrubbed_dir'
space = 'MNI152NLin2009cAsym'
cope = '_cope1'

# Set events file
if sub == 'N202' and task == 'phase3':
    events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_half_events.csv')
else:
    events_file = os.path.join(behav_dir, f'task-Narsad_{task}_half_events.csv')