import os
import json
import argparse
from bids.layout import BIDSLayout
from nipype.pipeline import engine as pe
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl, utility as niu, io as nio
from niworkflows.interfaces.bids import DerivativesDataSink as BIDSDerivatives
from nipype.interfaces.fsl import SUSAN, ApplyMask, FLIRT, FILMGLS, Level1Design
from pathlib import Path
import numpy as np
import pandas as pd
from nipype.interfaces.base.support import Bunch

# Parse the subject ID argument from the command line
parser = argparse.ArgumentParser(description="Run first-level fMRI analysis.")
parser.add_argument('--subject', required=True, help="Subject ID (e.g., sub-01, sub-02, etc.)")
args = parser.parse_args()

# Set up environment variables for FSL
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'  # Adjust this according to your FSL installation inside the container
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

plugin_settings = {
    'plugin': 'MultiProc',
    'plugin_args': {
        'n_procs': 4,
        'raise_insufficient': False,
        'maxtasksperchild': 1,
    }
}

# Define directories
root_dir = '/data'  # Path to the BIDS directory inside the container
#project_name = 'NARSAD'
data_dir = os.path.join(root_dir, 'MRI')
bids_dir = data_dir
derivatives_dir = os.path.join(data_dir, 'derivatives')
fmriprep_folder = os.path.join(derivatives_dir, 'fmriprep')
behav_dir = os.path.join(data_dir, 'source_data/behav')
scrubbed_dir = '/gscratch/scrubbed/fanglab/xiaoqian'

# Define the list of tasks to process
tasks = ['phase2', 'phase3']

# Loop through each task and process the corresponding data
for task in tasks:
    # Create directories for work and output if they do not exist
    work_dir = os.path.join(scrubbed_dir, f'work_flows/firstLevel/{task}')
    os.makedirs(work_dir, exist_ok=True)
    output_dir = os.path.join(derivatives_dir, 'fMRI_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Set up BIDS Layout and Space parameter
    layout = BIDSLayout(bids_dir, validate=False, derivatives=derivatives_dir)
    space = ['MNI152NLin2009cAsym']

    # Query preprocessed BOLD files for the given subject and task
    query = {
        'desc': 'preproc',
        'suffix': 'bold',
        'extension': ['.nii', '.nii.gz'],
        'subject': args.subject,  # Use the subject passed from the command line
        'task': task,  # Use the current task
    }

    # Get the preprocessed BOLD files
    prepped_bold = layout.get(**query)

    if not prepped_bold:
        print(
            f'No preprocessed files found for subject {args.subject} in task {task} in the derivatives folder: {derivatives_dir}.')
        exit()

    # Process the subject's data
    from workflows import first_level_wf

    for part in prepped_bold:
        base_entities = set(['subject', 'task'])
        inputs = {}
        entities = part.entities
        sub = entities['subject']
        task = entities['task']

        # Determine the path to the events file
        if sub == 'N202' and task == 'phase3':
            events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_half_events.csv')
        else:
            events_file = os.path.join(behav_dir, f'task-Narsad_{task}_half_events.csv')

        # Prepare input data
        inputs[sub] = {}
        base = base_entities.intersection(entities)
        subquery = {k: v for k, v in entities.items() if k in base}
        inputs[sub]['bold'] = part.path
        inputs[sub]['mask'] = \
        layout.get(suffix='mask', return_type='file', extension=['.nii', '.nii.gz'], space=query['space'], **subquery)[
            0]
        inputs[sub]['regressors'] = layout.get(desc='confounds', return_type='file', extension=['.tsv'], **subquery)[0]
        inputs[sub]['tr'] = entities['RepetitionTime']
        inputs[sub]['events'] = events_file

        # Create and run the workflow
        workflow = first_level_wf(inputs, output_dir)
        workflow.base_dir = os.path.join(work_dir, f'sub_{sub}')
        workflow.run(**plugin_settings)

    print(f"Subject {args.subject} processing for task {task} is complete.")
