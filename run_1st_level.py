import os
import json
import argparse
import logging
from pathlib import Path
from bids.layout import BIDSLayout
from nipype.pipeline import engine as pe
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl, utility as niu, io as nio
from niworkflows.interfaces.bids import DerivativesDataSink as BIDSDerivatives
from nipype.interfaces.fsl import SUSAN, ApplyMask, FLIRT, FILMGLS, Level1Design
import numpy as np
import pandas as pd
from nipype.interfaces.base.support import Bunch
from workflows import first_level_wf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse the subject ID argument from the command line
parser = argparse.ArgumentParser(description="Run first-level fMRI analysis.")
parser.add_argument('--subject', required=True, help="Subject ID (e.g., sub-01, sub-02, etc.)")
args = parser.parse_args()

# Set up environment variables for FSL
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'  # Adjust this if necessary
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Define directories
root_dir = '/data'
data_dir = os.path.join(root_dir, 'MRI')
bids_dir = data_dir
derivatives_dir = os.path.join(data_dir, 'derivatives')
fmriprep_folder = os.path.join(derivatives_dir, 'fmriprep')
behav_dir = os.path.join(data_dir, 'source_data/behav')
scrubbed_dir = '/gscratch/scrubbed/fanglab/xiaoqian'

# Tasks
tasks = ['phase2', 'phase3']

logger.info(f"Processing subject: {args.subject}")

# Loop through tasks
for task in tasks:
    work_dir = os.path.join(scrubbed_dir, f'work_flows/firstLevel/{task}')
    output_dir = os.path.join(derivatives_dir, 'fMRI_analysis')
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # BIDS Layout
    try:
        layout = BIDSLayout(bids_dir, validate=False, derivatives=derivatives_dir)
    except Exception as e:
        logger.error(f"Error loading BIDS Layout: {e}")
        exit(1)

    query = {
        'desc': 'preproc',
        'suffix': 'bold',
        'extension': ['.nii', '.nii.gz'],
        'subject': args.subject,
        'task': task,
    }

    prepped_bold = layout.get(**query)

    if not prepped_bold:
        logger.warning(f'No preprocessed files found for subject {args.subject}, task {task}.')
        continue

    for part in prepped_bold:
        logger.info(f"Processing: {part.path}")

        entities = part.entities
        sub = entities['subject']
        task = entities['task']

        if sub == 'N202' and task == 'phase3':
            events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_half_events.csv')
        else:
            events_file = os.path.join(behav_dir, f'task-Narsad_{task}_half_events.csv')

        try:
            mask = layout.get(
                suffix='mask', return_type='file', extension=['.nii', '.nii.gz'], space='MNI152NLin2009cAsym',
                subject=sub, task=task
            )[0]
            regressors = layout.get(
                desc='confounds', return_type='file', extension=['.tsv'], subject=sub, task=task
            )[0]
        except IndexError as e:
            logger.error(f"Missing mask or confound file for subject {sub}, task {task}. {e}")
            continue

        inputs = {
            sub: {
                'bold': part.path,
                'mask': mask,
                'regressors': regressors,
                'tr': entities.get('RepetitionTime'),
                'events': events_file,
            }
        }

        workflow = first_level_wf(inputs, output_dir)
        workflow.base_dir = os.path.join(work_dir, f'sub_{sub}')
        workflow.run(plugin='MultiProc', plugin_args={'n_procs': 4})

    logger.info(f"Subject {args.subject}, task {task} completed.")
