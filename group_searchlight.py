import os
import argparse
import numpy as np
import pandas as pd
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataSink
from nipype.interfaces.fsl.model import FLAMEO, SmoothEstimate, Cluster, Randomise
from nipype.interfaces.fsl.utils import Merge, ImageMaths
import logging
from glob import glob

# Import workflows from group_level_workflows.py
from group_level_workflows import wf_flameo, wf_randomise, create_flexible_design_matrix, save_vest_file

# Configure logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logging()



def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run group-level searchlight analysis for a single map type.')
    parser.add_argument('--map_type', required=True, help='Map type to process (e.g., within-FIXATION, between-FIXATION-CS-)')
    parser.add_argument('--method', choices=['flameo', 'randomise'], default='flameo', help='Analysis method: flameo or randomise')
    args = parser.parse_args()
    map_type = args.map_type
    method = args.method
    logger.info(f"Processing group-level analysis for map type: {map_type}, method: {method}")

    # Paths
    root_dir = os.getenv('DATA_DIR', '/data')
    project_name = 'NARSAD'
    derivatives_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives')
    data_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'firstLevel', 'all_subjects', 'similarity', 'searchlight')

    # Process both tasks
    tasks = ['phase2', 'phase3']
    for task in tasks:
        logger.info(f"Processing task: {task}")
        output_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'groupLevel', 'searchlight', method, task)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory for {task}: {output_dir}")

        # Collect subjects
        subject_files = glob(os.path.join(data_dir, f'sub-*_task-{task}_within-FIXATION.nii.gz'))
        subjects = sorted([os.path.basename(f).split('_')[0].replace('sub-', '') for f in subject_files])
        logger.info(f"Found {len(subjects)} subjects for {task}: {subjects}")

        # Common mask - find the correct session directory
        subject_fmriprep_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives', 'fmriprep', f'sub-{subjects[0]}')
        if not os.path.exists(subject_fmriprep_dir):
            logger.error(f"Subject fmriprep directory not found: {subject_fmriprep_dir}")
            continue
            
        # Find session directories
        session_dirs = [d for d in os.listdir(subject_fmriprep_dir) if d.startswith('ses-')]
        if not session_dirs:
            logger.error(f"No session directories found in {subject_fmriprep_dir}")
            continue
            
        # Use the first session found (you can modify this logic if needed)
        session_name = session_dirs[0]
        mask_file = os.path.join(
            subject_fmriprep_dir, session_name, 'func',
            f'sub-{subjects[0]}_{session_name}_task-{task}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
        )
        logger.info(f"Using mask for {task}: {mask_file}, exists: {os.path.exists(mask_file)}")
        if not os.path.exists(mask_file):
            logger.error(f"Mask file not found for {task}: {mask_file}")
            continue

        # Collect cope files for the specified map type
        cope_files = []
        subjects_with_data = []
        for sub in subjects:
            cope_file = os.path.join(data_dir, f'sub-{sub}_task-{task}_{map_type}.nii.gz')
            if os.path.exists(cope_file):
                cope_files.append(cope_file)
                subjects_with_data.append(sub)
            else:
                logger.warning(f"Cope file missing for sub-{sub}, {map_type}, {task}: {cope_file}")
        if len(cope_files) < 2:
            logger.error(f"Insufficient cope files for {map_type}, {task}: {len(cope_files)} found")
            continue
        logger.info(f"Found {len(cope_files)} cope files for {map_type}, {task}")
        
        # Collect var_cope files only if using FLAMEO
        var_cope_files = []
        if method == 'flameo':
            for sub in subjects_with_data:
                var_cope_file = os.path.join(data_dir, f'sub-{sub}_task-{task}_{map_type}_var.nii.gz')
                if os.path.exists(var_cope_file):
                    var_cope_files.append(var_cope_file)
                else:
                    logger.warning(f"Var_cope file missing for sub-{sub}, {map_type}, {task}: {var_cope_file}")
            if len(var_cope_files) != len(cope_files):
                logger.error(f"Var_cope files missing for FLAMEO: {len(var_cope_files)} vs {len(cope_files)} cope files")
                continue
            logger.info(f"Found {len(var_cope_files)} var_cope files for FLAMEO")

        # Create design matrix and contrasts using only subjects with data
        design, contrasts = create_flexible_design_matrix(subjects_with_data, group_coding='1/0', contrast_type='standard')
        design_file = os.path.join(output_dir, f'design_{map_type}.mat')
        con_file = os.path.join(output_dir, f'contrast_{map_type}.con')
        save_vest_file(design, design_file)
        
        # Save all contrasts in VEST format
        contrast_matrix = np.array(contrasts)
        save_vest_file(contrast_matrix, con_file)
        logger.info(f"Created design for {task}: {design_file}, contrasts: {con_file}")
        logger.info(f"Contrasts: 1) patients>controls, 2) patients<controls, 3) mean_effect_patients, 4) mean_effect_controls")
        
        # Log group information
        patients = [s for s in subjects_with_data if s.startswith('1')]
        controls = [s for s in subjects_with_data if s.startswith('2')]
        logger.info(f"Group analysis: {len(patients)} patients vs {len(controls)} controls")
        
        # Check group balance
        if len(patients) < 1 or len(controls) < 1:
            logger.error(f"Insufficient subjects in one or both groups for {map_type}, {task}: {len(patients)} patients, {len(controls)} controls")
            continue

        # Select workflow from group_level_workflows.py
        wf_name = f'wf_{method}_{map_type}_{task}'
        if method == 'flameo':
            wf = wf_flameo(output_dir=output_dir, name=wf_name)
        else:  # randomise
            wf = wf_randomise(output_dir=output_dir, name=wf_name)

        # Set inputs
        wf.inputs.inputnode.cope_files = cope_files
        wf.inputs.inputnode.mask_file = mask_file
        wf.inputs.inputnode.design_file = design_file
        wf.inputs.inputnode.con_file = con_file
        if method == 'flameo':
            wf.inputs.inputnode.var_cope_files = var_cope_files
            wf.inputs.inputnode.result_dir = os.path.join(output_dir, map_type)

        # Run workflow
        try:
            logger.info(f"Running {method} workflow for {map_type}, {task}")
            wf.run()
            logger.info(f"Completed group-level analysis for {map_type}, {task} with {method}")
        except Exception as e:
            logger.error(f"Error running {method} workflow for {map_type}, {task}: {e}")

if __name__ == '__main__':
    main()