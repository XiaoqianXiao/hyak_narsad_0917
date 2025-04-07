#%%
import os
import json
from bids.layout import BIDSLayout
from templateflow.api import get as tpl_get, templates as get_tpl_list
import pandas as pd
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import subprocess

# Set FSL environment variables for the container
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'  # Matches the Docker image
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Nipype plugin settings for local execution
plugin_settings = {
    'plugin': 'MultiProc',
    'plugin_args': {
        'n_procs': 4,
        'raise_insufficient': False,
        'maxtasksperchild': 1,
    }
}

# Use environment variables for data paths
root_dir = os.getenv('DATA_DIR', '/data')  # Default to /data if not set
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
bids_dir = data_dir
derivatives_dir = os.path.join(data_dir, 'derivatives')
fmriprep_folder = os.path.join(derivatives_dir, 'fmriprep')
behav_dir = os.path.join(data_dir, 'source_data/behav')
#scrubbed_dir = '/gscratch/scrubbed/fanglab/xiaoqian'
scrubbed_dir = '/scrubbed_dir'
# Workflow and output directories
participant_label = []  # Can be set via args or env if needed
run = []
#task = ['phase3']

output_dir = os.path.join(derivatives_dir, 'fMRI_analysis')
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
space = ['MNI152NLin2009cAsym']

# Initialize BIDS layout
layout = BIDSLayout(str(bids_dir), validate=False, derivatives=str(derivatives_dir))
subjects = layout.get(target='subject', return_type='id')
sessions = layout.get(target='session', return_type='id')
runs = layout.get(target='run', return_type='id')

# Query for preprocessed BOLD files
query = {
    'desc': 'preproc',
    'suffix': 'bold',
    'extension': ['.nii', '.nii.gz']
}
if participant_label:
    query['subject'] = '|'.join(participant_label)
if run:
    query['run'] = '|'.join(run)
if task:
    query['task'] = '|'.join(task)
if space:
    query['space'] = '|'.join(space)
prepped_bold = layout.get(**query)
if not prepped_bold:
    print(f'No preprocessed files found under: {derivatives_dir}.')
    exit(1)
entities = prepped_bold[0].entities

# Function to generate Slurm script for a subject
def create_slurm_script(sub, inputs, work_dir, output_dir):
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=first_level_sub_{sub}
#SBATCH --account=<your-account>  # Replace with your Hyak account
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --output={work_dir}/sub_{sub}_%j.out
#SBATCH --error={work_dir}/sub_{sub}_%j.err

module load apptainer
apptainer exec -B {root_dir}:/data narsad-fmri_1.0.sif python3 /app/run_1st_level.py --subject {sub}
"""
    script_path = os.path.join(work_dir, f'sub_{sub}_slurm.sh')
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    return script_path

# Function to run workflow for a single subject
def run_subject_workflow(sub, inputs, work_dir, output_dir):
    from workflows import first_level_wf  # Assumes workflows.py is in /app
    workflow = first_level_wf(inputs, output_dir)
    workflow.base_dir = os.path.join(work_dir, f'sub_{sub}')
    workflow.run(**plugin_settings)

# Main execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run first-level fMRI analysis.")
    parser.add_argument('--subject', type=str, help="Specific subject ID to process")
    args = parser.parse_args()

    if args.subject:  # Run for a single subject (called by Slurm job or manually)
        found = False
        for part in prepped_bold:
            if part.entities['subject'] == args.subject:
                found = True
                entities = part.entities
                sub = entities['subject']
                task = entities['task']
                work_dir = os.path.join(scrubbed_dir, project_name, f'work_flows/firstLevel/{task[0]}')
                if not os.path.exists(work_dir):
                    os.makedirs(work_dir, exist_ok=True)
                # Set events file based on subject and task
                if sub == 'N202' and task == 'phase3':
                    events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_half_events.csv')
                else:
                    events_file = os.path.join(behav_dir, f'task-Narsad_{task}_half_events.csv')
                # Prepare inputs dictionary
                inputs = {sub: {}}
                base = {'subject', 'task'}.intersection(entities)
                subquery = {k: v for k, v in entities.items() if k in base}
                inputs[sub]['bold'] = part.path
                try:
                    inputs[sub]['mask'] = layout.get(suffix='mask', return_type='file',
                                                    extension=['.nii', '.nii.gz'],
                                                    space=query['space'], **subquery)[0]
                    inputs[sub]['regressors'] = layout.get(desc='confounds', return_type='file',
                                                          extension=['.tsv'], **subquery)[0]
                except IndexError as e:
                    print(f"Error: Missing required file (mask or regressors) for subject {sub}")
                    exit(1)
                inputs[sub]['tr'] = entities['RepetitionTime']
                inputs[sub]['events'] = events_file
                print(f"Running first-level analysis for subject {sub}")
                run_subject_workflow(sub, inputs, work_dir, output_dir)
                break
        if not found:
            print(f"Error: Subject {args.subject} not found in preprocessed BOLD files")
            exit(1)
    else:  # Submit Slurm jobs for all subjects
        for part in prepped_bold:
            entities = part.entities
            sub = entities['subject']
            task = entities['task']
            work_dir = os.path.join(scrubbed_dir, project_name, f'work_flows/firstLevel/{task[0]}')
            if not os.path.exists(work_dir):
                os.makedirs(work_dir, exist_ok=True)
            if sub == 'N202' and task == 'phase3':
                events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_half_events.csv')
            else:
                events_file = os.path.join(behav_dir, f'task-Narsad_{task}_half_events.csv')
            inputs = {sub: {}}
            base = {'subject', 'task'}.intersection(entities)
            subquery = {k: v for k, v in entities.items() if k in base}
            inputs[sub]['bold'] = part.path
            inputs[sub]['mask'] = layout.get(suffix='mask', return_type='file',
                                            extension=['.nii', '.nii.gz'],
                                            space=query['space'], **subquery)[0]
            inputs[sub]['regressors'] = layout.get(desc='confounds', return_type='file',
                                                  extension=['.tsv'], **subquery)[0]
            inputs[sub]['tr'] = entities['RepetitionTime']
            inputs[sub]['events'] = events_file
            # Generate and submit Slurm job
            script_path = create_slurm_script(sub, inputs, work_dir, output_dir)
            subprocess.run(['sbatch', script_path], check=True)
            print(f"Submitted Slurm job for subject {sub}")