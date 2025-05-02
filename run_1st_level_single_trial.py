#!/usr/bin/env python3
"""
run_1st_level_single_trial.py

Script to either run single-trial LSA & LSS GLM estimations for specified subjects/tasks,
or generate SLURM submission wrappers for batch processing.
"""
import os
import argparse
import pandas as pd
from bids.layout import BIDSLayout
from first_level_workflows import first_level_single_trial_wf

# Set FSL environment variables
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Nipype plugin settings for parallel execution
plugin_settings = {
    'plugin': 'MultiProc',
    'plugin_args': {
        'n_procs': 4,
        'raise_insufficient': False,
        'maxtasksperchild': 1,
    }
}

# Project paths (override with env or CLI args)
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
bids_dir = data_dir
derivatives_dir = os.path.join(data_dir, 'derivatives')
behav_dir = os.path.join(data_dir, 'source_data', 'behav')
scrubbed_dir = os.getenv('SCRUBBED_DIR', '/scrubbed_dir')
container_path = os.getenv('CONTAINER_IMG', '/scrubbed_dir/images/narsad-fmri_1st_level_1.0.sif')

# Logging directory for SLURM scripts
output_logs = os.path.join(derivatives_dir, 'logs', 'single_trial')
os.makedirs(output_logs, exist_ok=True)

# BIDS space filter
space = ['MNI152NLin2009cAsym']

def create_slurm_script(sub, task):
    """Generate a SLURM script to run single_trial inside the container."""
    work_bind = scrubbed_dir
    work_dir = os.path.join(work_bind, 'workflows/firstLevel', 'single_trial', task)
    os.makedirs(work_dir, exist_ok=True)
    out_dir = os.path.join(output_logs, task)
    os.makedirs(out_dir, exist_ok=True)

    sbatch = f"""#!/bin/bash
#SBATCH --job-name=st_{task}_sub_{sub}
#SBATCH --account=fang
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output={out_dir}/st_{task}_sub_{sub}_%j.out
#SBATCH --error={out_dir}/st_{task}_sub_{sub}_%j.err

module load apptainer

apptainer exec \\
  -B {bids_dir}:/data \\
  -B {scrubbed_dir}:/scrubbed_dir \\
  {container_path} \\
  single_trial \\
    --subject {sub} \\
    --task {task} \\
    --bids_dir /data \\
    --derivatives_dir /data/derivatives \\
    --work_dir /scrubbed_dir/workflows/firstLevel/single_trial/{task}/sub-{sub}
"""
    script_path = os.path.join(work_dir, f'st_sub-{sub}_task-{task}.sh')
    with open(script_path, 'w') as f:
        f.write(sbatch)
    return script_path

def run_subject_workflow(sub, task):
    """Execute the single-trial workflow for one subject/task."""
    layout = BIDSLayout(bids_dir, validate=False, derivatives=derivatives_dir)
    runs = layout.get(subject=sub, task=task,
                      desc='preproc', suffix='bold',
                      space=space, extension=['.nii', '.nii.gz'])
    if not runs:
        raise FileNotFoundError(f"No preprocessed BOLD for sub-{sub} task-{task}")
    bold = runs[0].path
    entities = runs[0].entities
    tr = entities.get('RepetitionTime')
    mask = layout.get(subject=sub, task=task,
                      suffix='mask', extension=['.nii', '.nii.gz'],
                      space=space)[0].path

    events = os.path.join(behav_dir, f'task-Narsad_{task}_half_events.csv')
    if not os.path.exists(events):
        raise FileNotFoundError(f"Events file missing: {events}")

    wf = first_level_single_trial_wf()
    wf.base_dir = os.path.join(
        scrubbed_dir, 'workflows', 'single_trial', task, f'sub-{sub}'
    )

    wf.inputs.inputnode.func_img    = bold
    wf.inputs.inputnode.mask_img    = mask
    wf.inputs.inputnode.events_file = events
    wf.inputs.inputnode.t_r         = tr
    wf.inputs.inputnode.hrf_model   = 'glover'
    wf.inputs.inputnode.out_base    = os.path.join(
        derivatives_dir, 'fMRI_analysis_single_trial',
        f'sub-{sub}', f'task-{task}'
    )

    df = pd.read_csv(events)
    wf.inputs.inputnode.method    = ['LSA', 'LSS']
    wf.inputs.inputnode.trial_idx = sorted(df['trial_idx'].unique())

    wf.run(**plugin_settings)
    print(f"Completed single-trial for sub-{sub} task-{task}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run or generate SLURM scripts for single-trial first-level GLM"
    )
    parser.add_argument('--subject', help="Subject ID (no 'sub-')")
    parser.add_argument('--task',    help="Task label (no 'task-')")
    args = parser.parse_args()

    if args.subject and args.task:
        run_subject_workflow(args.subject, args.task)
    else:
        layout = BIDSLayout(bids_dir, validate=False, derivatives=derivatives_dir)
        query = {
            'desc': 'preproc',
            'suffix': 'bold',
            'extension': ['.nii', '.nii.gz'],
            'space': space
        }
        for run in layout.get(**query):
            sub  = run.entities['subject']
            task = run.entities['task']
            script = create_slurm_script(sub, task)
            print('Wrote SLURM script:', script)
