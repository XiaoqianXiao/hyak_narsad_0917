import os
import json
import pandas as pd
from bids.layout import BIDSLayout
from first_level_workflows import first_level_wf_LSS  # your LSS-specific workflow
from nipype import config, logging

# Set FSL environment (use container-compatible paths if needed)
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Nipype plugin settings
plugin_settings = {
    'plugin': 'MultiProc',
    'plugin_args': {'n_procs': 4, 'raise_insufficient': False, 'maxtasksperchild': 1}
}

config.set('execution', 'remove_unnecessary_outputs', 'false')
logging.update_logging(config)

# Paths
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
derivatives_dir = os.path.join(data_dir, 'derivatives')
behav_dir = os.path.join(data_dir, 'source_data', 'behav')
output_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS')
os.makedirs(output_dir, exist_ok=True)

scrubbed_dir = '/scrubbed_dir'
container_path = "/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif"

space = ['MNI152NLin2009cAsym']
layout = BIDSLayout(str(data_dir), validate=False, derivatives=str(derivatives_dir))


# Trial-level LSS function
def run_lss_trial(sub, task, trial_ID, bold_file, mask_file, regressors_file, tr, events_file, work_dir, output_dir):
    in_dict = {
        sub: {
            'bold': bold_file,
            'mask': mask_file,
            'regressors': regressors_file,
            'tr': tr,
            'events': events_file,
            'trial_ID': trial_ID
        }
    }
    wf = first_level_wf_LSS(in_dict, output_dir)
    wf.base_dir = os.path.join(work_dir, f'sub-{sub}', f'trial-{trial_ID}')
    wf.run(**plugin_settings)


# Slurm script generation
def create_slurm_script(sub, task, trial_ID, work_dir):
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=LSS_{sub}_{trial_ID}
#SBATCH --account=fang
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss/{task}_sub-{sub}_trial-{trial_ID}_%j.out
#SBATCH --error=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss/{task}_sub-{sub}_trial-{trial_ID}_%j.err

module load apptainer
apptainer exec -B /gscratch/fang:/data -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir {container_path} \
    python3 /app/run_LSS.py --subject {sub} --task {task} --trial {trial_ID}
"""
    script_path = os.path.join(work_dir, f'sub_{sub}_trial_{trial_ID}_slurm.sh')
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    return script_path


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--trial', required=True, type=int)
    args = parser.parse_args()

    sub = args.subject
    task = args.task
    trial_ID = args.trial

    query = {
        'desc': 'preproc', 'suffix': 'bold', 'extension': ['.nii', '.nii.gz'],
        'subject': sub, 'task': task, 'space': space[0]
    }
    bold_files = layout.get(**query)
    if not bold_files:
        raise FileNotFoundError(f"No BOLD file found for subject {sub}, task {task}")

    part = bold_files[0]
    entities = part.entities
    subquery = {k: v for k, v in entities.items() if k in ['subject', 'task', 'run']}

    bold_file = part.path
    mask_file = layout.get(suffix='mask', extension=['.nii', '.nii.gz'], space=space[0], **subquery)[0].path
    regressors_file = layout.get(desc='confounds', extension=['.tsv'], **subquery)[0].path
    tr = entities.get('RepetitionTime', 1.5)
    events_file = os.path.join(behav_dir, f'single_trial_task-Narsad_{task}_half_events.csv')

    events_df = pd.read_csv(events_file)
    trial_ids = events_df['trial_ID'].unique()

    work_dir = os.path.join(scrubbed_dir, project_name, f'work_flows/Lss/{task}')
    os.makedirs(work_dir, exist_ok=True)

    # Optionally write SLURM script
    for trial_ID in trial_ids:
        script_path = create_slurm_script(sub, task, trial_ID, work_dir)
        print(f"Slurm script created: {script_path}")

