# Adapted script for LSS-based first-level GLM (one trial per run)
import os
import json
from bids.layout import BIDSLayout
import pandas as pd
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

# Set environment
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/Users/xiaoqianxiao/fsl'  # Adjust this according to your FSL installation
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

plugin_settings = {
    'plugin': 'MultiProc',
    'plugin_args': {'n_procs': 4, 'raise_insufficient': False, 'maxtasksperchild': 1}
}

root_dir = '/Users/xiaoqianxiao/projects'
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
bids_dir = data_dir
derivatives_dir = os.path.join(data_dir, 'derivatives')
fmriprep_folder = os.path.join(derivatives_dir, 'fmriprep')
behav_dir = os.path.join(data_dir, 'source_data/behav')
output_dir = os.path.join(derivatives_dir, 'fMRI_analysis/LSS')
os.makedirs(output_dir, exist_ok=True)

space = ['MNI152NLin2009cAsym']
layout = BIDSLayout(str(bids_dir), validate=False, derivatives=str(derivatives_dir))

# Function to run LSS for one trial
#%%
def run_lss_trial(sub, task, trial_ID, bold_file, mask_file, regressors_file, tr, events_file, work_dir, output_dir):
    from first_level_workflows import first_level_wf_LSS  # your LSS-specific workflow
    in_dict = {
        sub: {
            'bold': bold_file,
            'mask': mask_file,
            'regressors': regressors_file,
            'tr': tr,
            'events': events_file,
            'trial_ID': trial_ID  # critical for LSS
        }
    }
    wf = first_level_wf_LSS(in_dict, output_dir, trial_ID=trial_ID)
    wf.base_dir = os.path.join(work_dir, f'sub-{sub}', f'trial-{trial_ID}')
    wf.run(**plugin_settings)

#%%
#Instructions for IPython interactive use:
#1. Set subject and task manually:
sub = 'N101'
task = 'phase2'

#2. Query files manually:
query = {'desc': 'preproc', 'suffix': 'bold', 'extension': ['.nii', '.nii.gz'], 'subject': sub, 'task': task, 'space': space[0]}
bold_files = layout.get(**query)
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
#%%
#3. Choose and run one trial:
trial_ID = trial_ids[0]  # or any specific ID
work_dir = os.path.join(derivatives_dir, f'work_flows/Lss/{task}')
os.makedirs(work_dir, exist_ok=True)
run_lss_trial(sub, task, trial_ID, bold_file, mask_file, regressors_file, tr, events_file, work_dir, output_dir)
