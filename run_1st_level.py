import os, json
from bids.layout import BIDSLayout
from nipype.pipeline import engine as pe
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl, utility as niu, io as nio
from niworkflows.interfaces.bids import DerivativesDataSink as BIDSDerivatives
from utils import _dict_ds
from utils import _bids2nipypeinfo
from nipype.interfaces.fsl import SUSAN, ApplyMask, FLIRT, FILMGLS, Level1Design
from pathlib import Path
import numpy as np
import pandas as pd
from nipype.interfaces.base.support import Bunch

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/Users/xiaoqianxiao/fsl'  # Adjust this according to your FSL installation
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')
plugin_settings = {
    'plugin': 'MultiProc',
    'plugin_args': {
        'n_procs': 4,
        'raise_insufficient': False,
        'maxtasksperchild': 1,
    }
}

root_dir = '/Users/xiaoqianxiao/projects'
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
bids_dir = data_dir
derivatives_dir = os.path.join(data_dir, 'derivatives')
fmriprep_folder = os.path.join(derivatives_dir, 'fmriprep')
behav_dir = os.path.join(data_dir, 'source_data/behav')

participant_label = []
run = []
task = ['phase3']
work_dir = os.path.join(derivatives_dir, f'work_flows/firstLevel/{task[0]}')
if not os.path.exists(work_dir):
    os.makedirs(work_dir)
output_dir = os.path.join(derivatives_dir, 'fMRI_analysis')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
space = ['MNI152NLin2009cAsym']
layout = BIDSLayout(str(bids_dir), validate=False, derivatives=str(derivatives_dir))
subjects = layout.get(target='subject', return_type='id')
sessions = layout.get(target='session', return_type='id')
runs = layout.get(target='run', return_type='id')

query = {'desc': 'preproc',
         'suffix': 'bold',
         'extension': ['.nii', '.nii.gz']}
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
    print(f'No preprocessed files found under the given derivatives '
          'folder: {derivatives_dir}.')
entities = prepped_bold[0].entities

if __name__ == "__main__":
    for part in prepped_bold:
        from workflows import first_level_wf
        base_entities = set(['subject', 'task'])
        inputs = {}
        entities = part.entities
        sub = entities['subject']
        task = entities['task']
        if (sub == 'N202') & (task == 'phase3'):
            events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_half_events.csv')
        else:
            events_file = os.path.join(behav_dir, 'task-Narsad_'+task+'_half_events.csv')
        session = entities['session']
        inputs[sub] = {}
        base = base_entities.intersection(entities)
        subquery = {k: v for k, v in entities.items() if k in base}
        inputs[sub]['bold'] = part.path
        inputs[sub]['mask'] = layout.get(suffix='mask',
                                            return_type='file',
                                            extension=['.nii', '.nii.gz'],
                                            space=query['space'],
                                            **subquery)[0]
        inputs[sub]['regressors'] = layout.get(desc='confounds',
                                                return_type='file',
                                                extension=['.tsv'],
                                                **subquery)[0]
        inputs[sub]['tr'] = entities['RepetitionTime']
        inputs[sub]['events'] = events_file
        workflow = first_level_wf(inputs, output_dir)
        workflow.base_dir = os.path.join(work_dir, f'sub_{sub}')
        workflow.run(**plugin_settings)


