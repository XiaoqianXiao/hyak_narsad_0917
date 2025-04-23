#%%
import os
import argparse
from group_level_workflows import wf_randomise, wf_flameo
from nipype import config, logging
from templateflow.api import get as tpl_get

config.set('execution', 'remove_unnecessary_outputs', 'false')
logging.update_logging(config)

# Define directories
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
derivatives_dir = os.path.join(data_dir, 'derivatives')
results_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'groupLevel', 'Placebo')
scrubbed_dir = '/scrubbed_dir'
workflows_dir = os.path.join(scrubbed_dir, project_name, 'work_flows', 'groupLevel', 'Placebo')

# Ensure writable crash-dump folder
crash_dir = os.path.join(results_dir, 'crashdumps')
os.makedirs(crash_dir, exist_ok = True)
config.set('logging', 'crashdump_dir', crash_dir)

def run_group_level_wf(task, contrast, analysis_type, paths):
    wf_name = f"wf_{analysis_type}_{task}_cope{contrast}"

    if analysis_type == 'flameo':
        # one-sample FLAMEO (no covsplit)
        wf = wf_flameo(
            output_dir = paths['result_dir'],
            use_covsplit = False,
            name = wf_name
        )
    else:
        wf = wf_randomise(
            output_dir = paths['result_dir'],
            name = wf_name
        )

    wf.base_dir = paths['workflow_dir']

    wf.inputs.inputnode.cope_file = paths['cope_file']
    wf.inputs.inputnode.mask_file = paths['mask_file']
    wf.inputs.inputnode.design_file = paths['design_file']
    wf.inputs.inputnode.con_file = paths['con_file']
    wf.inputs.inputnode.result_dir = paths['result_dir']

    if analysis_type == 'flameo':
        wf.inputs.inputnode.var_cope_file = paths['varcope_file']

    # run serially to avoid multiprocessing crashes
    wf.run(plugin = 'Linear')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required = True)
    parser.add_argument('--contrast', required = True, type = int)
    parser.add_argument(
        '--analysis_type',
        default = 'randomise',
        choices = ['randomise', 'flameo']
    )
    parser.add_argument('--base_dir', required = True)
    args = parser.parse_args()

    # Standard MNI brain mask
    group_mask = str(
        tpl_get(
            'MNI152NLin2009cAsym',
            resolution = 2, desc = 'brain', suffix = 'mask'
        )
    )

    task = args.task
    contrast = args.contrast

    # Build paths
    contrast_dir = os.path.join(results_dir, f"task-{task}", f"cope{contrast}")
    analysis_dir = os.path.join(contrast_dir, 'whole_brain')
    workflow_dir = os.path.join(workflows_dir, f"task-{task}", f"cope{contrast}", 'whole_brain')

    paths = {
        'result_dir':    analysis_dir,
        'workflow_dir':  workflow_dir,
        'cope_file':     os.path.join(contrast_dir, 'merged_cope.nii.gz'),
        'varcope_file':  os.path.join(contrast_dir, 'merged_varcope.nii.gz'),
        'design_file':   os.path.join(contrast_dir, 'design_files', 'design.mat'),
        'con_file':      os.path.join(contrast_dir, 'design_files', 'contrast.con'),
        'grp_file':      os.path.join(contrast_dir, 'design_files', 'design.grp'),
        'mask_file':     group_mask
    }

    os.makedirs(paths['result_dir'], exist_ok = True)
    os.makedirs(paths['workflow_dir'], exist_ok = True)

    run_group_level_wf(
        task,
        contrast,
        args.analysis_type,
        paths
    )
