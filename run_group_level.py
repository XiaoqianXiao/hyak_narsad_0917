import os
import shutil
from bids.layout import BIDSLayout
import pandas as pd
from nipype import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataSink
from group_level_workflows import data_prepare_wf, roi_based_wf, wf_ROI
from templateflow.api import get as tpl_get, templates as get_tpl_list

# Set FSL environment
os.environ['FSLDIR'] = '/Users/xiaoqianxiao/fsl'
os.environ['PATH'] = f"{os.environ['FSLDIR']}/share/fsl/bin:{os.environ['FSLDIR']}/bin:{os.environ['PATH']}"

# Define directories
root_dir = '/Users/xiaoqianxiao/projects'
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
derivatives_dir = os.path.join(data_dir, 'derivatives')
workflow_dir = os.path.join(derivatives_dir, 'work_flows/groupLevel')
results_dir = os.path.join(derivatives_dir, 'fMRI_analysis/groupLevel')

for d in [workflow_dir, results_dir]:
    os.makedirs(d, exist_ok=True)

# Define standard reference image (e.g., MNI152 template from FSL)
group_mask = str(tpl_get('MNI152NLin2009cAsym', resolution=2, desc='brain', suffix='mask'))

sub_no_MRI_phase2 = ['N102', 'N208']
sub_no_MRI_phase3 = ['N102', 'N208', 'N120']

SCR_dir = os.path.join(root_dir, project_name, 'EDR')
drug_file = os.path.join(SCR_dir, 'drug_order.csv')
ECR_file = os.path.join(SCR_dir, 'ECR.csv')

# Load behavioral data
df_drug = pd.read_csv(drug_file)
df_drug['group'] = df_drug['subID'].apply(lambda x: 'Patients' if x.startswith('N1') else 'Controls')
df_ECR = pd.read_csv(ECR_file)
df_behav = df_drug.merge(df_ECR, how='left', left_on='subID', right_on='subID')

# Map groups and drugs
group_levels = df_behav['group'].unique()
drug_levels = df_behav['Drug'].unique()
group_map = {level: idx + 1 for idx, level in enumerate(group_levels)}
drug_map = {level: idx + 1 for idx, level in enumerate(drug_levels)}
df_behav['group_id'] = df_behav['group'].map(group_map)
df_behav['drug_id'] = df_behav['Drug'].map(drug_map)

# Load first-level data
firstlevel_dir = os.path.join(derivatives_dir, 'fMRI_analysis/firstlevel')
glayout = BIDSLayout(firstlevel_dir, validate=False, config=['bids', 'derivatives'])
sub_list = sorted(glayout.get_subjects())


contr_list = list(range(1,26))
tasks = ['phase2', 'phase3']

from group_level_workflows import wf_randomise, wf_flameo
if __name__ == "__main__":
    for task in tasks:
        task_results_dir = os.path.join(results_dir, f'task-{task}')
        task_workflow_dir = os.path.join(workflow_dir, f'task-{task}')
        os.makedirs(task_results_dir, exist_ok=True)
        if os.path.exists(task_workflow_dir):
            shutil.rmtree(task_workflow_dir)
        os.makedirs(task_workflow_dir, exist_ok=True)

        for contrast in contr_list:
            design_files_path = os.path.join(task_results_dir, f'cope{contrast}/design_files')
            cope_path = os.path.join(task_results_dir, f'cope{contrast}')
            cope_file_path = os.path.join(cope_path, 'merged_cope.nii.gz')
            varcope_file_path = os.path.join(cope_path, 'merged_varcope.nii.gz')
            mask_file_path = group_mask

            contrast_results_dir = os.path.join(task_results_dir, f'cope{contrast}/whole_brain')
            contrast_workflow_dir = os.path.join(task_workflow_dir, f'cope{contrast}/whole_brain')
            os.makedirs(contrast_results_dir, exist_ok=True)
            os.makedirs(contrast_workflow_dir, exist_ok=True)

            # Choose workflow: 'flameo' or 'randomise'
            analysis_type = 'randomise'  # Switch to 'flameo' to run FLAMEO

            if analysis_type == 'flameo':
                wf_func = wf_flameo
                wf_name = f"wf_flameo_{task}_cope{contrast}"
                cleanup_nodes = ['flameo', 'smoothness', 'clustering']
            else:  # randomise
                wf_func = wf_randomise
                wf_name = f"wf_randomise_{task}_cope{contrast}"
                cleanup_nodes = ['randomise', 'fdr_ztop']

            analysis_wf = wf_func(output_dir=contrast_results_dir, name=wf_name)
            analysis_wf.base_dir = contrast_workflow_dir

            # Set common inputs
            analysis_wf.inputs.inputnode.cope_file = cope_file_path
            analysis_wf.inputs.inputnode.mask_file = mask_file_path
            analysis_wf.inputs.inputnode.design_file = os.path.join(design_files_path, 'design.mat')
            analysis_wf.inputs.inputnode.con_file = os.path.join(design_files_path, 'contrast.con')
            analysis_wf.inputs.inputnode.result_dir = contrast_results_dir

            # Set FLAMEO-specific inputs
            if analysis_type == 'flameo':
                analysis_wf.inputs.inputnode.var_cope_file = varcope_file_path
                analysis_wf.inputs.inputnode.grp_file = os.path.join(design_files_path, 'design.grp')

            analysis_wf.run(plugin='MultiProc', plugin_args={'n_procs': 4})

            # Cleanup
            intermediate_dirs = [os.path.join(contrast_workflow_dir, node) for node in cleanup_nodes]
            for d in intermediate_dirs:
                if os.path.exists(d):
                    shutil.rmtree(d)