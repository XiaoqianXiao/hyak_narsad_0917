from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.fsl import Merge, FLIRT
from nipype import DataSink
import os
import shutil
import glob


def create_dummy_design_files(group_info, output_dir):
    """
    Create:
      • design.mat      — the design matrix for FLAMEO
      • design.grp      — a VEST‐formatted group file (one integer per subject)
      • contrast.con    — contrast definitions
    """
    import os
    import subprocess

    design_dir = os.path.join(output_dir, 'design_files')
    os.makedirs(design_dir, exist_ok=True)

    design_file = os.path.join(design_dir, 'design.mat')
    grp_file    = os.path.join(design_dir, 'design.grp')      # will become VEST
    con_file    = os.path.join(design_dir, 'contrast.con')

    drug_ids     = [drug for _, _, drug in group_info]
    unique_drugs = sorted(set(drug_ids))
    n            = len(group_info)

    # --- 1) design.mat & contrast.con as before ---
    if len(unique_drugs) == 1:
        # one‐sample Patients vs Controls
        rows = ['1' if grp == 1 else '-1'
                for _, grp, _ in group_info]
        with open(design_file, 'w') as f:
            f.write("/NumWaves 1\n")
            f.write(f"/NumPoints {n}\n")
            f.write("/Matrix\n")
            f.write("\n".join(rows))

        with open(con_file, 'w') as f:
            f.write("/NumWaves 1\n")
            f.write("/NumContrasts 1\n")
            f.write("/Matrix\n1\n")

    else:
        # full 2×2 ANOVA: [P+Pl, P+Ox, C+Pl, C+Ox]
        design_rows = []
        for _, grp, drug in group_info:
            row = [0,0,0,0]
            if grp == 1:  # Patient
                idx = 0 if drug == unique_drugs[0] else 1
            else:         # Control
                idx = 2 if drug == unique_drugs[0] else 3
            row[idx] = 1
            design_rows.append(" ".join(map(str,row)))
        with open(design_file, 'w') as f:
            f.write("/NumWaves 4\n")
            f.write(f"/NumPoints {n}\n")
            f.write("/Matrix\n")
            f.write("\n".join(design_rows))

        contrasts = [
            "1  1 -1 -1",  # Group
            "1 -1  1 -1",  # Drug
            "1 -1 -1  1",  # Interaction
        ]
        with open(con_file, 'w') as f:
            f.write("/NumWaves 4\n")
            f.write(f"/NumContrasts {len(contrasts)}\n")
            f.write("/Matrix\n")
            f.write("\n".join(contrasts))

    # --- 2) write plain text covsplit, then convert to VEST ---
    covtxt = os.path.join(design_dir, 'covsplit.txt')
    with open(covtxt, 'w') as f:
        for _, grp, _ in group_info:
            f.write(f"{grp}\n")

    # Text2Vest is shipped with FSL; this produces a true VEST file
    subprocess.run(['Text2Vest', covtxt, grp_file], check=True)
    os.remove(covtxt)

    return design_file, grp_file, con_file



def check_file_exists(in_file):
    """Check if a file exists and raise an error if not."""
    print(f"DEBUG: Checking file existence: {in_file}")
    import os
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"File {in_file} does not exist!")
    return in_file


def rename_file(in_file, output_dir, contrast, file_type):
    """Rename the merged file to a simpler name with error checking."""
    print(f"DEBUG: Received in_file: {in_file}, contrast: {contrast}, file_type: {file_type}")
    import shutil
    import os
    try:
        contrast_str = str(int(contrast))
    except (ValueError, TypeError):
        print(f"Warning: Invalid contrast value '{contrast}', defaulting to 'unknown'")
        contrast_str = "unknown"

    new_name = f"merged_{file_type}.nii.gz"
    out_file = os.path.join(output_dir, new_name)

    if os.path.exists(in_file):
        shutil.move(in_file, out_file)
        print(f"Renamed {in_file} -> {out_file}")
    else:
        raise FileNotFoundError(f"Input file {in_file} does not exist!")

    return out_file

def data_prepare_wf(output_dir, contrast, name="data_prepare"):
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node
    inputnode = Node(IdentityInterface(fields=['in_copes', 'in_varcopes', 'group_info', 'result_dir', 'group_mask']),
                     name='inputnode')

    # Design generation
    design_gen = Node(Function(input_names=['group_info', 'output_dir'],
                               output_names=['design_file', 'grp_file', 'con_file'],
                               function=create_dummy_design_files),
                      name='design_gen')
    design_gen.inputs.output_dir = output_dir

    # Merge nodes
    merge_copes = Node(Merge(dimension='t', output_type='NIFTI_GZ'), name='merge_copes')
    merge_varcopes = Node(Merge(dimension='t', output_type='NIFTI_GZ'), name='merge_varcopes')


    # Resample nodes with explicit output file specification
    resample_copes = Node(FLIRT(apply_isoxfm=2), name='resample_copes')
    resample_varcopes = Node(FLIRT(apply_isoxfm=2), name='resample_varcopes')

    # Rename nodes
    rename_copes = Node(Function(input_names=['in_file', 'output_dir', 'contrast', 'file_type'],
                                 output_names=['out_file'],
                                 function=rename_file),
                        name='rename_copes')
    rename_copes.inputs.output_dir = output_dir
    rename_copes.inputs.contrast = contrast
    rename_copes.inputs.file_type = 'cope'

    rename_varcopes = Node(Function(input_names=['in_file', 'output_dir', 'contrast', 'file_type'],
                                    output_names=['out_file'],
                                    function=rename_file),
                           name='rename_varcopes')
    rename_varcopes.inputs.output_dir = output_dir
    rename_varcopes.inputs.contrast = contrast
    rename_varcopes.inputs.file_type = 'varcope'

    # DataSink
    datasink = Node(DataSink(base_directory=output_dir), name="datasink")

    # Workflow connections
    wf.connect([
        (inputnode, design_gen, [('group_info', 'group_info')]),
        (inputnode, merge_copes, [('in_copes', 'in_files')]),
        (inputnode, merge_varcopes, [('in_varcopes', 'in_files')]),
        (inputnode, resample_copes, [('group_mask', 'reference')]),
        (inputnode, resample_varcopes, [('group_mask', 'reference')]),
        (merge_copes, resample_copes, [('merged_file', 'in_file')]),
        (merge_varcopes, resample_varcopes, [('merged_file', 'in_file')]),
        (resample_copes, rename_copes, [('out_file', 'in_file')]),
        (resample_varcopes, rename_varcopes, [('out_file', 'in_file')]),
        (rename_copes, datasink, [('out_file', 'merged_copes')]),
        (rename_varcopes, datasink, [('out_file', 'merged_varcopes')]),
        (design_gen, datasink, [('design_file', 'design_files.design_file'),
                                ('grp_file', 'design_files.grp_file'),
                                ('con_file', 'design_files.con_file')])
    ])

    return wf

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.fsl import ExtractROI, FLAMEO, ImageMaths, SmoothEstimate, Threshold
from nipype import DataSink

def get_roi_files(roi):
    import glob
    import os
    roi_dir = "/Users/xiaoqianxiao/tool/parcellation/ROIs"
    roi_pattern = os.path.join(roi_dir, f'*{roi}*_resampled.nii*')  # Use *roi* to match variations
    roi_files = glob.glob(roi_pattern)  # Expand wildcard into list of files
    if not roi_files:
        raise ValueError(f"No ROI files found matching pattern '{roi_pattern}' in {roi_dir}")
    return roi_files
# Define a standalone function to flatten nested lists
def flatten_zstats(zstats):
    return [item for sublist in zstats for item in sublist]
def roi_based_wf(output_dir, name="roi_based_wf"):
    """Workflow for ROI-based analysis using FLAMEO and statistical thresholding."""
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node for ROI-based workflow
    inputnode = Node(IdentityInterface(fields=['roi','cope_file', 'var_cope_file',
                                               'design_file', 'grp_file', 'con_file', 'result_dir']),
                     name='inputnode')

    # ROI node to fetch ROI files
    roi_node = Node(Function(input_names=['roi'], output_names=['roi_files'],
                             function=get_roi_files),
                    name='roi_node')

    # Masking for copes and varcopes - Iterate over roi_file
    mask_copes = MapNode(ImageMaths(op_string='-mul'),  # Multiply input by mask
                         iterfield=['in_file2'],  # Iterate over ROI masks
                         name='mask_copes')

    mask_varcopes = MapNode(ImageMaths(op_string='-mul'),
                            iterfield=['in_file2'],
                            name='mask_varcopes')

    # FLAMEO for each ROI
    flameo = MapNode(FLAMEO(run_mode='flame1'),
                     iterfield=['cope_file', 'var_cope_file', 'mask_file'],  # Add mask_file to iterfield
                     name='flameo')

    # Statistical thresholding for each ROI
    fdr_ztop = MapNode(ImageMaths(op_string='-ztop', suffix='_pval'),
                       iterfield=['in_file'],
                       name='fdr_ztop')

    smoothness = MapNode(SmoothEstimate(),
                         iterfield=['zstat_file', 'mask_file'],  # Add mask_file to iterfield
                         name='smoothness')

    fwe_thresh = MapNode(Threshold(thresh=0.05, direction='above'),
                         iterfield=['in_file'],
                         name='fwe_thresh')

    # Output node
    outputnode = Node(IdentityInterface(fields=['zstats', 'fdr_thresh', 'fwe_thresh']),
                      name='outputnode')

    # DataSink for ROI analysis outputs
    datasink = Node(DataSink(base_directory=output_dir), name='datasink')

    # Workflow connections
    wf.connect([
        (inputnode, roi_node, [('roi', 'roi')]),
        # ROI files from roi_node
        (roi_node, mask_copes, [('roi_files', 'in_file2')]),  # ROI masks as in_file2
        (roi_node, mask_varcopes, [('roi_files', 'in_file2')]),
        (roi_node, flameo, [('roi_files', 'mask_file')]),
        (roi_node, smoothness, [('roi_files', 'mask_file')]),

        # Inputs to masking
        (inputnode, mask_copes, [('cope_file', 'in_file')]),  # Cope file as in_file
        (inputnode, mask_varcopes, [('var_cope_file', 'in_file')]),  # Varcope file as in_file

        # Masked outputs to FLAMEO
        (mask_copes, flameo, [('out_file', 'cope_file')]),
        (mask_varcopes, flameo, [('out_file', 'var_cope_file')]),

        # Design files to FLAMEO
        (inputnode, flameo, [('design_file', 'design_file'),
                             ('grp_file', 'cov_split_file'),
                             ('con_file', 't_con_file')]),

        # FLAMEO outputs to statistical processing with flattened zstats
        (flameo, fdr_ztop, [(('zstats', flatten_zstats), 'in_file')]),
        (flameo, smoothness, [(('zstats', flatten_zstats), 'zstat_file')]),
        (flameo, fwe_thresh, [(('zstats', flatten_zstats), 'in_file')]),

        # Outputs to outputnode
        (flameo, outputnode, [('zstats', 'zstats')]),
        (fdr_ztop, outputnode, [('out_file', 'fdr_thresh')]),
        (fwe_thresh, outputnode, [('out_file', 'fwe_thresh')]),
        (roi_node, outputnode, [('roi_files', 'roi_files')]),

        # Outputs to DataSink
        (outputnode, datasink, [('zstats', 'zstats'),
                                ('fdr_thresh', 'fdr_thresh'),
                                ('fwe_thresh', 'fwe_thresh'),
                                ('roi_files', 'roi_files')])
    ])

    return wf

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.fsl import FLAMEO, ImageMaths, SmoothEstimate, Threshold, Cluster
from nipype import DataSink
import os

def flatten_zstats(zstats):
    """Flatten a potentially nested list of z-stat file paths into a single list."""
    if not zstats:  # Handle empty input
        return []
    if isinstance(zstats, str):  # If it's a single string, wrap it in a list
        return [zstats]
    if isinstance(zstats[0], list):  # If it's a nested list, flatten it
        return [item for sublist in zstats for item in sublist]
    return zstats  # Already a flat list of strings


def flatten_stats(stats):
    """Flatten a potentially nested list of stat file paths into a single list."""
    if not stats:
        return []
    if isinstance(stats, str):
        return [stats]
    if isinstance(stats[0], list):
        return [item for sublist in stats for item in sublist]
    return stats

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.fsl import Randomise, ImageMaths, SmoothEstimate, Threshold, Cluster
from nipype import DataSink
import os

def flatten_stats(stats):
    """Flatten a potentially nested list of stat file paths into a single list."""
    if not stats:
        return []
    if isinstance(stats, str):
        return [stats]
    if isinstance(stats[0], list):
        return [item for sublist in stats for item in sublist]
    return stats

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.fsl import Randomise, ImageMaths, SmoothEstimate, Threshold, Cluster
from nipype import DataSink
import os

def flatten_stats(stats):
    """Flatten a potentially nested list of stat file paths into a single list."""
    if not stats:
        return []
    if isinstance(stats, str):
        return [stats]
    if isinstance(stats[0], list):
        return [item for sublist in stats for item in sublist]
    return stats

def whole_brain_wf(output_dir, name="whole_brain_wf"):
    """Workflow for whole-brain analysis with Randomise (TFCE) and clustering (GRF with dlh)."""
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node with whole-brain mask
    inputnode = Node(IdentityInterface(fields=['cope_file', 'mask_file', 'design_file', 'con_file', 'result_dir']),
                     name='inputnode')

    # Randomise node (TFCE-based inference)
    randomise = Node(Randomise(num_perm=10000,  # Number of permutations
    #randomise = Node(Randomise(num_perm=5000,  # Number of permutations
                               tfce=True,      # Use TFCE
                               vox_p_values=True),  # Output voxelwise p-values
                     name='randomise')

    # Statistical thresholding nodes (optional for TFCE p-values)
    fdr_ztop = MapNode(ImageMaths(op_string='-ztop', suffix='_pval'),
                       iterfield=['in_file'],
                       name='fdr_ztop')

    # Smoothness estimation for GRF clustering
    smoothness = MapNode(SmoothEstimate(),
                         iterfield=['zstat_file', 'mask_file'],
                         name='smoothness')

    # FWE thresholding (optional for t-stats)
    fwe_thresh = MapNode(Threshold(thresh=0.05, direction='above'),
                         iterfield=['in_file'],
                         name='fwe_thresh')

    # Clustering node with dlh for GRF-based correction
    clustering = MapNode(Cluster(threshold=2.3,  # Z-threshold (e.g., 2.3 or 3.1)
                                 connectivity=26,  # 3D connectivity
                                 out_threshold_file=True,
                                 out_index_file=True,
                                 out_localmax_txt_file=True,  # Local maxima text file
                                 pthreshold=0.05),  # Cluster-level FWE threshold
                         iterfield=['in_file', 'dlh'],
                         name='clustering')

    # Output node with fields for TFCE and GRF comparison
    outputnode = Node(IdentityInterface(fields=['tstat_files', 'tfce_corr_p_files',  # TFCE outputs
                                                'fdr_thresh', 'fwe_thresh',         # Additional thresholding
                                                'cluster_thresh', 'cluster_index', 'cluster_peaks']),  # GRF outputs
                      name='outputnode')

    # DataSink
    datasink = Node(DataSink(base_directory=output_dir), name='datasink')

    # Workflow connections
    wf.connect([
        # Inputs to Randomise
        (inputnode, randomise, [('cope_file', 'in_file'),
                                ('mask_file', 'mask'),
                                ('design_file', 'design_mat'),
                                ('con_file', 'tcon')]),

        # Statistical processing (TFCE-related)
        (randomise, fdr_ztop, [(('t_corrected_p_files', flatten_stats), 'in_file')]),  # Use corrected p-values
        (randomise, smoothness, [(('tstat_files', flatten_stats), 'zstat_file')]),
        (inputnode, smoothness, [('mask_file', 'mask_file')]),
        (randomise, fwe_thresh, [(('tstat_files', flatten_stats), 'in_file')]),

        # Clustering with dlh for GRF correction
        (randomise, clustering, [(('tstat_files', flatten_stats), 'in_file')]),
        (smoothness, clustering, [('dlh', 'dlh')]),

        # Outputs to outputnode
        (randomise, outputnode, [('tstat_files', 'tstat_files'),
                                 ('t_corrected_p_files', 'tfce_corr_p_files')]),  # Correct output name
        (fdr_ztop, outputnode, [('out_file', 'fdr_thresh')]),
        (fwe_thresh, outputnode, [('out_file', 'fwe_thresh')]),
        (clustering, outputnode, [('threshold_file', 'cluster_thresh'),
                                  ('index_file', 'cluster_index'),
                                  ('localmax_txt_file', 'cluster_peaks')]),

        # Outputs to DataSink
        (outputnode, datasink, [('tstat_files', 'stats.@tstats'),
                                ('tfce_corr_p_files', 'stats.@tfce_corr_p'),  # TFCE results
                                ('fdr_thresh', 'fdr_thresh'),
                                ('fwe_thresh', 'fwe_thresh'),
                                ('cluster_thresh', 'cluster_results.@thresh'),  # GRF results
                                ('cluster_index', 'cluster_results.@index'),
                                ('cluster_peaks', 'cluster_results.@peaks')])
    ])

    return wf

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.fsl import FLAMEO, SmoothEstimate, Cluster
from nipype import DataSink
import os

def flatten_stats(stats):
    """Flatten a potentially nested list of stat file paths into a single list."""
    if not stats:
        return []
    if isinstance(stats, str):
        return [stats]
    if isinstance(stats[0], list):
        return [item for sublist in stats for item in sublist]
    return stats

def wf_flameo(output_dir, name="wf_flameo"):
    """Workflow for group-level analysis with FLAMEO and clustering (GRF with dlh)."""
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node
    inputnode = Node(IdentityInterface(fields=['cope_file', 'var_cope_file', 'mask_file',
                                               'design_file', 'grp_file', 'con_file', 'result_dir']),
                     name='inputnode')

    # FLAMEO node
    flameo = Node(FLAMEO(run_mode='flame1'), name='flameo')  # flame1 for mixed effects

    # Smoothness estimation for GRF clustering
    smoothness = MapNode(SmoothEstimate(),
                         iterfield=['zstat_file'],  # Only zstat_file iterates
                         name='smoothness')

    # Clustering node with dlh for GRF-based correction
    clustering = MapNode(Cluster(threshold=2.3,  # Z-threshold (e.g., 2.3 or 3.1)
                                 connectivity=26,  # 3D connectivity
                                 out_threshold_file=True,
                                 out_index_file=True,
                                 out_localmax_txt_file=True,  # Local maxima text file
                                 pthreshold=0.05),  # Cluster-level FWE threshold
                         iterfield=['in_file', 'dlh', 'volume'],
                         name='clustering')

    # Output node
    outputnode = Node(IdentityInterface(fields=['zstats', 'cluster_thresh', 'cluster_index', 'cluster_peaks']),
                      name='outputnode')

    # DataSink
    datasink = Node(DataSink(base_directory=output_dir), name='datasink')

    # Workflow connections
    wf.connect([
        # Inputs to FLAMEO
        (inputnode, flameo, [('cope_file', 'cope_file'),
                             ('var_cope_file', 'var_cope_file'),
                             ('mask_file', 'mask_file'),
                             ('design_file', 'design_file'),
                             ('grp_file', 'cov_split_file'),
                             ('con_file', 't_con_file')]),

        # Smoothness estimation
        (flameo, smoothness, [(('zstats', flatten_stats), 'zstat_file')]),
        (inputnode, smoothness, [('mask_file', 'mask_file')]),  # Single mask, no iteration

        # Clustering with dlh
        (flameo, clustering, [(('zstats', flatten_stats), 'in_file')]),
        (smoothness, clustering, [('volume', 'volume')]),
        (smoothness, clustering, [('dlh', 'dlh')]),

        # Outputs to outputnode
        (flameo, outputnode, [('zstats', 'zstats')]),
        (clustering, outputnode, [('threshold_file', 'cluster_thresh'),
                                  ('index_file', 'cluster_index'),
                                  ('localmax_txt_file', 'cluster_peaks')]),

        # Outputs to DataSink
        (outputnode, datasink, [('zstats', 'stats.@zstats'),
                                ('cluster_thresh', 'cluster_results.@thresh'),
                                ('cluster_index', 'cluster_results.@index'),
                                ('cluster_peaks', 'cluster_results.@peaks')])
    ])

    return wf

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.fsl import Randomise, ImageMaths
from nipype import DataSink
import os

def flatten_stats(stats):
    """Flatten a potentially nested list of stat file paths into a single list."""
    if not stats:
        return []
    if isinstance(stats, str):
        return [stats]
    if isinstance(stats[0], list):
        return [item for sublist in stats for item in sublist]
    return stats

def wf_randomise(output_dir, name="wf_randomise"):
    """Workflow for group-level analysis with Randomise and TFCE."""
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node
    inputnode = Node(IdentityInterface(fields=['cope_file', 'mask_file', 'design_file', 'con_file', 'result_dir']),
                     name='inputnode')

    # Randomise node (TFCE-based inference)
    randomise = Node(Randomise(num_perm=5000,  # Number of permutations
                               tfce=True,      # Use TFCE
                               vox_p_values=True),  # Output voxelwise p-values
                     name='randomise')

    # Optional: Convert TFCE p-values to z-scores for visualization
    fdr_ztop = MapNode(ImageMaths(op_string='-ztop', suffix='_pval'),
                       iterfield=['in_file'],
                       name='fdr_ztop')

    # Output node
    outputnode = Node(IdentityInterface(fields=['tstat_files', 'tfce_corr_p_files', 'fdr_thresh']),
                      name='outputnode')

    # DataSink
    datasink = Node(DataSink(base_directory=output_dir), name='datasink')

    # Workflow connections
    wf.connect([
        # Inputs to Randomise
        (inputnode, randomise, [('cope_file', 'in_file'),
                                ('mask_file', 'mask'),
                                ('design_file', 'design_mat'),
                                ('con_file', 'tcon')]),

        # Optional TFCE p-value conversion
        (randomise, fdr_ztop, [(('t_corrected_p_files', flatten_stats), 'in_file')]),

        # Outputs to outputnode
        (randomise, outputnode, [('tstat_files', 'tstat_files'),
                                 ('t_corrected_p_files', 'tfce_corr_p_files')]),
        (fdr_ztop, outputnode, [('out_file', 'fdr_thresh')]),

        # Outputs to DataSink
        (outputnode, datasink, [('tstat_files', 'stats.@tstats'),
                                ('tfce_corr_p_files', 'stats.@tfce_corr_p'),
                                ('fdr_thresh', 'stats.@fdr_thresh')])
    ])

    return wf


from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.fsl import ImageStats
from nipype import DataSink
import os
import glob
import pandas as pd
import numpy as np


def get_roi_files(roi_dir):
    """Retrieve list of ROI mask files from directory."""
    import os, glob
    roi_files = sorted(glob.glob(os.path.join(roi_dir, '*.nii.gz')))
    if not roi_files:
        raise ValueError(f"No ROI files found in {roi_dir}")
    return roi_files


def extract_roi_values(cope_file, roi_mask, baseline_file=None, output_dir=None):
    """Extract mean beta values (and PSC if baseline provided) for a single ROI across subjects."""
    from nipype.interfaces.fsl import ImageStats
    import os
    import numpy as np

    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Compute mean beta values within ROI for each subject
    stats = ImageStats(in_file=cope_file, op_string='-k %s -m', mask_file=roi_mask)
    result = stats.run()
    beta_values = np.array(result.outputs.out_stat)  # Mean beta per subject

    # If baseline_file is provided, compute PSC
    if baseline_file:
        baseline_stats = ImageStats(in_file=baseline_file, op_string='-k %s -m', mask_file=roi_mask)
        baseline_result = baseline_stats.run()
        baseline_values = np.array(baseline_result.outputs.out_stat)
        # Ensure baseline matches cope_file in length
        if len(baseline_values) != len(beta_values):
            raise ValueError("Baseline file subject count does not match cope file")
        # Compute PSC: (beta / baseline) * 100
        psc_values = (beta_values / baseline_values) * 100
    else:
        psc_values = None  # No PSC without baseline

    # Save to text files
    roi_name = os.path.basename(roi_mask).replace('.nii.gz', '')
    beta_file = os.path.join(output_dir, f'beta_{roi_name}.txt')
    np.savetxt(beta_file, beta_values, fmt='%.6f')

    if psc_values is not None:
        psc_file = os.path.join(output_dir, f'psc_{roi_name}.txt')
        np.savetxt(psc_file, psc_values, fmt='%.6f')
        return beta_file, psc_file
    return beta_file, None


def combine_roi_values(beta_files, psc_files, output_dir):
    """Combine beta and PSC values from all ROIs into CSV files."""
    import pandas as pd
    import os

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Combine beta values
    beta_data = {}
    for beta_file in beta_files:
        roi_name = os.path.basename(beta_file).replace('beta_', '').replace('.txt', '')
        beta_values = np.loadtxt(beta_file)
        beta_data[roi_name] = beta_values
    beta_df = pd.DataFrame(beta_data)
    beta_df.index.name = 'subject'
    beta_csv = os.path.join(output_dir, 'beta_all_rois.csv')
    beta_df.to_csv(beta_csv)

    # Combine PSC values if available
    if psc_files and all(f is not None for f in psc_files):
        psc_data = {}
        for psc_file in psc_files:
            roi_name = os.path.basename(psc_file).replace('psc_', '').replace('.txt', '')
            psc_values = np.loadtxt(psc_file)
            psc_data[roi_name] = psc_values
        psc_df = pd.DataFrame(psc_data)
        psc_df.index.name = 'subject'
        psc_csv = os.path.join(output_dir, 'psc_all_rois.csv')
        psc_df.to_csv(psc_csv)
        return beta_csv, psc_csv
    return beta_csv, None


def wf_ROI(output_dir, roi_dir="/Users/xiaoqianxiao/tool/parcellation/ROIs", name="wf_ROI"):
    """Workflow to extract ROI beta values and PSC from fMRI data."""
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node
    inputnode = Node(IdentityInterface(fields=['cope_file', 'baseline_file', 'result_dir']),
                     name='inputnode')

    # Node to get ROI files
    roi_node = Node(Function(input_names=['roi_dir'], output_names=['roi_files'], function=get_roi_files),
                    name='roi_node')
    roi_node.inputs.roi_dir = roi_dir

    # MapNode to extract values for each ROI
    roi_extract = MapNode(Function(input_names=['cope_file', 'roi_mask', 'baseline_file', 'output_dir'],
                                   output_names=['beta_file', 'psc_file'],
                                   function=extract_roi_values),
                          iterfield=['roi_mask'],
                          name='roi_extract')
    roi_extract.inputs.output_dir = os.path.join(output_dir, 'roi_temp')  # Temporary directory

    # Node to combine values into CSV
    roi_combine = Node(Function(input_names=['beta_files', 'psc_files', 'output_dir'],
                                output_names=['beta_csv', 'psc_csv'],
                                function=combine_roi_values),
                       name='roi_combine')
    roi_combine.inputs.output_dir = output_dir

    # Output node
    outputnode = Node(IdentityInterface(fields=['beta_csv', 'psc_csv']),
                      name='outputnode')

    # DataSink
    datasink = Node(DataSink(base_directory=output_dir), name='datasink')

    # Workflow connections
    wf.connect([
        # Inputs to roi_extract
        (inputnode, roi_extract, [('cope_file', 'cope_file'),
                                  ('baseline_file', 'baseline_file')]),
        (roi_node, roi_extract, [('roi_files', 'roi_mask')]),

        # Combine results
        (roi_extract, roi_combine, [('beta_file', 'beta_files'),
                                    ('psc_file', 'psc_files')]),

        # Outputs to outputnode
        (roi_combine, outputnode, [('beta_csv', 'beta_csv'),
                                   ('psc_csv', 'psc_csv')]),

        # Outputs to DataSink
        (outputnode, datasink, [('beta_csv', 'roi_results.@beta_csv'),
                                ('psc_csv', 'roi_results.@psc_csv')])
    ])

    return wf