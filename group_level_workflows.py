from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.fsl import Merge, FLIRT, ExtractROI, FLAMEO, Randomise, ImageMaths, SmoothEstimate, Threshold, Cluster
from nipype import DataSink
import os
import shutil
import glob
import subprocess
import pandas as pd
import numpy as np


def create_dummy_design_files(group_info, output_dir, use_guess=False):
    """
    Create design.mat, design.grp, and contrast.con for fMRI analysis based on
    the number of unique drug levels and the 'use_guess' parameter.

    - If len(unique_drugs) == 1: Two-sample t-test (Patients vs Controls).
    - If len(unique_drugs) >= 2 and use_guess is False: 2x2 ANOVA (Group x Drug).
    - If len(unique_drugs) >= 2 and use_guess is True: 2x2x2 ANOVA (Group x Drug x Guess).

    Assumes group_info is a list of tuples:
    (subject_id, group [0:controls, 1:patients], drug_use_level, [optional: guess])
    If use_guess is True, the fourth element of the tuple is expected to be the guess.
    """
    import os
    import numpy as np

    design_dir = os.path.join(output_dir, 'design_files')
    os.makedirs(design_dir, exist_ok=True)
    design_file = os.path.join(design_dir, 'design.mat')
    grp_file    = os.path.join(design_dir, 'design.grp')
    con_file    = os.path.join(design_dir, 'contrast.con')

    # Extract drug levels and optionally guesses
    drug_ids     = [item[2] for item in group_info]
    unique_drugs = sorted(set(drug_ids))
    n            = len(group_info)
    unique_guesses = []
    if use_guess:
        guess_ids = [item[3] for item in group_info]
        unique_guesses = sorted(set(guess_ids))

    if len(unique_drugs) == 1:
        # --- Two-sample test Patients vs Controls ---
        design_rows     = []
        variance_groups = []
        for _, grp, _ in group_info:
            if grp == 1:                # Patients
                design_rows.append("1 0")
            else:                       # Controls
                design_rows.append("0 1")
            variance_groups.append("1") # single variance group

        num_evs = 2
        contrasts = [
            "1 -1",  # Patients > Controls
            "1  0",  # Patients mean
            "0  1"   # Controls mean
        ]

        # write design.mat
        with open(design_file, 'w') as f:
            f.write(f"/NumWaves {num_evs}\n")
            f.write(f"/NumPoints {n}\n")
            f.write("/Matrix\n")
            f.write("\n".join(design_rows))

        with open(grp_file, 'w') as f:
            f.write("/NumWaves 1\n")
            f.write(f"/NumPoints {n}\n")
            f.write("/Matrix\n")
            f.write("\n".join(variance_groups))

        with open(con_file, 'w') as f:
            f.write(f"/NumWaves {num_evs}\n")
            f.write(f"/NumContrasts {len(contrasts)}\n")
            f.write("/Matrix\n")
            f.write("\n".join(contrasts))

    elif len(unique_drugs) >= 2 and not use_guess:
        # --- Full 2×2 ANOVA (Group × Drug) ---
        design_mat     = []
        variance_groups = []
        num_evs        = 4
        for _, grp, drug in group_info:
            row = [0, 0, 0, 0]
            if grp == 1:  # Patients
                if drug == unique_drugs[0]:
                    row[0] = 1; variance_groups.append("1")
                else:
                    row[1] = 1; variance_groups.append("2")
            else:         # Controls
                if drug == unique_drugs[0]:
                    row[2] = 1; variance_groups.append("3")
                else:
                    row[3] = 1; variance_groups.append("4")
            design_mat.append(" ".join(map(str, row)))

        contrasts = [
            "1  1 -1 -1",  # Group effect (Patients > Controls, averaged across drugs)
            "1 -1  1 -1",  # Drug effect (Drug1 > Drug2, averaged across groups)
            "1 -1 -1  1",  # Interaction (Group x Drug)
            "1  0  0  0",  # Patients & Drug1 mean
            "0  1  0  0",  # Patients & Drug2 mean
            "0  0  1  0",  # Controls & Drug1 mean
            "0  0  0  1",  # Controls & Drug2 mean
        ]

        # write design.mat
        with open(design_file, 'w') as f:
            f.write(f"/NumWaves {num_evs}\n")
            f.write(f"/NumPoints {n}\n")
            f.write("/Matrix\n")
            f.write("\n".join(design_mat))

        # write design.grp
        with open(grp_file, 'w') as f:
            f.write("/NumWaves 1\n")
            f.write(f"/NumPoints {n}\n")
            f.write("/Matrix\n")
            f.write("\n".join(variance_groups))

        # write contrast.con
        with open(con_file, 'w') as f:
            f.write(f"/NumWaves {num_evs}\n")
            f.write(f"/NumContrasts {len(contrasts)}\n")
            f.write("/Matrix\n")
            f.write("\n".join(contrasts))

    elif len(unique_drugs) >= 2 and use_guess:
        # --- Full 2×2×2 ANOVA (Group × Drug × Guess) ---
        if not unique_guesses:
            raise ValueError("Guess information is required in group_info when use_guess=True.")

        design_mat     = []
        variance_groups = []
        num_evs        = 2 * len(unique_drugs) * len(unique_guesses)
        ev_index_map   = {}
        current_index  = 0

        for grp_idx in range(2):  # 0: Controls, 1: Patients
            for drug_idx, drug in enumerate(unique_drugs):
                for guess_idx, guess in enumerate(unique_guesses):
                    ev_index_map[(grp_idx, drug, guess)] = current_index
                    current_index += 1

        for item in group_info:
            grp, drug, guess = item[1], item[2], item[3]
            row = [0] * num_evs
            if (grp, drug, guess) in ev_index_map:
                row[ev_index_map[(grp, drug, guess)]] = 1
                variance_groups.append(str(ev_index_map[(grp, drug, guess)] + 1))
            design_mat.append(" ".join(map(str, row)))

        contrasts = []
        # Main effect of Group
        contrast_group = [0] * num_evs
        for drug in unique_drugs:
            for guess in unique_guesses:
                if (1, drug, guess) in ev_index_map:
                    contrast_group[ev_index_map[(1, drug, guess)]] = 1 / (len(unique_drugs) * len(unique_guesses))
                if (0, drug, guess) in ev_index_map:
                    contrast_group[ev_index_map[(0, drug, guess)]] = -1 / (len(unique_drugs) * len(unique_guesses))
        contrasts.append("Group_Effect")
        contrasts.append(" ".join(map(str, contrast_group)))

        # Main effect of Drug (assuming 'use_drug' and 'placebo' are in unique_drugs)
        if 'use_drug' in unique_drugs and 'placebo' in unique_drugs:
            drug1_idx = unique_drugs.index('use_drug')
            drug2_idx = unique_drugs.index('placebo')
            contrast_drug = [0] * num_evs
            for grp_idx in range(2):
                for guess in unique_guesses:
                    if (grp_idx, unique_drugs[drug1_idx], guess) in ev_index_map:
                        contrast_drug[ev_index_map[(grp_idx, unique_drugs[drug1_idx], guess)]] = 1 / 2
                    if (grp_idx, unique_drugs[drug2_idx], guess) in ev_index_map:
                        contrast_drug[ev_index_map[(grp_idx, unique_drugs[drug2_idx], guess)]] = -1 / 2
            contrasts.append("Drug_Effect")
            contrasts.append(" ".join(map(str, contrast_drug)))

            # Interaction Group x Drug
            contrast_group_drug_interaction = [0] * num_evs
            for guess in unique_guesses:
                if (1, unique_drugs[drug1_idx], guess) in ev_index_map:
                    contrast_group_drug_interaction[ev_index_map[(1, unique_drugs[drug1_idx], guess)]] = 1 / len(unique_guesses)
                if (0, unique_drugs[drug1_idx], guess) in ev_index_map:
                    contrast_group_drug_interaction[ev_index_map[(0, unique_drugs[drug1_idx], guess)]] = -1 / len(unique_guesses)
                if (1, unique_drugs[drug2_idx], guess) in ev_index_map:
                    contrast_group_drug_interaction[ev_index_map[(1, unique_drugs[drug2_idx], guess)]] = -1 / len(unique_guesses)
                if (0, unique_drugs[drug2_idx], guess) in ev_index_map:
                    contrast_group_drug_interaction[ev_index_map[(0, unique_drugs[drug2_idx], guess)]] = 1 / len(unique_guesses)
            contrasts.append("Group_x_Drug")
            contrasts.append(" ".join(map(str, contrast_group_drug_interaction)))

            # Drug effect for guess == 'use'
            if 'use' in unique_guesses:
                guess_use_idx = unique_guesses.index('use')
                contrast_drug_guess_use = [0] * num_evs
                for grp_idx in range(2):
                    if (grp_idx, unique_drugs[drug1_idx], unique_guesses[guess_use_idx]) in ev_index_map:
                        contrast_drug_guess_use[ev_index_map[(grp_idx, unique_drugs[drug1_idx], unique_guesses[guess_use_idx])]] = 1
                    if (grp_idx, unique_drugs[drug2_idx], unique_guesses[guess_use_idx]) in ev_index_map:
                        contrast_drug_guess_use[ev_index_map[(grp_idx, unique_drugs[drug2_idx], unique_guesses[guess_use_idx])]] = -1
                contrasts.append("Drug_Effect_Guess_Use")
                contrasts.append(" ".join(map(str, contrast_drug_guess_use)))

            # Drug effect for guess == 'no'
            if 'no' in unique_guesses:
                guess_no_idx = unique_guesses.index('no')
                contrast_drug_guess_no = [0] * num_evs
                for grp_idx in range(2):
                    if (grp_idx, unique_drugs[drug1_idx], unique_guesses[guess_no_idx]) in ev_index_map:
                        contrast_drug_guess_no[ev_index_map[(grp_idx, unique_drugs[drug1_idx], unique_guesses[guess_no_idx])]] = 1
                    if (grp_idx, unique_drugs[drug2_idx], unique_guesses[guess_no_idx]) in ev_index_map:
                        contrast_drug_guess_no[ev_index_map[(grp_idx, unique_drugs[drug2_idx], unique_guesses[guess_no_idx])]] = -1
                contrasts.append("Drug_Effect_Guess_No")
                contrasts.append(" ".join(map(str, contrast_drug_guess_no)))

        # write design.mat
        with open(design_file, 'w') as f:
            f.write(f"/NumWaves {num_evs}\n")
            f.write(f"/NumPoints {n}\n")
            f.write("/Matrix\n")
            f.write("\n".join(design_mat))

        # write design.grp
        with open(grp_file, 'w') as f:
            f.write("/NumWaves 1\n")
            f.write(f"/NumPoints {n}\n")
            f.write("/Matrix\n")
            f.write("\n".join(variance_groups))

        # write contrast.con
        with open(con_file, 'w') as f:
            f.write(f"/NumWaves {num_evs}\n")
            f.write(f"/NumContrasts {len(contrasts) // 2}\n")
            f.write("/Matrix\n")
            for i in range(0, len(contrasts), 2):
                f.write(f"{contrasts[i+1]}\n")

    else:
        print("Warning: Less than two unique drug levels found when use_guess=False.")
        # You might want to implement a different design if only one drug level exists

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
    inputnode = Node(IdentityInterface(fields=['in_copes', 'in_varcopes', 'group_info', 'use_guess', 'result_dir', 'group_mask']),
                     name='inputnode')

    # Design generation
    design_gen = Node(Function(input_names=['group_info', 'output_dir','use_guess'],
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
        (inputnode, design_gen, [('use_guess', 'use_guess')]),
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
# Define a standalone function to flatten nested lists


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


def wf_flameo(output_dir, name="wf_flameo", use_covsplit=True):
    """Workflow for group-level analysis with FLAMEO and GRF clustering (no flatten_stats)."""
    from nipype.pipeline.engine import Workflow
    from nipype import Node, MapNode
    from nipype.interfaces.utility import IdentityInterface
    from nipype.interfaces.io import DataSink
    from nipype.interfaces.fsl.model import FLAMEO
    from nipype.interfaces.fsl.model import SmoothEstimate, Cluster

    wf = Workflow(name=name, base_dir=output_dir)

    # Input node
    inputnode = Node(
        IdentityInterface(fields=[
            'cope_file', 'var_cope_file', 'mask_file',
            'design_file', 'grp_file', 'con_file', 'result_dir'
        ]),
        name='inputnode'
    )

    # FLAMEO node
    flameo = Node(
        FLAMEO(run_mode='flame1'),
        name='flameo'
    )

    # Smoothness estimation node
    smoothness = MapNode(
        SmoothEstimate(),
        iterfield=['zstat_file'],
        name='smoothness'
    )

    # Clustering node with GRF
    clustering = MapNode(
        Cluster(
            threshold=2.3,
            connectivity=26,
            out_threshold_file=True,
            out_index_file=True,
            out_localmax_txt_file=True,
            pthreshold=0.05
        ),
        iterfield=['in_file', 'dlh', 'volume'],
        name='clustering'
    )

    # Output node
    outputnode = Node(
        IdentityInterface(fields=[
            'zstats', 'cluster_thresh', 'cluster_index', 'cluster_peaks'
        ]),
        name='outputnode'
    )

    # DataSink
    datasink = Node(
        DataSink(base_directory=output_dir),
        name='datasink'
    )

    # Workflow connections
    wf.connect([
        # FLAMEO inputs
        (inputnode, flameo, [
            ('cope_file',      'cope_file'),
            ('var_cope_file',  'var_cope_file'),  # 🔧 corrected mapping: var_cope_file trait
            ('mask_file',      'mask_file'),
            ('design_file',    'design_file'),
            ('grp_file',       'cov_split_file'),
            ('con_file',       't_con_file')
        ]),

        # Smoothness estimation
        (flameo,     smoothness, [('zstats', 'zstat_file')]),
        (inputnode,  smoothness, [('mask_file', 'mask_file')]),

        # Clustering
        (flameo,      clustering, [('zstats', 'in_file')]),
        (smoothness,  clustering, [('volume', 'volume')]),
        (smoothness,  clustering, [('dlh', 'dlh')]),

        # Collect outputs
        (flameo,      outputnode, [('zstats', 'zstats')]),
        (clustering,  outputnode, [
            ('threshold_file',     'cluster_thresh'),
            ('index_file',         'cluster_index'),
            ('localmax_txt_file',  'cluster_peaks')
        ]),

        # Send to DataSink
        (outputnode, datasink, [
            ('zstats',         'stats.@zstats'),
            ('cluster_thresh', 'cluster_results.@thresh'),
            ('cluster_index',  'cluster_results.@index'),
            ('cluster_peaks',  'cluster_results.@peaks')
        ])
    ])

    return wf


def wf_randomise(output_dir, name="wf_randomise"):
    """Workflow for group-level analysis with Randomise + TFCE."""
    from nipype.pipeline.engine import Workflow, Node, MapNode
    from nipype.interfaces.utility import IdentityInterface
    from nipype.interfaces.fsl.model import Randomise
    from nipype.interfaces.fsl.utils import ImageMaths      # unchanged import
    from nipype.interfaces.io import DataSink

    wf = Workflow(name=name, base_dir=output_dir)

    # 1) Inputs: cope, mask, design.mat, contrast.con
    inputnode = Node(
        IdentityInterface(fields=[
            'cope_file',    # in_file for Randomise
            'mask_file',    # mask for Randomise
            'design_file',  # design_mat for Randomise
            'con_file',     # tcon for Randomise
        ]),
        name='inputnode'
    )

    # 2) Randomise: TFCE + voxelwise p-values
    randomise = Node(
        Randomise(
            num_perm=10000,
            tfce=True,
            vox_p_values=True
        ),
        name='randomise'
    )

    # 3) Convert TFCE-corrected p’s → z for easy thresholding
    fdr_ztop = MapNode(
        ImageMaths(op_string='-ztop', suffix='_zstat'),
        iterfield=['in_file'],  # will receive real file paths
        name='fdr_ztop'
    )

    # 4) Collect everything
    outputnode = Node(
        IdentityInterface(fields=[
            'tstat_files',         # raw t-stats
            'tfce_corr_p_files',   # TFCE-corrected p’s
            'z_thresh_files',      # z-transformed p-stats
        ]),
        name='outputnode'
    )

    # 5) Sink to disk
    datasink = Node(DataSink(base_directory=output_dir), name='datasink')

    # --- Connections ---
    wf.connect([
        # a) feed inputs into Randomise
        (inputnode, randomise, [
            ('cope_file',   'in_file'),
            ('mask_file',   'mask'),
            ('design_file', 'design_mat'),
            ('con_file',    'tcon'),
        ]),

        # b) take TFCE-corrected p’s → z-scores
        (randomise, fdr_ztop, [
            ('t_corrected_p_files', 'in_file'),  # now passing full paths
        ]),

        # c) collect Randomise outputs
        (randomise, outputnode, [
            ('tstat_files',         'tstat_files'),
            ('t_corrected_p_files', 'tfce_corr_p_files'),
        ]),
        (fdr_ztop, outputnode, [
            ('out_file', 'z_thresh_files'),
        ]),

        # d) write out to disk
        (outputnode, datasink, [
            ('tstat_files',       'stats.@tstats'),
            ('tfce_corr_p_files', 'stats.@tfce_p'),
            ('z_thresh_files',    'stats.@zscores'),
        ]),
    ])

    return wf




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

def flatten_list(nested):
    """Flatten a nested list of files into a 1D list."""
    return [f for sub in nested for f in sub]



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