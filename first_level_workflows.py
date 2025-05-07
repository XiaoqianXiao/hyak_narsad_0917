from nipype.pipeline import engine as pe
from nipype.pipeline.engine import Workflow, Node
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl, utility as niu, io as nio
from niworkflows.interfaces.bids import DerivativesDataSink as BIDSDerivatives
from utils import _dict_ds
from utils import _bids2nipypeinfo
from nipype.interfaces.fsl import SUSAN, ApplyMask, FLIRT, FILMGLS, Level1Design
import os
import pandas as pd
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.base import Bunch


class DerivativesDataSink(BIDSDerivatives):
    out_path_base = 'firstLevel'


DATA_ITEMS = ['bold', 'mask', 'events', 'regressors', 'tr']


def first_level_wf(in_files, output_dir, fwhm=6.0, brightness_threshold=1000):
    workflow = pe.Workflow(name='wf_1st_level')
    workflow.config['execution']['use_relative_paths'] = True
    workflow.config['execution']['remove_unnecessary_outputs'] = False

    datasource = pe.Node(niu.Function(function=_dict_ds, output_names=DATA_ITEMS),
                         name='datasource')
    datasource.inputs.in_dict = in_files
    datasource.iterables = ('sub', sorted(in_files.keys()))

    # Extract motion parameters from regressors file
    runinfo = pe.Node(niu.Function(
        input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names'],
        function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
        name='runinfo')

    # Set the column names to be used from the confounds file
    runinfo.inputs.regressors_names = ['dvars', 'framewise_displacement'] + \
                                      ['a_comp_cor_%02d' % i for i in range(6)] + \
                                      ['cosine%02d' % i for i in range(4)]

    # Mask
    apply_mask = pe.Node(ApplyMask(), name='apply_mask')
    # SUSAN smoothing
    susan = pe.Node(SUSAN(), name='susan')
    susan.inputs.fwhm = fwhm
    susan.inputs.brightness_threshold = brightness_threshold

    l1_spec = pe.Node(SpecifyModel(
        parameter_source='FSL',
        input_units='secs',
        high_pass_filter_cutoff=100
    ), name='l1_spec')

    # l1_model creates a first-level model design
    l1_model = pe.Node(fsl.Level1Design(
        bases={'dgamma': {'derivs': True}},
        model_serial_correlations=True,
        contrasts=[('CS+_safe>CS-', 'T', ['CSS_first_half', 'CSS_second_half', 'CS-_first_half', 'CS-_second_half'],
                    [0.5, 0.5, -0.5, -0.5]),
                   ('CS+_safe<CS-', 'T', ['CSS_first_half', 'CSS_second_half', 'CS-_first_half', 'CS-_second_half'],
                    [-0.5, -0.5, 0.5, 0.5]),
                   ('CS+_reinf>CS-', 'T', ['CSR_first_half', 'CSR_second_half', 'CS-_first_half', 'CS-_second_half'],
                    [0.5, 0.5, -0.5, -0.5]),
                   ('CS+_reinf<CS-', 'T', ['CSR_first_half', 'CSR_second_half', 'CS-_first_half', 'CS-_second_half'],
                    [-0.5, -0.5, 0.5, 0.5]),
                   ('CS+_safe>CS+_reinf', 'T',
                    ['CSS_first_half', 'CSS_second_half', 'CSR_first_half', 'CSR_second_half'], [0.5, 0.5, -0.5, -0.5]),
                   ('CS+_safe<CS+_reinf', 'T',
                    ['CSS_first_half', 'CSS_second_half', 'CSR_first_half', 'CSR_second_half'], [-0.5, -0.5, 0.5, 0.5]),
                   ('CS->FIXATION', 'T',
                    ['CS-_first_half', 'CS-_second_half', 'FIXATION_first_half', 'FIXATION_second_half'],
                    [0.5, 0.5, -0.5, -0.5]),
                   ('CS+_safe>FIXATION', 'T',
                    ['CSS_first_half', 'CSS_second_half', 'FIXATION_first_half', 'FIXATION_second_half'],
                    [0.5, 0.5, -0.5, -0.5]),
                   ('CS+_reinf>FIXATION', 'T',
                    ['CSR_first_half', 'CSR_second_half', 'FIXATION_first_half', 'FIXATION_second_half'],
                    [0.5, 0.5, -0.5, -0.5]),
                   ('first_half_CS+_safe>first_half_CS+_reinf', 'T', ['CSS_first_half', 'CSR_first_half'], [1, -1]),
                   ('first_half_CS+_safe<first_half_CS+_reinf', 'T', ['CSS_first_half', 'CSR_first_half'], [-1, 1]),
                   ('first_half_CS+_safe>CS-', 'T', ['CSS_first_half', 'CS-_first_half'], [1, -1]),
                   ('first_half_CS+_safe<CS-', 'T', ['CSS_first_half', 'CS-_first_half'], [-1, 1]),
                   ('first_half_CS+_reinf>CS-', 'T', ['CSR_first_half', 'CS-_first_half'], [1, -1]),
                   ('first_half_CS+_reinf<CS-', 'T', ['CSR_first_half', 'CS-_first_half'], [-1, 1]),
                   ('first_half_CS+_safe>FIXATION', 'T', ['CSS_first_half', 'FIXATION_first_half'], [1, -1]),
                   ('first_half_CS+_reinf>FIXATION', 'T', ['CSR_first_half', 'FIXATION_first_half'], [1, -1]),
                   ('second_half_CS+_safe>second_half_CS+_reinf', 'T', ['CSS_second_half', 'CSR_second_half'], [1, -1]),
                   ('second_half_CS+_safe<second_half_CS+_reinf', 'T', ['CSS_second_half', 'CSR_second_half'], [-1, 1]),
                   ('second_half_CS+_safe>CS-', 'T', ['CSS_second_half', 'CS-_second_half'], [1, -1]),
                   ('second_half_CS+_safe<CS-', 'T', ['CSS_second_half', 'CS-_second_half'], [-1, 1]),
                   ('second_half_CS+_reinf>CS-', 'T', ['CSR_second_half', 'CS-_second_half'], [1, -1]),
                   ('second_half_CS+_reinf<CS-', 'T', ['CSR_second_half', 'CS-_second_half'], [-1, 1]),
                   ('second_half_CS+_safe>FIXATION', 'T', ['CSS_second_half', 'FIXATION_second_half'], [1, -1]),
                   ('second_half_CS+_reinf>FIXATION', 'T', ['CSR_second_half', 'FIXATION_second_half'], [1, -1]),
                   ('first_half_CS+_safe>second_half_CS+_safe', 'T', ['CSS_first_half', 'CSS_second_half'], [1, -1]),
                   ('first_half_CS+_safe<second_half_CS+_safe', 'T', ['CSS_first_half', 'CSS_second_half'], [-1, 1]),
                   ('first_half_CS+_reinf>second_half_CS+_reinf', 'T', ['CSR_first_half', 'CSR_second_half'], [1, -1]),
                   ('first_half_CS+_reinf<second_half_CS+_reinf', 'T', ['CSR_first_half', 'CSR_second_half'], [-1, 1]),
                   ('early>later_CS+_safe>CS+_reinf', 'T',
                    ['CSS_first_half', 'CSS_second_half', 'CSR_first_half', 'CSR_second_half'], [0.5, -0.5, -0.5, 0.5]),
                   ('early<later_CS+_safe>CS+_reinf', 'T',
                    ['CSS_first_half', 'CSS_second_half', 'CSR_first_half', 'CSR_second_half'], [-0.5, 0.5, 0.5, -0.5])
                   ],
    ), name='l1_model')

    # feat_spec generates an fsf model specification file
    feat_spec = pe.Node(fsl.FEATModel(), name='feat_spec')
    # feat_fit actually runs FEAT
    feat_fit = pe.Node(fsl.FILMGLS(smooth_autocorr=True, mask_size=5), name='feat_fit', mem_gb=12)
    feat_select = pe.Node(nio.SelectFiles({
        **{f'cope{i}': f'cope{i}.nii.gz' for i in range(1, 32)},
        **{f'varcope{i}': f'varcope{i}.nii.gz' for i in range(1, 32)}
    }), name='feat_select')

    ds_copes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False, desc=f'cope{i}'),
            name=f'ds_cope{i}', run_without_submitting=True)
        for i in range(1, 32)
    ]

    ds_varcopes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False, desc=f'varcope{i}'),
            name=f'ds_varcope{i}', run_without_submitting=True)
        for i in range(1, 32)
    ]

    workflow.connect([
        (datasource, apply_mask, [('bold', 'in_file'),
                                  ('mask', 'mask_file')]),
        (apply_mask, susan, [('out_file', 'in_file')]),
        (datasource, runinfo, [
            ('events', 'events_file'),
            ('regressors', 'regressors_file')]),
        *[
            (datasource, ds_copes[i - 1], [('bold', 'source_file')])
            for i in range(1, 32)
        ],
        *[
            (datasource, ds_varcopes[i - 1], [('bold', 'source_file')])
            for i in range(1, 32)
        ],
        (susan, l1_spec, [('smoothed_file', 'functional_runs')]),
        (datasource, l1_spec, [('tr', 'time_repetition')]),
        (datasource, l1_model, [('tr', 'interscan_interval')]),
        (susan, runinfo, [('smoothed_file', 'in_file')]),
        (runinfo, l1_spec, [
            ('info', 'subject_info'),
            ('realign_file', 'realignment_parameters')]),
        (l1_spec, l1_model, [('session_info', 'session_info')]),
        (l1_model, feat_spec, [
            ('fsf_files', 'fsf_file'),
            ('ev_files', 'ev_files')]),
        # --- Corrected connections for FILMGLS ---
        (feat_spec, feat_fit, [
            ('design_file', 'design_file'),
            ('con_file', 'tcon_file')]),
        (susan, feat_fit, [('smoothed_file', 'in_file')]),
        (feat_fit, feat_select, [('results_dir', 'base_directory')]),
        *[
            (feat_select, ds_copes[i - 1], [(f'cope{i}', 'in_file')])
            for i in range(1, 32)
        ],
        *[
            (feat_select, ds_varcopes[i - 1], [(f'varcope{i}', 'in_file')])
            for i in range(1, 32)
        ],
    ])
    return workflow


def make_session_info_lsa(events_df):
    """
    Build session info with one regressor per trial (LSA),
    naming each EV purely by its trial_idx, e.g. "t1", "t2", ...
    """
    conds, onsets, durations = [], [], []
    for _, row in events_df.iterrows():
        tid = int(row['trial_idx'])
        conds.append(f"t{tid}")
        onsets.append([row['onset']])
        durations.append([row['duration']])
    return Bunch(
        conditions=conds,
        onsets=onsets,
        durations=durations,
        amplitudes=None
    )

def make_session_info_lss(events_df, target_idx):
    """
    Build session info for LSS:
      - one EV "t<target_idx>" for the target trial
      - one EV "others" for all the rest
    """
    # select target
    mask_target = (events_df['trial_idx'] == target_idx)
    if not mask_target.any():
        raise ValueError(f"Trial index {target_idx} not found in events_df")

    onset_t    = float(events_df.loc[mask_target, 'onset'].iloc[0])
    duration_t = float(events_df.loc[mask_target, 'duration'].iloc[0])
    others_df  = events_df.loc[~mask_target]

    return Bunch(
        conditions=['t' + str(target_idx), 'others'],
        onsets=[[onset_t], others_df['onset'].tolist()],
        durations=[[duration_t], others_df['duration'].tolist()],
        amplitudes=None
    )



def estimate_single_trial(func_img, mask_img, events_file, t_r, hrf_model, method, trial_idx, out_base):
    """
    Estimate beta map for a single trial (LSA or LSS) via FILMGLS.
    Returns path to stats directory.
    """
    import os, pandas as pd, numpy as np
    from first_level_workflows import make_session_info_lsa, make_session_info_lss
    from nipype.interfaces.base import Bunch

    events_df = pd.read_csv(events_file, sep='\t')
    events_df.columns = events_df.columns.str.lower()
    if 'trial_idx' not in events_df.columns:
        events_df['trial_idx'] = range(1, len(events_df) + 1)
    # Prepare session info and condition name
    if method == 'LSA':
        sess = make_session_info_lsa(events_df)
        cond_name = f"t{int(trial_idx)}"
        sess_info = sess
        prefix = f"LSA_trial_{trial_idx:03d}"
    else:
        sess_info = make_session_info_lss(events_df, trial_idx)
        cond_name = f"t{int(trial_idx)}"
        prefix = f"LSS_trial_{trial_idx:03d}"

    # Create design
    design_dir = os.path.join(out_base, prefix, 'design')
    os.makedirs(design_dir, exist_ok=True)
    level1 = Level1Design(
        interscan_interval=t_r,
        bases={hrf_model: {'derivs': False}},
        session_info=[sess_info],
        mask_image=mask_img,
        model_serial_correlations=True,
        film_threshold=1000,
        run_mode='fe',
        contrast_info=[(f"beta_{cond_name}", 'T', [cond_name], [1.0])],
        output_dir=design_dir
    )
    res1 = level1.run()

    # Run FILMGLS
    mat_file = res1.outputs.design_mat
    con_file = res1.outputs.design_con
    stats_dir = os.path.join(out_base, prefix, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    film = FILMGLS(
        in_file=func_img,
        design_file=mat_file,
        contrast_file=con_file,
        mask_file=mask_img,
        out_dir_file=stats_dir
    )
    film.run()
    return stats_dir


def first_level_single_trial_wf(name='single_trial_wf'):
    """Builds Nipype Workflow to run single-trial LSA & LSS GLM estimations."""
    wf = Workflow(name=name)

    # Input spec
    inputnode = Node(
        IdentityInterface(fields=[
            'func_img', 'mask_img', 'events_file',
            't_r', 'hrf_model', 'method', 'trial_idx', 'out_base'
        ]),
        name='inputnode'
    )

    # Trial estimation node
    est_node = Node(
        Function(
            input_names=[
                'func_img', 'mask_img', 'events_file',
                't_r', 'hrf_model', 'method', 'trial_idx', 'out_base'
            ],
            output_names=['stats_dir'],
            function=estimate_single_trial
        ),
        name='estimate_single_trial'
    )

    # Connect nodes
    wf.connect([
        (inputnode, est_node, [
            ('func_img', 'func_img'),
            ('mask_img', 'mask_img'),
            ('events_file', 'events_file'),
            ('t_r', 't_r'),
            ('hrf_model', 'hrf_model'),
            ('method', 'method'),
            ('trial_idx', 'trial_idx'),
            ('out_base', 'out_base')
        ])
    ])

    return wf
