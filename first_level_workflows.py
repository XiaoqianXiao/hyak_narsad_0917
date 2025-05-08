from nipype.pipeline import engine as pe
from nipype.pipeline.engine import Workflow, Node, MapNode
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
    Build LSA session info:  one regressor per trial, named 't1','t2',...
    Uses only trial_idx and onset/duration.
    """
    import numpy as np
    # lowercase columns
    events_df.columns = events_df.columns.str.lower()
    # inject trial_idx if missing
    if 'trial_idx' not in events_df.columns:
        events_df = events_df.reset_index(drop=True)
        events_df['trial_idx'] = np.arange(len(events_df)) + 1

    conds     = [f"t{int(i)}" for i in events_df['trial_idx']]
    onsets    = [[float(x)] for x in events_df['onset']]
    durations = [[float(x)] for x in events_df['duration']]
    return Bunch(conditions=conds,
                 onsets=onsets,
                 durations=durations,
                 amplitudes=None)


def make_session_info_lss(events_df, target_idx):
    """
    Build LSS session info:  one EV 't<target_idx>' for the target trial,
    one EV 'others' for all the rest.
    """
    import numpy as np
    events_df.columns = events_df.columns.str.lower()
    if 'trial_idx' not in events_df.columns:
        events_df = events_df.reset_index(drop=True)
        events_df['trial_idx'] = np.arange(len(events_df)) + 1

    mask_t = events_df['trial_idx'] == target_idx
    if not mask_t.any():
        raise ValueError(f"Trial {target_idx} not found")
    # target
    onset_t    = float(events_df.loc[mask_t,   'onset'].iloc[0])
    duration_t = float(events_df.loc[mask_t,  'duration'].iloc[0])
    # others
    others     = events_df.loc[~mask_t]
    return Bunch(
        conditions=[f"t{target_idx}", "others"],
        onsets=[[onset_t],     others['onset'].tolist()],
        durations=[[duration_t], others['duration'].tolist()],
        amplitudes=None
    )


def estimate_single_trial(func_img, mask_img, events_file,
                          t_r, hrf_model, method, trial_idx, out_base):
    """
    Compute design & FILMGLS for a single trial via LSA or LSS.
    """
    import os
    import pandas as pd
    import numpy as np
    from first_level_workflows import make_session_info_lsa, make_session_info_lss
    from nipype.interfaces.fsl import Level1Design, FILMGLS
    # read events (assume comma‐delimited)
    events_df = pd.read_csv(events_file, sep='\t')
    # pick session info
    if method == 'LSA':
        sess_info = make_session_info_lsa(events_df)
        prefix    = f"LSA_trial_{int(trial_idx):03d}"
    else:
        sess_info = make_session_info_lss(events_df, trial_idx)
        prefix    = f"LSS_trial_{int(trial_idx):03d}"

    # write design
    design_dir = os.path.join(out_base, prefix, 'design')
    os.makedirs(design_dir, exist_ok=True)
    l1 = Level1Design(
        interscan_interval=t_r,
        bases={hrf_model: {'derivs': True}},
        session_info=[sess_info],
        mask_file=mask_img,
        model_serial_correlations=True,
        film_threshold=1000,
        run_mode='fe',
        contrast_info=[(f"beta_t{int(trial_idx)}", 'T', [f"t{int(trial_idx)}"], [1.0])],
        output_dir=design_dir
    )
    res = l1.run()

    # run FILMGLS
    stats_dir = os.path.join(out_base, prefix, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    film = FILMGLS(
        in_file=func_img,
        design_file=res.outputs.design_mat,
        contrast_file=res.outputs.design_con,
        mask_file=mask_img,
        out_dir_file=stats_dir
    )
    film.run()
    return stats_dir


def first_level_single_trial_wf(name='single_trial_wf'):
    wf = Workflow(name=name)

    # inputs
    inputnode = Node(IdentityInterface(fields=[
        'func_img','mask_img','events_file','t_r','hrf_model','trial_idx','out_base'
    ]), name='inputnode')

    # MapNode for **LSA** (only iterating trial_idx)
    est_LSA = MapNode(
        Function(
            input_names=['func_img','mask_img','events_file','t_r','hrf_model','method','trial_idx','out_base'],
            output_names=['stats_dir'],
            function=estimate_single_trial
        ),
        iterfield=['trial_idx'],
        name='est_LSA'
    )
    est_LSA.inputs.method = 'LSA'

    # MapNode for **LSS**
    est_LSS = MapNode(
        Function(
            input_names=['func_img','mask_img','events_file','t_r','hrf_model','method','trial_idx','out_base'],
            output_names=['stats_dir'],
            function=estimate_single_trial
        ),
        iterfield=['trial_idx'],
        name='est_LSS'
    )
    est_LSS.inputs.method = 'LSS'

    # connect both
    wf.connect([
        (inputnode, est_LSA, [
            ('func_img','func_img'), ('mask_img','mask_img'),
            ('events_file','events_file'), ('t_r','t_r'),
            ('hrf_model','hrf_model'), ('trial_idx','trial_idx'),
            ('out_base','out_base')
        ]),
        (inputnode, est_LSS, [
            ('func_img','func_img'), ('mask_img','mask_img'),
            ('events_file','events_file'), ('t_r','t_r'),
            ('hrf_model','hrf_model'), ('trial_idx','trial_idx'),
            ('out_base','out_base')
        ]),
    ])

    return wf
