#Nipype v1.10.0.
from nipype.pipeline import engine as pe
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


def make_trial_info_lss(events_file, target_idx):
    """
    Create session_info for LSS: one EV for the target trial, one for all others.
    """
    import pandas as pd
    from nipype.interfaces.base import Bunch

    # Load events and ensure trial_idx exists
    df = pd.read_csv(events_file, sep='\t')
    if 'trial_idx' not in df.columns:
        df['trial_idx'] = range(1, len(df) + 1)

    # Grab the target trial
    row = df[df['trial_idx'] == target_idx]
    if row.empty:
        raise ValueError(f"Trial {target_idx} not found in events file.")

    onset, duration = row[['onset', 'duration']].iloc[0].astype(float)

    # All other trials
    others = df[df['trial_idx'] != target_idx]

    return Bunch(
        conditions=[f"t{target_idx}", "others"],
        onsets=[[onset], others['onset'].tolist()],
        durations=[[duration], others['duration'].tolist()],
        amplitudes=None
    )



def get_trial_idxs(events_file):
    """
    Return sorted unique trial indices from an events TSV file.
    """
    import pandas as pd
    df = pd.read_csv(events_file, sep='\t')
    if 'trial_idx' not in df.columns:
        df['trial_idx'] = range(1, len(df) + 1)
    return sorted(df['trial_idx'].unique())


def first_level_single_trial_LSS_wf(inputs, output_dir, hrf_model='dgamma'):
    wf = pe.Workflow('wf_single_trial_LSS')
    wf.config['execution']['use_relative_paths'] = True
    wf.config['execution']['remove_unnecessary_outputs'] = False

    # 1) Datasource: unpack per-subject inputs
    datasource = pe.Node(niu.Function(function=_dict_ds, output_names=DATA_ITEMS),
                         name='datasource')
    datasource.inputs.in_dict = inputs
    datasource.iterables = ('sub', sorted(inputs.keys()))

    # 2) Apply brain mask
    apply_mask = pe.Node(ApplyMask(), name='apply_mask')

    # 3) Extract confounds (motion etc.)
    runinfo = pe.Node(
        niu.Function(
            input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names'],
            output_names=['info', 'realign_file'],
            function=_bids2nipypeinfo
        ), name='runinfo')
    runinfo.inputs.regressors_names = [
        'dvars', 'framewise_displacement',
        *[f'a_comp_cor_{i:02d}' for i in range(6)],
        *[f'cosine{i:02d}' for i in range(4)]
    ]

    # 4) Enumerate trial indices
    trial_node = pe.Node(
        niu.Function(
            input_names=['events_file'],
            output_names=['trial_idx_list'],
            function=get_trial_idxs
        ), name='get_trial_idxs')

    # 5) Build per-trial session_info for LSS directly
    lss_info = pe.MapNode(
        niu.Function(
            input_names=['events_file', 'target_idx'],
            output_names=['trial_info'],
            function=make_trial_info_lss
        ),
        name='lss_info',
        iterfield=['target_idx']
    )

    # 6) SpecifyModel: create fsf and EV files per trial
    l1_spec = pe.MapNode(
        SpecifyModel(
            parameter_source='FSL',
            input_units='secs',
            high_pass_filter_cutoff=100
        ), name='l1_spec',
        iterfield=['subject_info']
    )

    # 7) Level1Design: generate design matrices
    l1_model = pe.MapNode(
        fsl.Level1Design(
            bases={hrf_model: {'derivs': True}},
            model_serial_correlations=True,
            contrasts=[('t', 'T', ['t'], [1.0])]
        ), name='l1_model',
        iterfield=['session_info']
    )

    # 8) FEATModel: produce design.mat and design.con
    feat_spec = pe.MapNode(
        fsl.FEATModel(), name='feat_spec',
        iterfield=['fsf_file', 'ev_files']
    )

    # 9) run FEAT
    feat_fit = pe.MapNode(
        FILMGLS(smooth_autocorr=True, mask_size=5), name='feat_fit',
        iterfield=['design_file', 'tcon_file', 'in_file']
    )

    # 10) Select and sink copes/varcopes
    feat_select = pe.Node(
        nio.SelectFiles({**{f'cope{i}': f'cope{i}.nii.gz' for i in range(1, 2)},
                     **{f'varcope{i}': f'varcope{i}.nii.gz' for i in range(1, 2)}}),
        name='feat_select'
    )
    ds_copes = [
        pe.Node(DerivativesDataSink(base_directory=output_dir, keep_dtype=False, desc=f'cope{i}'),
             name=f'ds_cope{i}', run_without_submitting=True)
        for i in range(1, 2)
    ]
    ds_varcopes = [
        pe.Node(DerivativesDataSink(base_directory=output_dir, keep_dtype=False, desc=f'varcope{i}'),
             name=f'ds_varcope{i}', run_without_submitting=True)
        for i in range(1, 2)
    ]

    # -- Connect nodes ---------------------------------------------------------
    wf.connect([
        (datasource, apply_mask, [('bold', 'in_file'), ('mask', 'mask_file')]),
        (datasource, runinfo, [('events', 'events_file'), ('regressors', 'regressors_file')]),
        (datasource, trial_node, [('events', 'events_file')]),
        (datasource, lss_info, [('events', 'events_file')]),
        (datasource, l1_spec, [('tr', 'time_repetition')]),
        (datasource, l1_model, [('tr', 'interscan_interval')]),

        (trial_node, lss_info, [('trial_idx_list', 'target_idx')]),

        (apply_mask, runinfo, [('out_file', 'in_file')]),
        (apply_mask, l1_spec, [('out_file', 'functional_runs')]),
        (apply_mask, feat_spec, [('out_file', 'in_file')]),

        (runinfo, l1_spec, [('realign_file', 'realignment_parameters')]),

        (lss_info, l1_spec, [('trial_info', 'subject_info')]),
        (lss_info, l1_model, [('trial_info', 'session_info')]),

        (l1_model, feat_spec, [('fsf_files', 'fsf_file'), ('ev_files', 'ev_files')]),

        (feat_spec, feat_fit, [('design_file', 'design_file'), ('con_file', 'tcon_file')]),
        (feat_fit, feat_select, [('results_dir', 'base_directory')]),
        *[(feat_select, ds_copes[i - 1], [(f'cope{i}', 'in_file')]) for i in range(1, 2)],
        *[(feat_select, ds_varcopes[i - 1], [(f'varcope{i}', 'in_file')]) for i in range(1, 2)],
    ])

    return wf
