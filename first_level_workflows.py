#Nipype v1.10.0.
from nipype.pipeline import engine as pe
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl, utility as niu, io as nio
from niworkflows.interfaces.bids import DerivativesDataSink as BIDSDerivatives
from utils import _dict_ds
from utils import _dict_ds_lss
from utils import _bids2nipypeinfo
from utils import _bids2nipypeinfo_lss
from nipype.interfaces.fsl import SUSAN, ApplyMask, FLIRT, FILMGLS, Level1Design, FEATModel
import logging

# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class DerivativesDataSink(BIDSDerivatives):
    """Custom data sink for first-level analysis outputs."""
    out_path_base = 'firstLevel'

DATA_ITEMS = ['bold', 'mask', 'events', 'regressors', 'tr']
DATA_ITEMS_LSS = ['bold', 'mask', 'events', 'regressors', 'tr', 'trial_ID']

# Available contrast patterns
CONTRAST_PATTERNS = {
    'all_vs_baseline': 'Each condition vs baseline',
    'pairwise': 'All pairwise comparisons between conditions',
    'first_vs_rest': 'First condition vs all others',
    'group_vs_group': 'First half vs second half of conditions',
    'linear_trend': 'Linear trend across conditions',
    'quadratic_trend': 'Quadratic trend across conditions',
    'specific_face_vs_house': 'Face vs house conditions (if present)',
    'direct_weights': 'Direct weight specification as list/tuple (e.g., [1, -1, 0])'
}

# Available contrast types
CONTRAST_TYPES = {
    'standard': 'Pairwise comparisons between all conditions',
    'minimal': 'Each condition vs baseline only',
    'custom': 'Use custom patterns defined in contrast_patterns'
}

# =============================================================================
# CONTRAST GENERATION FUNCTIONS
# =============================================================================

def create_contrasts(condition_names, contrast_type='standard'):
    """
    Create contrasts dynamically based on condition names.
    
    Args:
        condition_names (list): List of condition names
        contrast_type (str): Type of contrasts to create ('standard', 'minimal', 'custom')
    
    Returns:
        list: List of contrast tuples (name, type, conditions, weights)
    """
    if not condition_names:
        logger.warning("No condition names provided, returning empty contrasts list")
        return []
    
    contrasts = []
    
    if contrast_type == 'minimal':
        # Create simple contrasts for each condition vs baseline
        for condition in condition_names:
            contrasts.append((f'{condition}>baseline', 'T', [condition], [1]))
    
    elif contrast_type == 'standard':
        # Create pairwise contrasts between all conditions
        for i, cond1 in enumerate(condition_names):
            for j, cond2 in enumerate(condition_names):
                if i < j:  # Avoid duplicate contrasts
                    contrasts.append((f'{cond1}>{cond2}', 'T', [cond1, cond2], [1, -1]))
                    contrasts.append((f'{cond1}<{cond2}', 'T', [cond1, cond2], [-1, 1]))
    
    elif contrast_type == 'custom':
        # Define custom contrasts based on condition patterns
        if 'first_half' in str(condition_names) and 'second_half' in str(condition_names):
            # Split conditions by halves
            first_half_conds = [c for c in condition_names if 'first_half' in c]
            second_half_conds = [c for c in condition_names if 'second_half' in c]
            
            # Create contrasts between halves
            for f_cond in first_half_conds:
                base_cond = f_cond.replace('first_half', 'second_half')
                if base_cond in second_half_conds:
                    contrasts.append((f'{f_cond}>{base_cond}', 'T', [f_cond, base_cond], [1, -1]))
                    contrasts.append((f'{f_cond}<{base_cond}', 'T', [f_cond, base_cond], [-1, 1]))
    
    logger.info(f"Generated {len(contrasts)} contrasts for {len(condition_names)} conditions")
    return contrasts

def create_custom_contrasts(condition_names, contrast_patterns):
    """
    Create custom contrasts based on specific patterns.
    
    Args:
        condition_names (list): List of condition names
        contrast_patterns (list): List of contrast patterns or direct weight lists
    
    Returns:
        list: List of contrast tuples
    """
    if not condition_names:
        logger.warning("No condition names provided for custom contrasts")
        return []
    
    contrasts = []
    for pattern in contrast_patterns:
        if pattern == 'all_vs_baseline':
            for condition in condition_names:
                contrasts.append((f'{condition}>baseline', 'T', [condition], [1]))
        
        elif pattern == 'pairwise':
            for i, cond1 in enumerate(condition_names):
                for j, cond2 in enumerate(condition_names):
                    if i < j:
                        contrasts.append((f'{cond1}>{cond2}', 'T', [cond1, cond2], [1, -1]))
                        contrasts.append((f'{cond1}<{cond2}', 'T', [cond1, cond2], [-1, 1]))
        
        elif pattern == 'first_vs_rest':
            if len(condition_names) > 1:
                first_cond = condition_names[0]
                rest_conds = condition_names[1:]
                weights = [len(rest_conds)] + [-1] * len(rest_conds)
                contrasts.append((f'{first_cond}>rest', 'T', [first_cond] + rest_conds, weights))
        
        elif pattern == 'group_vs_group':
            # Split conditions into two groups
            mid = len(condition_names) // 2
            group1 = condition_names[:mid]
            group2 = condition_names[mid:]
            if group1 and group2:
                weights = [1/len(group1)] * len(group1) + [-1/len(group2)] * len(group2)
                contrasts.append(('group1>group2', 'T', group1 + group2, weights))
        
        elif pattern == 'linear_trend':
            # Create linear trend contrast
            n_conds = len(condition_names)
            if n_conds > 2:
                weights = [(i - (n_conds-1)/2) for i in range(n_conds)]
                contrasts.append(('linear_trend', 'T', condition_names, weights))
        
        elif pattern == 'quadratic_trend':
            # Create quadratic trend contrast
            n_conds = len(condition_names)
            if n_conds > 2:
                weights = [(i - (n_conds-1)/2)**2 for i in range(n_conds)]
                contrasts.append(('quadratic_trend', 'T', condition_names, weights))
        
        elif pattern.startswith('specific_'):
            # Handle specific contrast patterns
            if pattern == 'specific_face_vs_house':
                face_conds = [c for c in condition_names if 'face' in c.lower()]
                house_conds = [c for c in condition_names if 'house' in c.lower()]
                if face_conds and house_conds:
                    weights = [1/len(face_conds)] * len(face_conds) + [-1/len(house_conds)] * len(house_conds)
                    contrasts.append(('face>house', 'T', face_conds + house_conds, weights))
        
        elif isinstance(pattern, (list, tuple)) and len(pattern) == len(condition_names):
            # Direct weight specification: pattern is a list of weights for each condition
            weights = list(pattern)
            # Generate a descriptive name for the contrast
            contrast_name = 'custom_contrast'
            if any(w > 0 for w in weights) and any(w < 0 for w in weights):
                pos_conds = [condition_names[i] for i, w in enumerate(weights) if w > 0]
                neg_conds = [condition_names[i] for i, w in enumerate(weights) if w < 0]
                if pos_conds and neg_conds:
                    contrast_name = f"{'+'.join(pos_conds)}>{'+'.join(neg_conds)}"
            elif any(w > 0 for w in weights):
                pos_conds = [condition_names[i] for i, w in enumerate(weights) if w > 0]
                contrast_name = f"{'+'.join(pos_conds)}>baseline"
            elif any(w < 0 for w in weights):
                neg_conds = [condition_names[i] for i, w in enumerate(weights) if w < 0]
                contrast_name = f"baseline>{'+'.join(neg_conds)}"
            
            contrasts.append((contrast_name, 'T', condition_names, weights))
        
        else:
            logger.warning(f"Unknown contrast pattern or invalid weight list: {pattern}")
    
    logger.info(f"Generated {len(contrasts)} custom contrasts")
    return contrasts

# =============================================================================
# CORE WORKFLOW FUNCTIONS
# =============================================================================

def first_level_wf(in_files, output_dir, condition_names=None, contrasts=None, 
                   contrast_type='standard', contrast_patterns=None,
                   fwhm=6.0, brightness_threshold=1000, high_pass_cutoff=100,
                   use_smoothing=True, use_derivatives=True, model_serial_correlations=True):
    """
    Generic first-level workflow for fMRI analysis.
    
    Args:
        in_files (dict): Input files dictionary
        output_dir (str): Output directory path
        condition_names (list): List of condition names (auto-detected if None)
        contrasts (list): List of contrast tuples (auto-generated if None)
        contrast_type (str): Type of contrasts to auto-generate ('standard', 'minimal', 'custom')
        contrast_patterns (list): List of contrast patterns for custom generation
        fwhm (float): Smoothing FWHM
        brightness_threshold (float): SUSAN brightness threshold
        high_pass_cutoff (float): High-pass filter cutoff
        use_smoothing (bool): Whether to apply smoothing
        use_derivatives (bool): Whether to use temporal derivatives
        model_serial_correlations (bool): Whether to model serial correlations
    
    Returns:
        pe.Workflow: Configured first-level workflow
    """
    if not in_files:
        raise ValueError("in_files cannot be empty")
    
    workflow = pe.Workflow(name='wf_1st_level')
    workflow.config['execution']['use_relative_paths'] = True
    workflow.config['execution']['remove_unnecessary_outputs'] = False

    # Data source
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
    
    # Optional smoothing
    if use_smoothing:
        susan = pe.Node(SUSAN(), name='susan')
        susan.inputs.fwhm = fwhm
        susan.inputs.brightness_threshold = brightness_threshold
        preproc_output = susan
    else:
        preproc_output = apply_mask

    # Model specification
    l1_spec = pe.Node(SpecifyModel(
        parameter_source='FSL',
        input_units='secs',
        high_pass_filter_cutoff=high_pass_cutoff
    ), name='l1_spec')

    # Auto-generate contrasts if not provided
    if contrasts is None:
        if condition_names is None:
            # Default condition names - can be overridden
            condition_names = ['condition1', 'condition2', 'condition3']
        
        if contrast_type == 'custom' and contrast_patterns:
            contrasts = create_custom_contrasts(condition_names, contrast_patterns)
        else:
            contrasts = create_contrasts(condition_names, contrast_type=contrast_type)
    
    if not contrasts:
        logger.warning("No contrasts generated, workflow may fail")
    
    logger.info(f"Using {len(contrasts)} contrasts: {[c[0] for c in contrasts]}")

    # Level 1 model design
    l1_model = pe.Node(Level1Design(
        bases={'dgamma': {'derivs': use_derivatives}},
        model_serial_correlations=model_serial_correlations,
        contrasts=contrasts
    ), name='l1_model')

    # FEAT model specification
    feat_spec = pe.Node(FEATModel(), name='feat_spec')
    
    # FEAT fitting
    feat_fit = pe.Node(FILMGLS(smooth_autocorr=True, mask_size=5), name='feat_fit', mem_gb=12)
    
    # Select output files
    n_contrasts = len(contrasts)
    feat_select = pe.Node(nio.SelectFiles({
        **{f'cope{i}': f'cope{i}.nii.gz' for i in range(1, n_contrasts + 1)},
        **{f'varcope{i}': f'varcope{i}.nii.gz' for i in range(1, n_contrasts + 1)}
    }), name='feat_select')

    # Data sinks for copes and varcopes
    ds_copes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False, desc=f'cope{i}'),
            name=f'ds_cope{i}', run_without_submitting=True)
        for i in range(1, n_contrasts + 1)
    ]

    ds_varcopes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False, desc=f'varcope{i}'),
            name=f'ds_varcope{i}', run_without_submitting=True)
        for i in range(1, n_contrasts + 1)
    ]

    # Build workflow connections
    connections = _build_workflow_connections(
        datasource, apply_mask, runinfo, l1_spec, l1_model, 
        feat_spec, feat_fit, feat_select, preproc_output, use_smoothing
    )
    
    # Add data sink connections
    for i in range(1, n_contrasts + 1):
        connections.extend([
            (datasource, ds_copes[i - 1], [('bold', 'source_file')]),
            (datasource, ds_varcopes[i - 1], [('bold', 'source_file')]),
            (feat_select, ds_copes[i - 1], [(f'cope{i}', 'in_file')]),
            (feat_select, ds_varcopes[i - 1], [(f'varcope{i}', 'in_file')])
        ])

    workflow.connect(connections)
    return workflow

def first_level_wf_LSS(in_files, output_dir, trial_ID, condition_names=None, contrasts=None,
                       contrast_type='minimal', contrast_patterns=None,
                       fwhm=6.0, brightness_threshold=1000, high_pass_cutoff=100,
                       use_smoothing=False, use_derivatives=True, model_serial_correlations=True):
    """
    Generic LSS (Least Squares Separate) first-level workflow.
    
    Note: LSS analysis is recommended to be run WITHOUT smoothing to preserve
    fine-grained temporal information and avoid blurring trial-specific responses.
    
    Args:
        in_files (dict): Input files dictionary
        output_dir (str): Output directory path
        trial_ID (int): Trial ID for LSS analysis
        condition_names (list): List of condition names (auto-detected if None)
        contrasts (list): List of contrast tuples (auto-generated if None)
        contrast_type (str): Type of contrasts to auto-generate ('minimal', 'standard', 'custom')
        contrast_patterns (list): List of contrast patterns for custom generation
        fwhm (float): Smoothing FWHM (not recommended for LSS)
        brightness_threshold (float): SUSAN brightness threshold (not used if use_smoothing=False)
        high_pass_cutoff (float): High-pass filter cutoff
        use_smoothing (bool): Whether to apply smoothing (default: False for LSS)
        use_derivatives (bool): Whether to use temporal derivatives
        model_serial_correlations (bool): Whether to model serial correlations
    
    Returns:
        pe.Workflow: Configured LSS workflow
    """
    if not in_files:
        raise ValueError("in_files cannot be empty")
    
    workflow = pe.Workflow(name='wf_1st_level_LSS')
    workflow.config['execution']['use_relative_paths'] = True
    workflow.config['execution']['remove_unnecessary_outputs'] = False

    datasource = pe.Node(niu.Function(function=_dict_ds_lss, output_names=DATA_ITEMS_LSS),
                         name='datasource')
    datasource.inputs.in_dict = in_files
    datasource.iterables = ('sub', sorted(in_files.keys()))

    # Extract motion parameters from regressors file
    runinfo = pe.Node(niu.Function(
        input_names=['in_file', 'events_file', 'regressors_file',
                     'trial_ID', 'regressors_names', 'motion_columns',
                     'decimals', 'amplitude'],
        output_names=['info', 'realign_file'],
        function=_bids2nipypeinfo_lss),
        name='runinfo')

    # Set the column names to be used from the confounds file
    runinfo.inputs.regressors_names = ['dvars', 'framewise_displacement'] + \
                                      ['a_comp_cor_%02d' % i for i in range(6)] + \
                                      ['cosine%02d' % i for i in range(4)]

    # Mask
    apply_mask = pe.Node(ApplyMask(), name='apply_mask')

    # Model specification
    l1_spec = pe.Node(SpecifyModel(
        parameter_source='FSL',
        input_units='secs',
        high_pass_filter_cutoff=high_pass_cutoff
    ), name='l1_spec')
    
    # Note: LSS typically does not use smoothing to preserve temporal precision
    if use_smoothing:
        logger.warning("Smoothing is enabled for LSS analysis. This is not recommended as it may blur trial-specific responses.")

    # Auto-generate contrasts if not provided
    if contrasts is None:
        if condition_names is None:
            # Default LSS contrasts
            condition_names = ['trial', 'others']
        
        if contrast_type == 'custom' and contrast_patterns:
            contrasts = create_custom_contrasts(condition_names, contrast_patterns)
        else:
            contrasts = create_contrasts(condition_names, contrast_type=contrast_type)
    
    if not contrasts:
        logger.warning("No contrasts generated for LSS workflow")
    
    logger.info(f"LSS using {len(contrasts)} contrasts: {[c[0] for c in contrasts]}")

    # Level 1 model design
    l1_model = pe.Node(Level1Design(
        bases={'dgamma': {'derivs': use_derivatives}},
        model_serial_correlations=model_serial_correlations,
        contrasts=contrasts
    ), name='l1_model')

    # FEAT model specification
    feat_spec = pe.Node(FEATModel(), name='feat_spec')
    
    # FEAT fitting
    feat_fit = pe.Node(FILMGLS(smooth_autocorr=True, mask_size=5), name='feat_fit', mem_gb=12)
    
    # Select output files
    n_contrasts = len(contrasts)
    feat_select = pe.Node(nio.SelectFiles({
        **{f'cope{i}': f'cope{i}.nii.gz' for i in range(1, n_contrasts + 1)},
        **{f'varcope{i}': f'varcope{i}.nii.gz' for i in range(1, n_contrasts + 1)}
    }), name='feat_select')

    # Data sinks for copes and varcopes
    ds_copes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False,
            desc=f'trial{int(trial_ID)}_cope{i}'),
            name=f'ds_cope{i}',
            run_without_submitting=True)
        for i in range(1, n_contrasts + 1)
    ]

    ds_varcopes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False,
            desc=f'trial{int(trial_ID)}_varcope{i}'),
            name=f'ds_varcope{i}',
            run_without_submitting=True)
        for i in range(1, n_contrasts + 1)
    ]

    # Workflow connections
    connections = [
        (datasource, apply_mask, [('bold', 'in_file'), ('mask', 'mask_file')]),
        (datasource, runinfo, [('events', 'events_file'), ('trial_ID', 'trial_ID'), ('regressors', 'regressors_file')]),
        (datasource, l1_spec, [('tr', 'time_repetition')]),
        (datasource, l1_model, [('tr', 'interscan_interval')]),
        (apply_mask, l1_spec, [('out_file', 'functional_runs')]),
        (apply_mask, runinfo, [('out_file', 'in_file')]),
        (runinfo, l1_spec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')]),
        (l1_spec, l1_model, [('session_info', 'session_info')]),
        (l1_model, feat_spec, [('fsf_files', 'fsf_file'), ('ev_files', 'ev_files')]),
        (feat_spec, feat_fit, [('design_file', 'design_file'), ('con_file', 'tcon_file')]),
        (apply_mask, feat_fit, [('out_file', 'in_file')]),
        (feat_fit, feat_select, [('results_dir', 'base_directory')]),
    ]
    
    # Add data sink connections
    for i in range(1, n_contrasts + 1):
        connections.extend([
            (datasource, ds_copes[i - 1], [('bold', 'source_file')]),
            (datasource, ds_varcopes[i - 1], [('bold', 'source_file')]),
            (feat_select, ds_copes[i - 1], [(f'cope{i}', 'in_file')]),
            (feat_select, ds_varcopes[i - 1], [(f'varcope{i}', 'in_file')])
        ])

    workflow.connect(connections)
    return workflow

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_workflow_connections(datasource, apply_mask, runinfo, l1_spec, l1_model, 
                              feat_spec, feat_fit, feat_select, preproc_output, use_smoothing):
    """
    Build workflow connections based on smoothing configuration.
    
    Args:
        datasource: Data source node
        apply_mask: Mask application node
        runinfo: Run info node
        l1_spec: Level 1 specification node
        l1_model: Level 1 model node
        feat_spec: FEAT specification node
        feat_fit: FEAT fitting node
        feat_select: FEAT selection node
        preproc_output: Preprocessing output node
        use_smoothing: Whether smoothing is used
    
    Returns:
        list: List of workflow connections
    """
    connections = [
        (datasource, apply_mask, [('bold', 'in_file'), ('mask', 'mask_file')]),
        (datasource, runinfo, [('events', 'events_file'), ('regressors', 'regressors_file')]),
        (datasource, l1_spec, [('tr', 'time_repetition')]),
        (datasource, l1_model, [('tr', 'interscan_interval')]),
        (l1_spec, l1_model, [('session_info', 'session_info')]),
        (l1_model, feat_spec, [('fsf_files', 'fsf_file'), ('ev_files', 'ev_files')]),
        (feat_spec, feat_fit, [('design_file', 'design_file'), ('con_file', 'tcon_file')]),
        (feat_fit, feat_select, [('results_dir', 'base_directory')]),
    ]
    
    # Add smoothing connections if used
    if use_smoothing:
        connections.extend([
            (apply_mask, preproc_output, [('out_file', 'in_file')]),
            (preproc_output, l1_spec, [('smoothed_file', 'functional_runs')]),
            (preproc_output, runinfo, [('smoothed_file', 'in_file')]),
            (preproc_output, feat_fit, [('smoothed_file', 'in_file')])
        ])
    else:
        connections.extend([
            (apply_mask, l1_spec, [('out_file', 'functional_runs')]),
            (apply_mask, runinfo, [('out_file', 'in_file')]),
            (apply_mask, feat_fit, [('out_file', 'in_file')])
        ])
    
    # Add runinfo connections
    connections.extend([
        (runinfo, l1_spec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')])
    ])
    
    return connections

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_standard_contrasts():
    """Create standard contrasts for typical fMRI analysis."""
    condition_names = ['SHOCK', 'FIXATION', 'CS-', 'CSS', 'CSR']
    return create_contrasts(condition_names, contrast_type='standard')

def create_lss_contrasts():
    """Create standard LSS contrasts."""
    condition_names = ['trial', 'others']
    return create_contrasts(condition_names, contrast_type='minimal')

def create_face_house_contrasts():
    """Create contrasts for face vs house experiment."""
    condition_names = ['face', 'house', 'object']
    return create_contrasts(condition_names, contrast_type='standard')

def create_emotion_contrasts():
    """Create contrasts for emotion experiment."""
    condition_names = ['happy', 'sad', 'angry', 'neutral']
    return create_contrasts(condition_names, contrast_type='standard')

def create_working_memory_contrasts():
    """Create contrasts for working memory experiment."""
    condition_names = ['load1', 'load2', 'load3', 'load4']
    patterns = ['all_vs_baseline', 'linear_trend', 'quadratic_trend']
    return create_custom_contrasts(condition_names, patterns)

# =============================================================================
# BEST PRACTICES AND DOCUMENTATION
# =============================================================================

def create_lss_workflow_best_practices():
    """
    LSS (Least Squares Separate) Analysis Best Practices:
    
    1. NO SMOOTHING: LSS should typically be run without spatial smoothing
       to preserve fine-grained temporal information and avoid blurring
       trial-specific responses.
    
    2. TEMPORAL PRECISION: LSS is designed to capture trial-by-trial
       variability, so temporal precision is crucial.
    
    3. MINIMAL CONTRASTS: Usually only need trial vs baseline or trial vs others.
    
    4. HIGH-PASS FILTERING: Use appropriate high-pass filtering to remove
       low-frequency drifts while preserving trial-specific signals.
    
    Example:
        wf = first_level_wf_LSS(
            in_files=files, output_dir='output', trial_ID=1,
            use_smoothing=False,  # CRITICAL for LSS
            high_pass_cutoff=100,  # Adjust based on your design
            use_derivatives=False  # Often not needed for LSS
        )
    """
    pass

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_usage():
    """Examples of how to use the flexible contrast settings in the workflows."""
    
    print("=== Basic Usage ===")
    wf1 = first_level_wf(
        in_files={'sub-01': {'bold': 'bold.nii.gz', 'mask': 'mask.nii.gz'}},
        output_dir='output',
        condition_names=['face', 'house', 'object']
    )
    
    print("=== Custom Contrasts ===")
    wf2 = first_level_wf(
        in_files={'sub-01': {'bold': 'bold.nii.gz', 'mask': 'mask.nii.gz'}},
        output_dir='output',
        condition_names=['load1', 'load2', 'load3', 'load4'],
        contrast_type='custom',
        contrast_patterns=['all_vs_baseline', 'linear_trend']
    )
    
    print("=== LSS Analysis (No Smoothing) ===")
    wf3 = first_level_wf_LSS(
        in_files={'sub-01': {'bold': 'bold.nii.gz', 'mask': 'mask.nii.gz'}},
        output_dir='output',
        trial_ID=1,
        use_smoothing=False  # LSS best practice
    )
    
    print("=== Direct Weight Specification ===")
    wf4 = first_level_wf(
        in_files={'sub-01': {'bold': 'bold.nii.gz', 'mask': 'mask.nii.gz'}},
        output_dir='output',
        condition_names=['face', 'house', 'object'],
        contrast_type='custom',
        contrast_patterns=[[1, -1, 0], [1, 0, -1], [0, 1, -1]]
    )
    
    return [wf1, wf2, wf3, wf4]
