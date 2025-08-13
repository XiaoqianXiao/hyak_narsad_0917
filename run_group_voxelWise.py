#!/usr/bin/env python3
"""
Unified group-level fMRI analysis pipeline for NARSAD project.

This script runs group-level statistical analysis using either FLAMEO or Randomise
on pre-processed first-level data. It supports different analysis types and data sources.

Usage:
    # Standard analysis
    python run_group_level.py --task phase2 --contrast 1 --analysis-type randomise --base-dir /path/to/data
    
    # Placebo-only analysis
    python run_group_level.py --task phase2 --contrast 1 --analysis-type flameo --base-dir /path/to/data --data-source placebo
    
    # Analysis with guess condition
    python run_group_level.py --task phase2 --contrast 1 --analysis-type randomise --base-dir /path/to/data --data-source guess
    
    # Custom data paths
    python run_group_level.py --task phase2 --contrast 1 --analysis-type flameo --base-dir /path/to/data --custom-paths
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import argparse
import logging
from pathlib import Path
from group_level_workflows import wf_randomise, wf_flameo
from nipype import config, logging as nipype_logging
from templateflow.api import get as tpl_get

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# NIPYPE CONFIGURATION
# =============================================================================

# Nipype plugin settings
PLUGIN_SETTINGS = {
    'plugin': 'MultiProc',
    'plugin_args': {
        'n_procs': 4,
        'raise_insufficient': False,
        'maxtasksperchild': 1,
    }
}

config.set('execution', 'remove_unnecessary_outputs', 'false')
nipype_logging.update_logging(config)

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# Use environment variables for data paths
ROOT_DIR = os.getenv('DATA_DIR', '/data')
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
SCRUBBED_DIR = '/scrubbed_dir'

# =============================================================================
# DATA SOURCE CONFIGURATIONS
# =============================================================================

DATA_SOURCE_CONFIGS = {
    'standard': {
        'description': 'Standard analysis with all subjects',
        'results_subdir': 'groupLevel',
        'workflows_subdir': 'groupLevel',
        'requires_varcope': True,
        'requires_grp': True
    },
    'placebo': {
        'description': 'Placebo condition only analysis',
        'results_subdir': 'groupLevel/Placebo',
        'workflows_subdir': 'groupLevel/Placebo',
        'requires_varcope': True,
        'requires_grp': True
    },
    'guess': {
        'description': 'Analysis including guess condition',
        'results_subdir': 'groupLevel',
        'workflows_subdir': 'groupLevel',
        'requires_varcope': True,
        'requires_grp': True
    }
}

# =============================================================================
# WORKFLOW EXECUTION FUNCTIONS
# =============================================================================

def run_group_level_workflow(task, contrast, analysis_type, paths, data_source_config):
    """
    Run group-level workflow for a specific task and contrast.
    
    Args:
        task (str): Task name (e.g., 'phase2', 'phase3')
        contrast (int): Contrast number
        analysis_type (str): Analysis type ('randomise' or 'flameo')
        paths (dict): Dictionary containing all necessary file paths
        data_source_config (dict): Configuration for the data source
    """
    try:
        # Select workflow function based on analysis type
        wf_func = wf_randomise if analysis_type == 'randomise' else wf_flameo
        wf_name = f"wf_{analysis_type}_{task}_cope{contrast}"
        
        logger.info(f"Creating workflow: {wf_name}")
        logger.info(f"Analysis type: {analysis_type}")
        logger.info(f"Data source: {data_source_config['description']}")
        
        # Create workflow
        wf = wf_func(output_dir=paths['result_dir'], name=wf_name)
        wf.base_dir = paths['workflow_dir']
        
        # Set common inputs
        wf.inputs.inputnode.cope_file = paths['cope_file']
        wf.inputs.inputnode.mask_file = paths['mask_file']
        wf.inputs.inputnode.design_file = paths['design_file']
        wf.inputs.inputnode.con_file = paths['con_file']
        wf.inputs.inputnode.result_dir = paths['result_dir']
        
        # Set FLAMEO-specific inputs if needed
        if analysis_type == 'flameo':
            if data_source_config['requires_varcope'] and 'varcope_file' in paths:
                wf.inputs.inputnode.var_cope_file = paths['varcope_file']
                logger.info("Set varcope file for FLAMEO analysis")
            else:
                logger.warning("Varcope file not found but required for FLAMEO analysis")
            
            if data_source_config['requires_grp'] and 'grp_file' in paths:
                wf.inputs.inputnode.grp_file = paths['grp_file']
                logger.info("Set group file for FLAMEO analysis")
            else:
                logger.warning("Group file not found but required for FLAMEO analysis")
        
        # Create directories
        Path(paths['result_dir']).mkdir(parents=True, exist_ok=True)
        Path(paths['workflow_dir']).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running workflow: {wf_name}")
        logger.info(f"Results directory: {paths['result_dir']}")
        logger.info(f"Workflow directory: {paths['workflow_dir']}")
        
        # Run the workflow
        wf.run(**PLUGIN_SETTINGS)
        
        logger.info(f"Workflow completed successfully: {wf_name}")
        
    except Exception as e:
        logger.error(f"Failed to run workflow {wf_name}: {e}")
        raise

def get_standard_paths(task, contrast, base_dir, data_source):
    """
    Get standard file paths for group-level analysis.
    
    Args:
        task (str): Task name
        contrast (int): Contrast number
        base_dir (str): Base directory for data
        data_source (str): Data source type
    
    Returns:
        dict: Dictionary containing all necessary file paths
    """
    # Get data source configuration
    data_source_config = DATA_SOURCE_CONFIGS.get(data_source, DATA_SOURCE_CONFIGS['standard'])
    
    # Set up directories
    results_dir = os.path.join(base_dir, data_source_config['results_subdir'])
    workflows_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows', data_source_config['workflows_subdir'])
    
    # Use TemplateFlow to get group mask path
    group_mask = str(tpl_get('MNI152NLin2009cAsym', resolution=2, desc='brain', suffix='mask'))
    
    # Define paths
    result_dir = os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'whole_brain')
    workflow_dir = os.path.join(workflows_dir, f'task-{task}', f'cope{contrast}', 'whole_brain')
    
    paths = {
        'result_dir': result_dir,
        'workflow_dir': workflow_dir,
        'cope_file': os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'merged_cope.nii.gz'),
        'varcope_file': os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'merged_varcope.nii.gz'),
        'design_file': os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'design_files', 'design.mat'),
        'con_file': os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'design_files', 'contrast.con'),
        'grp_file': os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'design_files', 'design.grp'),
        'mask_file': group_mask
    }
    
    return paths, data_source_config

def get_custom_paths(task, contrast, base_dir, custom_paths_dict):
    """
    Get custom file paths specified by the user.
    
    Args:
        task (str): Task name
        contrast (int): Contrast number
        base_dir (str): Base directory for data
        custom_paths_dict (dict): Dictionary with custom file paths
    
    Returns:
        dict: Dictionary containing all necessary file paths
    """
    # Use TemplateFlow to get group mask path
    group_mask = str(tpl_get('MNI152NLin2009cAsym', resolution=2, desc='brain', suffix='mask'))
    
    # Set default paths if not provided
    default_result_dir = os.path.join(base_dir, f'task-{task}', f'cope{contrast}', 'whole_brain')
    default_workflow_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows', 'groupLevel', f'task-{task}', f'cope{contrast}', 'whole_brain')
    
    paths = {
        'result_dir': custom_paths_dict.get('result_dir', default_result_dir),
        'workflow_dir': custom_paths_dict.get('workflow_dir', default_workflow_dir),
        'cope_file': custom_paths_dict.get('cope_file', os.path.join(base_dir, f'task-{task}', f'cope{contrast}', 'merged_cope.nii.gz')),
        'varcope_file': custom_paths_dict.get('varcope_file', os.path.join(base_dir, f'task-{task}', f'cope{contrast}', 'merged_varcope.nii.gz')),
        'design_file': custom_paths_dict.get('design_file', os.path.join(base_dir, f'task-{task}', f'cope{contrast}', 'design_files', 'design.mat')),
        'con_file': custom_paths_dict.get('con_file', os.path.join(base_dir, f'task-{task}', f'cope{contrast}', 'design_files', 'contrast.con')),
        'grp_file': custom_paths_dict.get('grp_file', os.path.join(base_dir, f'task-{task}', f'cope{contrast}', 'design_files', 'design.grp')),
        'mask_file': custom_paths_dict.get('mask_file', group_mask)
    }
    
    # Create a default data source config for custom paths
    data_source_config = {
        'description': 'Custom analysis with user-specified paths',
        'requires_varcope': True,
        'requires_grp': True
    }
    
    return paths, data_source_config

def validate_paths(paths, analysis_type):
    """
    Validate that all required files exist.
    
    Args:
        paths (dict): Dictionary containing file paths
        analysis_type (str): Analysis type ('randomise' or 'flameo')
    
    Returns:
        bool: True if all required files exist, False otherwise
    """
    required_files = ['cope_file', 'mask_file', 'design_file', 'con_file']
    
    # Add FLAMEO-specific requirements
    if analysis_type == 'flameo':
        required_files.extend(['varcope_file', 'grp_file'])
    
    missing_files = []
    for file_key in required_files:
        file_path = paths.get(file_key)
        if not file_path or not os.path.exists(file_path):
            missing_files.append(f"{file_key}: {file_path}")
    
    if missing_files:
        logger.error(f"Missing required files for {analysis_type} analysis:")
        for missing in missing_files:
            logger.error(f"  {missing}")
        return False
    
    logger.info("All required files found")
    return True

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Unified group-level fMRI analysis pipeline for NARSAD project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard analysis
  python run_group_level.py --task phase2 --contrast 1 --analysis-type randomise --base-dir /path/to/data
  
  # Placebo-only analysis
  python run_group_level.py --task phase2 --contrast 1 --analysis-type flameo --base-dir /path/to/data --data-source placebo
  
  # Analysis with guess condition
  python run_group_level.py --task phase2 --contrast 1 --analysis-type randomise --base-dir /path/to/data --data-source guess
  
  # Custom data paths
  python run_group_level.py --task phase2 --contrast 1 --analysis-type flameo --base-dir /path/to/data --custom-paths
        """
    )
    
    # Required arguments
    parser.add_argument('--task', required=True, help='Task name (e.g., phase2, phase3)')
    parser.add_argument('--contrast', required=True, type=int, help='Contrast number')
    parser.add_argument('--base-dir', required=True, help='Base directory containing the data')
    
    # Optional arguments
    parser.add_argument('--analysis-type', default='randomise', choices=['randomise', 'flameo'],
                       help='Analysis type: randomise (non-parametric) or flameo (parametric)')
    parser.add_argument('--data-source', default='standard', choices=['standard', 'placebo', 'guess'],
                       help='Data source type (default: standard)')
    parser.add_argument('--custom-paths', action='store_true',
                       help='Use custom file paths instead of standard structure')
    
    # Custom path arguments
    parser.add_argument('--cope-file', help='Custom path to cope file')
    parser.add_argument('--varcope-file', help='Custom path to varcope file')
    parser.add_argument('--design-file', help='Custom path to design matrix file')
    parser.add_argument('--con-file', help='Custom path to contrast file')
    parser.add_argument('--grp-file', help='Custom path to group file')
    parser.add_argument('--mask-file', help='Custom path to mask file')
    parser.add_argument('--result-dir', help='Custom result directory')
    parser.add_argument('--workflow-dir', help='Custom workflow directory')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting unified group-level analysis pipeline")
        logger.info(f"Task: {args.task}")
        logger.info(f"Contrast: {args.contrast}")
        logger.info(f"Analysis type: {args.analysis_type}")
        logger.info(f"Data source: {args.data_source}")
        logger.info(f"Base directory: {args.base_dir}")
        
        # Get file paths
        if args.custom_paths:
            # Use custom paths
            custom_paths = {
                'cope_file': args.cope_file,
                'varcope_file': args.varcope_file,
                'design_file': args.design_file,
                'con_file': args.con_file,
                'grp_file': args.grp_file,
                'mask_file': args.mask_file,
                'result_dir': args.result_dir,
                'workflow_dir': args.workflow_dir
            }
            paths, data_source_config = get_custom_paths(args.task, args.contrast, args.base_dir, custom_paths)
        else:
            # Use standard paths
            paths, data_source_config = get_standard_paths(args.task, args.contrast, args.base_dir, args.data_source)
        
        # Validate paths
        if not validate_paths(paths, args.analysis_type):
            logger.error("Path validation failed. Exiting.")
            return 1
        
        # Run the workflow
        run_group_level_workflow(args.task, args.contrast, args.analysis_type, paths, data_source_config)
        
        logger.info("Group-level analysis pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Group-level analysis pipeline failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
