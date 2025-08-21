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

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
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
        'results_subdir': 'groupLevel/whole_brain',
        'workflows_subdir': 'groupLevel/whole_brain',
        'requires_varcope': True,
        'requires_grp': True
    },
    'placebo': {
        'description': 'Placebo condition only analysis',
        'results_subdir': 'groupLevel/whole_brain/Placebo',
        'workflows_subdir': 'groupLevel/whole_brain/Placebo',
        'requires_varcope': True,
        'requires_grp': True
    },
    'guess': {
        'description': 'Analysis including guess condition',
        'results_subdir': 'groupLevel/whole_brain/Guess',
        'workflows_subdir': 'groupLevel/whole_brain/Guess',
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
        
        # Create workflow with proper directory alignment
        # IMPORTANT: Set output_dir to workflow_dir so DataSink writes to writable location
        # We'll copy results to final results directory after completion
        wf = wf_func(output_dir=paths['workflow_dir'], name=wf_name)
        wf.base_dir = paths['workflow_dir']
        
        # Set crash directory to workflow directory to avoid permission issues
        wf.config['execution']['crash_dir'] = paths['workflow_dir']
        
        # The DataSink will now write to the workflow directory (which is writable)
        # After completion, we'll copy results to the final results directory
        
        # Set common inputs
        wf.inputs.inputnode.cope_file = paths['cope_file']
        wf.inputs.inputnode.mask_file = paths['mask_file']
        wf.inputs.inputnode.design_file = paths['design_file']
        wf.inputs.inputnode.con_file = paths['con_file']
        
        # Set FLAMEO-specific inputs if needed
        if analysis_type == 'flameo':
            if data_source_config['requires_varcope'] and 'varcope_file' in paths:
                wf.inputs.inputnode.var_cope_file = paths['varcope_file']
                logger.info("Set varcope file for FLAMEO analysis")
            else:
                logger.error("Varcope file not found but required for FLAMEO analysis")
                raise ValueError("Varcope file required for FLAMEO analysis")
            
            if data_source_config['requires_grp'] and 'grp_file' in paths:
                wf.inputs.inputnode.grp_file = paths['grp_file']
                logger.info("Set group file for FLAMEO analysis")
            else:
                logger.error("Group file not found but required for FLAMEO analysis")
                raise ValueError("Group file required for FLAMEO analysis")
        
        # Create directories FIRST with better error handling
        logger.info(f"Creating results directory: {paths['result_dir']}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Results directory parent exists: {os.path.exists(os.path.dirname(paths['result_dir']))}")
        try:
            Path(paths['result_dir']).mkdir(parents=True, exist_ok=True)
            logger.info(f"Results directory created/verified: {paths['result_dir']}")
        except Exception as e:
            logger.error(f"Failed to create results directory: {paths['result_dir']} - {e}")
            raise
            
        logger.info(f"Creating workflow directory: {paths['workflow_dir']}")
        try:
            Path(paths['workflow_dir']).mkdir(parents=True, exist_ok=True)
            logger.info(f"Workflow directory created/verified: {paths['workflow_dir']}")
        except Exception as e:
            logger.error(f"Failed to create workflow directory: {paths['workflow_dir']} - {e}")
            raise
        
        # Verify directories actually exist
        if not os.path.exists(paths['result_dir']):
            logger.error(f"Results directory does not exist after creation: {paths['result_dir']}")
            raise RuntimeError(f"Failed to create results directory: {paths['result_dir']}")
        else:
            logger.info(f"Results directory exists: {paths['result_dir']}")
            
        if not os.path.exists(paths['workflow_dir']):
            logger.error(f"Workflow directory does not exist after creation: {paths['workflow_dir']}")
            raise RuntimeError(f"Failed to create workflow directory: {paths['workflow_dir']}")
        else:
            logger.info(f"Workflow directory exists: {paths['workflow_dir']}")
        
        # Verify directories are writable AFTER creation
        try:
            test_file = os.path.join(paths['result_dir'], 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Results directory is writable: {paths['result_dir']}")
        except Exception as e:
            logger.error(f"Results directory is not writable: {paths['result_dir']} - {e}")
            # Don't fail here, just log the warning and continue
            logger.warning(f"Results directory write test failed, but continuing...")
            
        try:
            test_file = os.path.join(paths['workflow_dir'], 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Workflow directory is writable: {paths['workflow_dir']}")
        except Exception as e:
            logger.error(f"Workflow directory is not writable: {paths['workflow_dir']} - {e}")
            # Don't fail here, just log the warning and continue
            logger.warning(f"Workflow directory write test failed, but continuing...")
        
        logger.info(f"Running workflow: {wf_name}")
        logger.info(f"Results directory: {paths['result_dir']}")
        logger.info(f"Workflow directory: {paths['workflow_dir']}")
        
        # Run the workflow
        logger.info(f"Starting workflow execution with plugin settings: {PLUGIN_SETTINGS}")
        try:
            result = wf.run(**PLUGIN_SETTINGS)
            logger.info(f"Workflow completed successfully: {wf_name}")
            logger.info(f"Workflow result: {result}")
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Check if there are crash files
            crash_files = glob.glob(os.path.join(paths['workflow_dir'], 'crash-*.pklz'))
            if crash_files:
                logger.error(f"Found crash files: {crash_files}")
            raise
        
        # Check if workflow actually completed by looking at the result
        if result is None:
            logger.warning("Workflow result is None - this might indicate an issue")
        else:
            logger.info(f"Workflow nodes completed: {list(result.keys()) if hasattr(result, 'keys') else 'No keys'}")
        
        # Copy results from workflow directory to final results directory
        workflow_output_dir = os.path.join(paths['workflow_dir'], wf_name)
        logger.info(f"Looking for workflow output in: {workflow_output_dir}")
        
        # Check what's in the main workflow directory (where actual results are)
        workflow_dir_contents = os.listdir(paths['workflow_dir'])
        logger.info(f"Main workflow directory contains: {workflow_dir_contents}")
        
        if os.path.exists(workflow_output_dir):
            logger.info(f"Copying results from workflow directory: {paths['workflow_dir']}")
            logger.info(f"To final results directory: {paths['result_dir']}")
            
            # List what's in the workflow output directory for reference
            workflow_contents = os.listdir(workflow_output_dir)
            logger.info(f"Workflow output directory contains: {workflow_contents}")
            
            # Check for specific clustering results
            if 'cluster_results' in workflow_contents:
                cluster_dir = os.path.join(workflow_output_dir, 'cluster_results')
                if os.path.exists(cluster_dir):
                    cluster_files = os.listdir(cluster_dir)
                    logger.info(f"Cluster results directory contains: {cluster_files}")
                else:
                    logger.warning("Cluster results directory not found")
            else:
                logger.warning("No cluster_results directory found in workflow output")
            
            # Check for stats directory
            if 'stats' in workflow_contents:
                stats_dir = os.path.join(workflow_output_dir, 'stats')
                if os.path.exists(stats_dir):
                    stats_files = os.listdir(stats_dir)
                    logger.info(f"Stats directory contains: {stats_files}")
                else:
                    logger.warning("Stats directory not found")
            else:
                logger.warning("No stats directory found in workflow output")
            
            # Check for any other subdirectories that might contain results
            for item in workflow_contents:
                item_path = os.path.join(workflow_output_dir, item)
                if os.path.isdir(item_path):
                    subdir_contents = os.listdir(item_path)
                    logger.info(f"Subdirectory '{item}' contains: {subdir_contents}")
            
            # Create final results directory if it doesn't exist
            Path(paths['result_dir']).mkdir(parents=True, exist_ok=True)
            
            # Copy all files from workflow directory to final results
            import shutil
            try:
                logger.info(f"About to copy from {paths['workflow_dir']} to {paths['result_dir']}")
                
                if os.path.exists(paths['result_dir']):
                    logger.info(f"Removing existing results directory: {paths['result_dir']}")
                    # Remove existing results directory contents
                    shutil.rmtree(paths['result_dir'])
                
                # Copy entire workflow directory (where actual results are)
                logger.info(f"Executing: shutil.copytree({paths['workflow_dir']}, {paths['result_dir']})")
                shutil.copytree(paths['workflow_dir'], paths['result_dir'])
                logger.info(f"Successfully copied results to: {paths['result_dir']}")
                
                # List final results
                if os.path.exists(paths['result_dir']):
                    result_files = os.listdir(paths['result_dir'])
                    logger.info(f"Final results directory contains: {result_files}")
                    
                    # Check if clustering results made it to the final directory
                    final_cluster_dir = os.path.join(paths['result_dir'], 'cluster_results')
                    if os.path.exists(final_cluster_dir):
                        final_cluster_files = os.listdir(final_cluster_dir)
                        logger.info(f"Final cluster_results directory contains: {final_cluster_files}")
                    else:
                        logger.warning("Final cluster_results directory not found after copy")
                else:
                    logger.warning(f"Final results directory does not exist after copy")
                    
            except Exception as e:
                logger.error(f"Failed to copy results: {e}")
                logger.error(f"Exception type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        else:
            logger.warning(f"Workflow output directory not found: {workflow_output_dir}")
            logger.info(f"Checking workflow directory contents: {os.listdir(paths['workflow_dir'])}")
        
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
    
    # Define paths - whole_brain is already included in results_subdir
    result_dir = os.path.join(results_dir, f'task-{task}', f'cope{contrast}')
    workflow_dir = os.path.join(workflows_dir, f'task-{task}', f'cope{contrast}')
    
    paths = {
        'result_dir': result_dir,
        'workflow_dir': workflow_dir,
        # Pre-group results are still in old structure: groupLevel/task-phaseX/copeY/
        'cope_file': os.path.join(base_dir, data_source_config['results_subdir'], f'task-{task}', f'cope{contrast}', 'merged_cope.nii.gz'),
        'varcope_file': os.path.join(base_dir, data_source_config['results_subdir'], f'task-{task}', f'cope{contrast}', 'merged_varcope.nii.gz'),
        'design_file': os.path.join(base_dir, data_source_config['results_subdir'], f'task-{task}', f'cope{contrast}', 'design_files', 'design.mat'),
        'con_file': os.path.join(base_dir, data_source_config['results_subdir'], f'task-{task}', f'cope{contrast}', 'design_files', 'contrast.con'),
        'grp_file': os.path.join(base_dir, data_source_config['results_subdir'], f'task-{task}', f'cope{contrast}', 'design_files', 'design.grp'),
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
    
    # Set default paths if not provided - whole_brain moved right after groupLevel
    default_result_dir = os.path.join(base_dir, 'whole_brain', f'task-{task}', f'cope{contrast}')
    default_workflow_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows', 'groupLevel', 'whole_brain', f'task-{task}', f'cope{contrast}')
    
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
