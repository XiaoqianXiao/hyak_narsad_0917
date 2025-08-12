import os
import argparse
from itertools import combinations
import logging
from pathlib import Path

# Configure logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logging()

def generate_slurm_scripts(method, work_dir, slurm_dir, container_path, script_path):
    """
    Generate Slurm scripts for group-level searchlight analysis.
    
    Args:
        method (str): Analysis method ('flameo' or 'randomise')
        work_dir (str): Working directory
        slurm_dir (str): Directory to save Slurm scripts
        container_path (str): Path to the container image
        script_path (str): Path to the group_searchlight.py script
    """
    # Trial types
    trial_types = ['SHOCK', 'FIXATION', 'CS-', 'CSS', 'CSR']
    map_types = [f'within-{ttype}' for ttype in trial_types] + [f'between-{t1}-{t2}' for t1, t2 in combinations(trial_types, 2)]
    tasks = ['phase2', 'phase3']
    
    logger.info(f"Generating Slurm scripts for {len(map_types)} map types and {len(tasks)} tasks with method {method}")
    logger.info(f"Map types: {map_types}")

    # Set time limit based on method (fixed format)
    if method == 'flameo':
        time_limit = '02:00:00'  # 2 hours for FLAMEO
        mem_limit = '20G'
        cpus_per_task = 16
    else:  # randomise
        time_limit = '04:00:00'  # 4 hours for Randomise (longer due to permutations)
        mem_limit = '32G'        # More memory for Randomise
        cpus_per_task = 8        # Fewer CPUs for Randomise (I/O bound)

    # Slurm template with configurable paths
    slurm_template = f"""#!/bin/bash
#SBATCH --account=fang                                                                                            
#SBATCH --partition=ckpt-all  
#SBATCH --job-name=group_searchlight_{{map_type}}_{{task}}_{{method}}
#SBATCH --output={work_dir}/{{task}}_group_searchlight_{{map_type}}_{{method}}_%j.out
#SBATCH --error={work_dir}/{{task}}_group_searchlight_{{map_type}}_{{method}}_%j.err
#SBATCH --time={{time}}
#SBATCH --mem={{mem}}
#SBATCH --cpus-per-task={{cpus}}

module load apptainer
export OMP_NUM_THREADS=4

# Create output directory if it doesn't exist
mkdir -p {work_dir}

apptainer exec \\
    -B /gscratch/fang:/data \\
    -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir \\
    -B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad/group_searchlight.py:/app/group_searchlight.py \\
    {container_path} python3 /app/group_searchlight.py --map_type {{map_type}} --method {{method}}
"""

    # Generate Slurm script for each map type and task
    script_paths = []
    for map_type in map_types:
        for task in tasks:
            try:
                script_path = os.path.join(slurm_dir, f'group_searchlight_{map_type}_{task}_slurm.sh')
                
                # Format template with current values
                script_content = slurm_template.format(
                    map_type=map_type, 
                    task=task, 
                    method=method, 
                    time=time_limit,
                    mem=mem_limit,
                    cpus=cpus_per_task
                )
                
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                os.chmod(script_path, 0o755)  # Make executable
                script_paths.append(script_path)
                logger.info(f"Generated Slurm script: {script_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate script for {map_type}_{task}: {e}")
                continue
    
    logger.info(f"Successfully generated {len(script_paths)} Slurm scripts")
    return script_paths

def main():
    """Main function to generate Slurm scripts for group-level searchlight analysis."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate Slurm scripts for group-level searchlight analysis.')
    parser.add_argument('--method', choices=['flameo', 'randomise'], default='flameo', 
                       help='Analysis method: flameo or randomise')
    parser.add_argument('--container-path', 
                       default='/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif',
                       help='Path to the container image')
    parser.add_argument('--script-path', 
                       default='/gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad/group_searchlight.py',
                       help='Path to the group_searchlight.py script')
    parser.add_argument('--work-dir', 
                       default='/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_group_searchlight',
                       help='Working directory for outputs and logs')
    
    args = parser.parse_args()
    method = args.method

    try:
        # Create work_dir and slurm_dir
        work_dir = args.work_dir
        slurm_dir = os.path.join(work_dir, method)
        
        # Ensure directories exist
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(slurm_dir, exist_ok=True)
        
        logger.info(f"Working directory: {work_dir}")
        logger.info(f"Slurm scripts directory: {slurm_dir}")
        logger.info(f"Container path: {args.container_path}")
        logger.info(f"Script path: {args.script_path}")

        # Validate paths
        if not os.path.exists(args.container_path):
            logger.error(f"Container not found: {args.container_path}")
            return 1
            
        if not os.path.exists(args.script_path):
            logger.error(f"Script not found: {args.script_path}")
            return 1

        # Generate Slurm scripts
        script_paths = generate_slurm_scripts(
            method, work_dir, slurm_dir, 
            args.container_path, args.script_path
        )
        
        logger.info(f"Script generation completed successfully!")
        logger.info(f"Generated {len(script_paths)} scripts in: {slurm_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to generate Slurm scripts: {e}")
        return 1

if __name__ == '__main__':
    exit(main())