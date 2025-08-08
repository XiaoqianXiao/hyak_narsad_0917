import os
import argparse
from itertools import combinations
import logging

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

def generate_slurm_scripts(method, work_dir, slurm_dir):
    # Trial types
    trial_types = ['SHOCK', 'FIXATION', 'CS-', 'CSS', 'CSR']
    map_types = [f'within-{ttype}' for ttype in trial_types] + [f'between-{t1}-{t2}' for t1, t2 in combinations(trial_types, 2)]
    logger.info(f"Generating Slurm scripts for {len(map_types)} map types with method {method}: {map_types}")

    # Slurm template
    slurm_template = """#!/bin/bash
#SBATCH --job-name=group_searchlight_{map_type}_{method}
#SBATCH --output=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_group_searchlight/%s_group_searchlight_{map_type}_{method}_%%j.out
#SBATCH --error=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_group_searchlight/%s_group_searchlight_{map_type}_{method}_%%j.err
#SBATCH --time={time}
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS=4
apptainer exec /gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif python /data/NARSAD/scripts/group_searchlight.py --map_type {map_type} --method {method}
"""

    # Set time limit based on method
    time_limit = '00:30:00' if method == 'flameo' else '01:00:00'  # 1 hour for Randomise

    # Generate Slurm script for each map type
    script_paths = []
    for map_type in map_types:
        script_path = os.path.join(slurm_dir, f'group_searchlight_{map_type}_slurm.sh')
        tasks = ['phase2', 'phase3']
        with open(script_path, 'w') as f:
            for task in tasks:
                f.write(slurm_template % (task, task))
        os.chmod(script_path, 0o755)  # Make executable
        script_paths.append(script_path)
        logger.info(f"Generated Slurm script: {script_path}")

    # Generate master submission script
    master_script = os.path.join(slurm_dir, f'submit_all_group_searchlight_{method}.sh')
    with open(master_script, 'w') as f:
        f.write("#!/bin/bash\n")
        for script_path in script_paths:
            f.write(f"sbatch {script_path}\n")
    os.chmod(master_script, 0o755)
    logger.info(f"Generated master submission script: {master_script}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate Slurm scripts for group-level searchlight analysis.')
    parser.add_argument('--method', choices=['flameo', 'randomise'], default='flameo', help='Analysis method: flameo or randomise')
    args = parser.parse_args()
    method = args.method

    # Create work_dir and slurm_dir
    scrubbed_dir = '/gscratch/scrubbed/fanglab/xiaoqian'
    project_name = 'NARSAD'
    work_dir = os.path.join(scrubbed_dir, project_name, 'work_flows', 'Lss_group_searchlight')
    slurm_dir = os.path.join(work_dir, method)
    os.makedirs(slurm_dir, exist_ok=True)
    logger.info(f"Slurm scripts directory: {slurm_dir}")

    # Generate Slurm scripts
    generate_slurm_scripts(method, work_dir, slurm_dir)

if __name__ == '__main__':
    main()