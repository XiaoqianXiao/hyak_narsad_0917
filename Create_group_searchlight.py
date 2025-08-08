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

def generate_slurm_scripts(method):
    # Paths
    root_dir = os.getenv('DATA_DIR', '/data')
    project_name = 'NARSAD'
    derivatives_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives')
    output_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'groupLevel', 'searchlight', method)
    slurm_dir = os.path.join(output_dir, 'slurm_scripts')
    os.makedirs(slurm_dir, exist_ok=True)
    logger.info(f"Slurm scripts directory: {slurm_dir}")

    # Task and trial types
    task = 'phase3'
    trial_types = ['SHOCK', 'FIXATION', 'CS-', 'CSS', 'CSR']
    map_types = [f'within-{ttype}' for ttype in trial_types] + [f'between-{t1}-{t2}' for t1, t2 in combinations(trial_types, 2)]
    logger.info(f"Generating Slurm scripts for {len(map_types)} map types with method {method}: {map_types}")

    # Slurm template
    slurm_template = """#!/bin/bash
#SBATCH --job-name=group_searchlight_{map_type}_{method}
#SBATCH --output={slurm_dir}/group_searchlight_{map_type}_{method}_%j.out
#SBATCH --error={slurm_dir}/group_searchlight_{map_type}_{method}_%j.err
#SBATCH --time={time}
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS=4
apptainer exec /gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif python3 /data/NARSAD/scripts/group_searchlight.py --map_type {map_type} --method {method}
"""

    # Set time limit based on method
    time_limit = '1:00:00' if method == 'flameo' else '03:00:00'  # 1 hour for Randomise

    # Generate Slurm script for each map type
    script_paths = []
    for map_type in map_types:
        script_path = os.path.join(slurm_dir, f'group_searchlight_{map_type}_{method}.sh')
        with open(script_path, 'w') as f:
            f.write(slurm_template.format(map_type=map_type, method=method, slurm_dir=slurm_dir, time=time_limit))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Slurm scripts for group-level searchlight analysis.')
    parser.add_argument('--method', choices=['flameo', 'randomise'], default='flameo', help='Analysis method: flameo or randomise')
    args = parser.parse_args()
    generate_slurm_scripts(args.method)