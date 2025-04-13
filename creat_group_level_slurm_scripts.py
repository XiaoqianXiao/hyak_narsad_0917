#%%
import os

container_path = '/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-group-level_1.0.sif'
base_dir = '/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/MRI/derivatives'

tasks = ['phase2', 'phase3']
contrasts = list(range(1, 30))
analysis_types = ['randomise', 'flameo']  # Note: plural variable name

def create_slurm_script(task, contrast, analysis_type):
    job_name = f"group_{task}_cope{contrast}_{analysis_type}"
    script_dir = os.path.join(base_dir, 'groupLevel', 'whole_brain')
    os.makedirs(script_dir, exist_ok=True)
    script_path = os.path.join(script_dir, f"{job_name}.sh")
    out_path = os.path.join(script_dir, f"{job_name}_%j.out")
    err_path = os.path.join(script_dir, f"{job_name}_%j.err")

    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=fang
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output={out_path}
#SBATCH --error={err_path}

module load apptainer
apptainer exec -B /gscratch:/gscratch {container_path} \\
    python3 /app/run_group_level.py \\
    --task {task} \\
    --contrast {contrast} \\
    --analysis_type {analysis_type} \\
    --base_dir {base_dir}
""")
    os.chmod(script_path, 0o755)
    return script_path

#%%
if __name__ == '__main__':
    for task in tasks:
        for contrast in contrasts:
            for analysis_type in analysis_types:
                script_path = create_slurm_script(task, contrast, analysis_type)
                print(f"Created Slurm script: {script_path}")
