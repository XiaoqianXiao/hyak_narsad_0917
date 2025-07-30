import os
from pathlib import Path
from bids.layout import BIDSLayout

# ----------------------------
# Host vs container paths
# ----------------------------
HOST_DATA_ROOT = Path("/gscratch/fang")
HOST_SCRUBBED_ROOT = Path("/gscratch/scrubbed/fanglab/xiaoqian")

CONT_DATA_ROOT = Path("/data")
CONT_SCRUBBED_ROOT = Path("/scrubbed_dir")

project_name = "NARSAD"

# BIDS on host
data_dir_host = HOST_DATA_ROOT / project_name / "MRI"
derivatives_dir_host = data_dir_host / "derivatives"
behav_dir_host = data_dir_host / "source_data" / "behav"

# Outputs/work on host
output_dir_host = derivatives_dir_host / "fMRI_analysis" / "LSS"
LSS_dir_host = output_dir_host / "firstLevel"
results_dir_host = LSS_dir_host / "all_subjects"
results_dir_host.mkdir(parents=True, exist_ok=True)

# Container image path (host path used by apptainer exec)
container_image = HOST_SCRUBBED_ROOT / "images" / "narsad-fmri_1st_level_1.0.sif"

# Inputs that live under HOST_SCRUBBED_ROOT; will be translated for container
combined_atlas_host = HOST_SCRUBBED_ROOT / "parcellation" / "Tian" / "3T" / "Cortex-Subcortex" / "MNIvolumetric" / \
    "Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_3T_MNI152NLin2009cAsym_2mm.nii.gz"

roi_names_host = HOST_SCRUBBED_ROOT / "parcellation" / "Tian" / "3T" / "Cortex-Subcortex" / \
    "Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_label.txt"

# Where to write Slurm scripts (host)
scripts_root_host = HOST_SCRUBBED_ROOT / project_name / "work_flows" / "Lss_step3"
scripts_root_host.mkdir(parents=True, exist_ok=True)

# Utility: host → container path translation
def to_container_path(p: Path) -> Path:
    p = p.resolve()
    p_str = str(p)
    if p_str.startswith(str(HOST_DATA_ROOT)):
        return Path(str(p).replace(str(HOST_DATA_ROOT), str(CONT_DATA_ROOT), 1))
    if p_str.startswith(str(HOST_SCRUBBED_ROOT)):
        return Path(str(p).replace(str(HOST_SCRUBBED_ROOT), str(CONT_SCRUBBED_ROOT), 1))
    # Fallback: return as-is (may be okay if container binds the original root)
    return p

# BIDS layout (host)
layout = BIDSLayout(
    str(data_dir_host),
    validate=False,
    derivatives=str(derivatives_dir_host)
)

space = "MNI152NLin2009cAsym"

def create_slurm_script(sub: str, task: str, run: str | None,
                        mask_img_host: Path,
                        combined_atlas_host: Path,
                        roi_names_host: Path) -> Path:
    # Translate all argument paths to container-visible paths
    mask_img_cont = to_container_path(mask_img_host)
    atlas_cont = to_container_path(combined_atlas_host)
    roi_cont = to_container_path(roi_names_host)

    # One subdir per task under scripts root (host)
    work_dir_host = scripts_root_host / task
    work_dir_host.mkdir(parents=True, exist_ok=True)

    # Log files directory (host)
    logs_dir_host = HOST_SCRUBBED_ROOT / project_name / "work_flows" / "Lss_step3"
    logs_dir_host.mkdir(parents=True, exist_ok=True)

    run_tag = f"_run-{run}" if run else ""
    script_path = work_dir_host / f"sub-{sub}_task-{task}{run_tag}.sh"

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=LSS_3_sub-{sub}_task-{task}{run_tag}
#SBATCH --account=fang
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output={logs_dir_host}/task_{task}_sub_{sub}{run_tag}_%j.out
#SBATCH --error={logs_dir_host}/task_{task}_sub_{sub}{run_tag}_%j.err

set -euo pipefail

module load apptainer

apptainer exec \\
    -B {HOST_DATA_ROOT}:{CONT_DATA_ROOT} \\
    -B {HOST_SCRUBBED_ROOT}:{CONT_SCRUBBED_ROOT} \\
    -B {HOST_SCRUBBED_ROOT}/repo/hyak_narsad/Creat_LSS_3_slurm_scripts.py:/app/Creat_LSS_3_slurm_scripts.py \\
    -B {HOST_SCRUBBED_ROOT}/repo/hyak_narsad/LSS_3_similarity.py:/app/LSS_3_similarity.py \\
    {container_image} \\
    python3 /app/LSS_3_similarity.py \\
    --subject "{sub}" \\
    --task "{task}" \\
    --mask_img_path "{mask_img_cont}" \\
    --combined_atlas_path "{atlas_cont}" \\
    --roi_names_file "{roi_cont}"
"""
    script_path.write_text(slurm_script)
    return script_path

if __name__ == "__main__":
    subjects = layout.get_subjects()
    tasks = layout.get_tasks()

    for sub in subjects:
        for task in tasks:
            # Find preproc BOLD in requested space
            bold_files = layout.get(
                desc="preproc",
                suffix="bold",
                extension=[".nii", ".nii.gz"],
                subject=sub,
                task=task,
                space=space
            )
            if not bold_files:
                print(f"[skip] No preproc BOLD for sub-{sub}, task-{task} in space {space}")
                continue

            # Process by run (if present)
            for bf in bold_files:
                entities = getattr(bf, "entities", {})
                run = entities.get("run")

                # Find the matching mask (prefer desc='brain')
                mask_candidates = layout.get(
                    suffix="mask",
                    extension=[".nii", ".nii.gz"],
                    subject=sub,
                    task=task,
                    run=run,
                    space=space,
                    desc="brain",
                    return_type="file"
                )
                if not mask_candidates:
                    # fall back to any mask if 'brain' not found
                    mask_candidates = layout.get(
                        suffix="mask",
                        extension=[".nii", ".nii.gz"],
                        subject=sub,
                        task=task,
                        run=run,
                        space=space,
                        return_type="file"
                    )
                if not mask_candidates:
                    print(f"[warn] No mask found for sub-{sub}, task-{task}, run-{run} (space {space}); skipping")
                    continue

                mask_img_host = Path(mask_candidates[0])

                script_path = create_slurm_script(
                    sub=sub,
                    task=task,
                    run=run,
                    mask_img_host=mask_img_host,
                    combined_atlas_host=combined_atlas_host,
                    roi_names_host=roi_names_host
                )
                print(f"Slurm script created: {script_path}")
