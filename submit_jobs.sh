#!/bin/bash

module load apptainer

# Define paths
BIDS_DIR="/gscratch/fang/narsad"
CONTAINER_IMAGE="/mmfs1/home/xxqian/repos/apptainers/1st_level.sif"

SUBJECTS='N101'

for subj in $SUBJECTS; do
    sbatch <<EOT

#SBATCH --job-name=fmri_${subj}
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fmri_${subj}_%j.out
#SBATCH --error=logs/fmri_${subj}_%j.err

apptainer run --bind $BIDS_DIR:/data $CONTAINER_IMAGE --subject $subj
EOT
done
