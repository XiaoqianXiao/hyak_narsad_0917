import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations, product
from nilearn.image import load_img, index_img, new_img_like
import nibabel as nib
from similarity import searchlight_similarity

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--subject', required=True)
parser.add_argument('--task', required=True)
args = parser.parse_args()
sub = args.subject
task = args.task

# Paths
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
derivatives_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives')
behav_dir = os.path.join(root_dir, project_name, 'MRI', 'source_data', 'behav')
data_dir = os.path.join(derivatives_dir, 'LSS', 'firstLevel', 'all_subjects')
mask_img_path = '/path/to/brain_mask.nii.gz'
output_dir = os.path.join(data_dir, '/searchlight_similarity/sub-{sub}_task-{task}')
os.makedirs(output_dir, exist_ok=True)

bold_4d_path = os.path.join(data_dir, f'sub-{sub}_task-{task}.nii')


# Event file
if sub == 'N202' and task == 'phase3':
    events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_half_events.csv')
else:
    events_file = os.path.join(behav_dir, f'task-Narsad_{task}_half_events.csv')

events = pd.read_csv(events_file)
trial_types = events['trial_type'].unique()
trial_to_type = dict(enumerate(events['trial_type'].values))

bold_4d = load_img(bold_4d_path)
mask_img = load_img(mask_img_path)

# ---- Helper: extract trial indices by type ----
type_to_indices = {
    t: [i for i, tt in trial_to_type.items() if tt == t]
    for t in trial_types
}

# ---- Within-type similarity ----
for ttype, indices in type_to_indices.items():
    sim_maps = []
    for i, j in combinations(indices, 2):
        img1 = index_img(bold_4d, i)
        img2 = index_img(bold_4d, j)
        sim = searchlight_similarity(img1, img2, radius=4, mask_img=mask_img)
        sim_maps.append(sim.get_fdata())

    if sim_maps:
        avg_map = np.nanmean(np.stack(sim_maps, axis=0), axis=0)
        avg_img = new_img_like(mask_img, avg_map)
        nib.save(avg_img, os.path.join(out_dir, f"sub-{sub}_task-{task}_within-{ttype}_searchlight.nii.gz"))

# ---- Between-type similarity ----
for t1, t2 in combinations(trial_types, 2):
    sims = []
    for i in type_to_indices[t1]:
        for j in type_to_indices[t2]:
            img1 = index_img(bold_4d, i)
            img2 = index_img(bold_4d, j)
            sim = searchlight_similarity(img1, img2, radius=4, mask_img=mask_img)
            sims.append(sim.get_fdata())

    if sims:
        avg_map = np.nanmean(np.stack(sims, axis=0), axis=0)
        avg_img = new_img_like(mask_img, avg_map)
        nib.save(avg_img, os.path.join(output_dir, f"sub-{sub}_task-{task}_between-{t1}_vs_{t2}_searchlight.nii.gz"))
