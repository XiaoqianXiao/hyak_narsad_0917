import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations, product
from nilearn.image import load_img, index_img, new_img_like
import nibabel as nib
from similarity import searchlight_similarity
from similarity import roi_similarity
from similarity import load_roi_names
from similarity import get_roi_labels

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--subject', required=True)
parser.add_argument('--task', required=True)
parser.add_argument('--mask_img_path', required=True)
parser.add_argument('--combined_atlas_path', required=True)
parser.add_argument('--roi_names_file', required=True)
args = parser.parse_args()
sub = args.subject
task = args.task
mask_img_path = args.mask_img_path
combined_atlas_path = args.combined_atlas_path
roi_names_file = args.roi_names_file

# Paths
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
derivatives_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives')
behav_dir = os.path.join(root_dir, project_name, 'MRI', 'source_data', 'behav')
data_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'firstLevel', 'all_subjects')
output_dir = os.path.join(data_dir, 'similarity')
os.makedirs(output_dir, exist_ok=True)

combined_atlas = load_img(combined_atlas_path)
combined_roi_labels = get_roi_labels(combined_atlas, 'Schaefer+Tian')
roi_names = load_roi_names(roi_names_file, combined_roi_labels)

bold_4d_path = os.path.join(data_dir, f'sub-{sub}_task-{task}.nii')

# Event file
if sub == 'N202' and task == 'phase3':
    events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_events.csv')
else:
    events_file = os.path.join(behav_dir, f'task-Narsad_{task}_events.csv')

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
        nib.save(avg_img, os.path.join(output_dir, 'searchlight', f"sub-{sub}_task-{task}_within-{ttype}.nii.gz"))

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
        nib.save(avg_img, os.path.join(output_dir, 'searchlight', f"sub-{sub}_task-{task}_between-{t1}-{t2}.nii.gz"))

# ---- ROI-based Similarity ----
print(f"Computing ROI similarities for sub-{sub}, task-{task}")

# Initialize DataFrames for combined atlas
roi_dfs = {}
atlas_name = 'combined'
columns = [roi_names[label] for label in combined_roi_labels]
index = [roi_names[label] for label in combined_roi_labels]
for ttype in trial_types:
    roi_dfs[f"within-{ttype}"] = pd.DataFrame(index=index, columns=columns)
for t1, t2 in combinations(trial_types, 2):
    roi_dfs[f"between-{t1}-{t2}"] = pd.DataFrame(index=index, columns=columns)

# Check and align ROI atlas with BOLD data space
from nilearn.image import resample_to_img
combined_atlas_aligned = combined_atlas
if not np.allclose(bold_4d.affine, combined_atlas.affine) or bold_4d.shape != combined_atlas.shape:
    print(f"ROI atlas and BOLD data are not in the same space. Resampling atlas to match BOLD data.")
    combined_atlas_aligned = resample_to_img(combined_atlas, bold_4d, interpolation='nearest')
else:
    print(f"ROI atlas and BOLD data are in the same space. No resampling needed.")

# Process ROI similarities
n_rois = len(combined_roi_labels)
# Within-category similarities
for ttype, indices in type_to_indices.items():
    sim_matrices = []
    for i, j in combinations(indices, 2):
        try:
            img1 = index_img(bold_4d, i)
            img2 = index_img(bold_4d, j)
            sim_matrix = roi_similarity(img1, img2, combined_atlas_aligned, combined_roi_labels, similarity='pearson')
            sim_matrices.append(sim_matrix)
        except Exception as e:
            print(f"Error computing ROI similarity for {atlas_name} {ttype} trials {i} vs {j}: {e}")
            continue

    if sim_matrices:
        avg_sim_matrix = np.nanmean(np.stack(sim_matrices, axis=0), axis=0)
        df = roi_dfs[f"within-{ttype}"]
        for i in range(n_rois):
            for j in range(n_rois):
                df.iloc[i, j + 2] = avg_sim_matrix[i, j]  # +2 to skip subject, task columns

# Between-category similarities
for t1, t2 in combinations(trial_types, 2):
    sim_matrices = []
    for i, j in product(type_to_indices[t1], type_to_indices[t2]):
        try:
            img1 = index_img(bold_4d, i)
            img2 = index_img(bold_4d, j)
            sim_matrix = roi_similarity(img1, img2, combined_atlas_aligned, combined_roi_labels, similarity='pearson')
            sim_matrices.append(sim_matrix)
        except Exception as e:
            print(f"Error computing ROI similarity for {atlas_name} between {t1} trial {i} and {t2} trial {j}: {e}")
            continue

    if sim_matrices:
        avg_sim_matrix = np.nanmean(np.stack(sim_matrices, axis=0), axis=0)
        df = roi_dfs[f"between-{t1}-{t2}"]
        for i in range(n_rois):
            for j in range(n_rois):
                df.iloc[i, j + 2] = avg_sim_matrix[i, j]

# Save ROI DataFrames
for df_name, df in roi_dfs.items():
    output_path = os.path.join(output_dir, 'roi', f"sub-{sub}_task-{task}_{df_name}.csv")
    df.to_csv(output_path, index=True, index_label='ROI1')
    print(f"Saved {df_name} ROI similarities to {output_path}")
