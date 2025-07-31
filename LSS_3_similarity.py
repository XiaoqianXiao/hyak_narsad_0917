import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations, product
from nilearn.image import load_img, index_img, new_img_like, resample_img
import nibabel as nib
from similarity import searchlight_similarity, roi_similarity, load_roi_names, get_roi_labels
from joblib import Parallel, delayed
import cProfile
import pstats

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--subject', required=True)
parser.add_argument('--task', required=True)
parser.add_argument('--mask_img_path', required=True)
parser.add_argument('--combined_atlas_path', required=True)
parser.add_argument('--roi_names_file', required=True)
parser.add_argument('--profile', action='store_true', help='Enable cProfile for debugging')
args = parser.parse_args()

def main():
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
    os.makedirs(os.path.join(output_dir, 'searchlight'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'roi'), exist_ok=True)

    # Load data
    combined_atlas = load_img(combined_atlas_path)
    bold_4d_path = os.path.join(data_dir, f'sub-{sub}_task-{task}.nii')
    bold_4d = load_img(bold_4d_path)
    mask_img = load_img(mask_img_path)

    combined_roi_labels = get_roi_labels(combined_atlas, 'Schaefer+Tian')
    roi_names = load_roi_names(roi_names_file, combined_roi_labels)

    # Event file
    if sub == 'N202' and task == 'phase3':
        events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_events.csv')
    else:
        events_file = os.path.join(behav_dir, f'task-Narsad_{task}_events.csv')
    events = pd.read_csv(events_file)
    trial_types = events['trial_type'].unique()
    trial_to_type = dict(enumerate(events['trial_type'].values))

    # Extract trial indices by type
    type_to_indices = {t: [i for i, tt in trial_to_type.items() if tt == t] for t in trial_types}

    # ---- Searchlight Similarity ----
    def compute_searchlight_pair(i, j, bold_4d, mask_img):
        img1 = index_img(bold_4d, i)
        img2 = index_img(bold_4d, j)
        return searchlight_similarity(img1, img2, radius=4, mask_img=mask_img, n_jobs=4).get_fdata()

    # Within-type similarity
    for ttype, indices in type_to_indices.items():
        pairs = list(combinations(indices, 2))
        if pairs:
            sim_maps = Parallel(n_jobs=4)(
                delayed(compute_searchlight_pair)(i, j, bold_4d, mask_img)
                for i, j in pairs
            )
            avg_map = np.nanmean(np.stack(sim_maps, axis=0), axis=0)
            avg_img = new_img_like(mask_img, avg_map)
            nib.save(avg_img, os.path.join(output_dir, 'searchlight', f"sub-{sub}_task-{task}_within-{ttype}.nii.gz"))

    # Between-type similarity
    for t1, t2 in combinations(trial_types, 2):
        pairs = list(product(type_to_indices[t1], type_to_indices[t2]))
        if pairs:
            sims = Parallel(n_jobs=4)(
                delayed(compute_searchlight_pair)(i, j, bold_4d, mask_img)
                for i, j in pairs
            )
            avg_map = np.nanmean(np.stack(sims, axis=0), axis=0)
            avg_img = new_img_like(mask_img, avg_map)
            nib.save(avg_img, os.path.join(output_dir, 'searchlight', f"sub-{sub}_task-{task}_between-{t1}-{t2}.nii.gz"))

    # ---- ROI-based Similarity ----
    print(f"Computing ROI similarities for sub-{sub}, task-{task}")

    # Initialize DataFrames
    roi_dfs = {}
    columns = [roi_names[label] for label in combined_roi_labels]
    index = [roi_names[label] for label in combined_roi_labels]
    for ttype in trial_types:
        roi_dfs[f"within-{ttype}"] = pd.DataFrame(index=index, columns=columns)
    for t1, t2 in combinations(trial_types, 2):
        roi_dfs[f"between-{t1}-{t2}"] = pd.DataFrame(index=index, columns=columns)

    # Align atlas with BOLD
    combined_atlas_aligned = combined_atlas
    if not np.allclose(bold_4d.affine, combined_atlas.affine) or bold_4d.shape[:3] != combined_atlas.shape:
        print(f"Resampling atlas to match BOLD data")
        combined_atlas_aligned = resample_img(combined_atlas, bold_4d, interpolation='nearest')

    # Compute ROI similarities
    def compute_roi_pair(i, j, bold_4d, atlas_img, roi_labels):
        try:
            img1 = index_img(bold_4d, i)
            img2 = index_img(bold_4d, j)
            return roi_similarity(img1, img2, atlas_img, roi_labels, similarity='pearson', n_jobs=4)
        except Exception as e:
            print(f"Error computing ROI similarity for trials {i} vs {j}: {e}")
            return None

    n_rois = len(combined_roi_labels)
    # Within-type
    for ttype, indices in type_to_indices.items():
        pairs = list(combinations(indices, 2))
        if pairs:
            sim_matrices = Parallel(n_jobs=4)(
                delayed(compute_roi_pair)(i, j, bold_4d, combined_atlas_aligned, combined_roi_labels)
                for i, j in pairs
            )
            sim_matrices = [m for m in sim_matrices if m is not None]
            if sim_matrices:
                avg_sim_matrix = np.nanmean(np.stack(sim_matrices, axis=0), axis=0)
                df = roi_dfs[f"within-{ttype}"]
                for i in range(n_rois):
                    for j in range(n_rois):
                        df.iloc[i, j] = avg_sim_matrix[i, j]

    # Between-type
    for t1, t2 in combinations(trial_types, 2):
        pairs = list(product(type_to_indices[t1], type_to_indices[t2]))
        if pairs:
            sim_matrices = Parallel(n_jobs=4)(
                delayed(compute_roi_pair)(i, j, bold_4d, combined_atlas_aligned, combined_roi_labels)
                for i, j in pairs
            )
            sim_matrices = [m for m in sim_matrices if m is not None]
            if sim_matrices:
                avg_sim_matrix = np.nanmean(np.stack(sim_matrices, axis=0), axis=0)
                df = roi_dfs[f"between-{t1}-{t2}"]
                for i in range(n_rois):
                    for j in range(n_rois):
                        df.iloc[i, j] = avg_sim_matrix[i, j]

    # Save ROI DataFrames
    for df_name, df in roi_dfs.items():
        output_path = os.path.join(output_dir, 'roi', f"sub-{sub}_task-{task}_{df_name}.csv")
        df.to_csv(output_path, index=True, index_label='ROI1')
        print(f"Saved {df_name} ROI similarities to {output_path}")

if __name__ == '__main__':
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.dump_stats('profile.out')
        print("Profiling stats saved to profile.out")
    else:
        main()