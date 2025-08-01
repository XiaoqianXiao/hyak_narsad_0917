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
import time

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
    print(f"Starting processing for sub-{sub}, task-{task}")
    print(f"Mask path: {mask_img_path}, exists: {os.path.exists(mask_img_path)}")
    print(f"Atlas path: {combined_atlas_path}, exists: {os.path.exists(combined_atlas_path)}")
    print(f"ROI names file: {roi_names_file}, exists: {os.path.exists(roi_names_file)}")

    # Paths
    root_dir = os.getenv('DATA_DIR', '/data')
    project_name = 'NARSAD'
    derivatives_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives')
    behav_dir = os.path.join(root_dir, project_name, 'MRI', 'source_data', 'behav')
    data_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'firstLevel', 'all_subjects')
    output_dir = os.path.join(data_dir, 'similarity')
    try:
        os.makedirs(os.path.join(output_dir, 'searchlight'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'roi'), exist_ok=True)
        print(f"Output directories created: {output_dir}/searchlight, {output_dir}/roi")
    except Exception as e:
        print(f"Error creating output directories: {e}")
        return

    # Load data
    print(f"Loading BOLD data from {data_dir}")
    bold_4d_path = os.path.join(data_dir, f'sub-{sub}_task-{task}.nii')
    print(f"BOLD path: {bold_4d_path}, exists: {os.path.exists(bold_4d_path)}")
    try:
        bold_4d = load_img(bold_4d_path)
        print(f"BOLD shape: {bold_4d.shape}, affine: {bold_4d.affine}")
    except Exception as e:
        print(f"Error loading BOLD data {bold_4d_path}: {e}")
        return
    try:
        mask_img = load_img(mask_img_path)
        print(f"Mask shape: {mask_img.shape}, affine: {mask_img.affine}")
        print(f"Number of voxels in mask: {np.sum(mask_img.get_fdata() > 0)}")
    except Exception as e:
        print(f"Error loading mask {mask_img_path}: {e}")
        return
    try:
        combined_atlas = load_img(combined_atlas_path)
        print(f"Atlas shape: {combined_atlas.shape}, affine: {combined_atlas.affine}")
    except Exception as e:
        print(f"Error loading atlas {combined_atlas_path}: {e}")
        return

    combined_roi_labels = get_roi_labels(combined_atlas, 'Schaefer+Tian')
    print(f"ROI labels: {len(combined_roi_labels)} found")
    roi_names = load_roi_names(roi_names_file, combined_roi_labels)
    print(f"Loaded {len(roi_names)} ROI names")

    # Event file
    if sub == 'N202' and task == 'phase3':
        events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_events.csv')
    else:
        events_file = os.path.join(behav_dir, f'task-Narsad_{task}_events.csv')
    print(f"Events file: {events_file}, exists: {os.path.exists(events_file)}")
    try:
        events = pd.read_csv(events_file)
        print(f"Events loaded, shape: {events.shape}, columns: {events.columns}")
    except Exception as e:
        print(f"Error loading events file {events_file}: {e}")
        return
    trial_types = events['trial_type'].unique()
    print(f"Trial types: {trial_types}")
    trial_to_type = dict(enumerate(events['trial_type'].values))

    # Extract trial indices by type
    type_to_indices = {t: [i for i, tt in trial_to_type.items() if tt == t] for t in trial_types}
    print(f"Type to indices: {type_to_indices}")

    # ---- Searchlight Similarity ----
    def compute_searchlight_pair(i, j, bold_4d, mask_img):
        try:
            img1 = index_img(bold_4d, i)
            img2 = index_img(bold_4d, j)
            start_time = time.time()
            result = searchlight_similarity(img1, img2, radius=4, mask_img=mask_img, n_jobs=4).get_fdata()
            print(f"Computed searchlight similarity for trials {i} vs {j} in {time.time() - start_time:.2f} seconds")
            return result
        except Exception as e:
            print(f"Error computing searchlight similarity for trials {i} vs {j}: {e}")
            return None

    # Within-type similarity
    for ttype, indices in type_to_indices.items():
        pairs = list(combinations(indices, 2))
        print(f"Within-type {ttype}: {len(pairs)} pairs")
        if pairs:
            start_time = time.time()
            sim_maps = Parallel(n_jobs=4, verbose=10)(
                delayed(compute_searchlight_pair)(i, j, bold_4d, mask_img)
                for i, j in pairs
            )
            print(f"Searchlight for {ttype} took {time.time() - start_time:.2f} seconds")
            sim_maps = [m for m in sim_maps if m is not None]
            if sim_maps:
                avg_map = np.nanmean(np.stack(sim_maps, axis=0), axis=0)
                avg_img = new_img_like(mask_img, avg_map)
                output_path = os.path.join(output_dir, 'searchlight', f"sub-{sub}_task-{task}_within-{ttype}.nii.gz")
                print(f"Saving searchlight to {output_path}")
                try:
                    nib.save(avg_img, output_path)
                    print(f"Saved searchlight for {ttype}")
                except Exception as e:
                    print(f"Error saving searchlight {output_path}: {e}")
            else:
                print(f"No valid searchlight maps for {ttype}")

    # Between-type similarity
    for t1, t2 in combinations(trial_types, 2):
        pairs = list(product(type_to_indices[t1], type_to_indices[t2]))
        print(f"Between-type {t1}-{t2}: {len(pairs)} pairs")
        if pairs:
            start_time = time.time()
            sims = Parallel(n_jobs=4, verbose=10)(
                delayed(compute_searchlight_pair)(i, j, bold_4d, mask_img)
                for i, j in pairs
            )
            print(f"Searchlight for {t1}-{t2} took {time.time() - start_time:.2f} seconds")
            sims = [m for m in sims if m is not None]
            if sims:
                avg_map = np.nanmean(np.stack(sims, axis=0), axis=0)
                avg_img = new_img_like(mask_img, avg_map)
                output_path = os.path.join(output_dir, 'searchlight', f"sub-{sub}_task-{task}_between-{t1}-{t2}.nii.gz")
                print(f"Saving searchlight to {output_path}")
                try:
                    nib.save(avg_img, output_path)
                    print(f"Saved searchlight for {t1}-{t2}")
                except Exception as e:
                    print(f"Error saving searchlight {output_path}: {e}")
            else:
                print(f"No valid searchlight maps for {t1}-{t2}")

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
        try:
            combined_atlas_aligned = resample_img(combined_atlas, bold_4d, interpolation='nearest')
            print(f"Resampled atlas shape: {combined_atlas_aligned.shape}, affine: {combined_atlas_aligned.affine}")
            if np.any(np.isnan(combined_atlas_aligned.get_fdata())):
                print("Warning: Resampled atlas contains NaNs")
        except Exception as e:
            print(f"Error resampling atlas: {e}")
            return

    # Compute ROI similarities
    def compute_roi_pair(i, j, bold_4d, atlas_img, roi_labels):
        try:
            img1 = index_img(bold_4d, i)
            img2 = index_img(bold_4d, j)
            start_time = time.time()
            result = roi_similarity(img1, img2, atlas_img, roi_labels, similarity='pearson', n_jobs=4)
            print(f"Computed ROI similarity for trials {i} vs {j} in {time.time() - start_time:.2f} seconds")
            return result
        except Exception as e:
            print(f"Error computing ROI similarity for trials {i} vs {j}: {e}")
            return None

    n_rois = len(combined_roi_labels)
    print(f"Number of ROIs: {n_rois}, pairs: {n_rois * n_rois}")
    # Within-type
    for ttype, indices in type_to_indices.items():
        pairs = list(combinations(indices, 2))
        print(f"Within-type {ttype} ROI: {len(pairs)} pairs")
        if pairs:
            start_time = time.time()
            sim_matrices = Parallel(n_jobs=4, verbose=10)(
                delayed(compute_roi_pair)(i, j, bold_4d, combined_atlas_aligned, combined_roi_labels)
                for i, j in pairs
            )
            print(f"ROI similarity for {ttype} took {time.time() - start_time:.2f} seconds")
            sim_matrices = [m for m in sim_matrices if m is not None]
            if sim_matrices:
                avg_sim_matrix = np.nanmean(np.stack(sim_matrices, axis=0), axis=0)
                df = roi_dfs[f"within-{ttype}"]
                for i in range(n_rois):
                    for j in range(n_rois):
                        df.iloc[i, j] = avg_sim_matrix[i, j]
            else:
                print(f"No valid ROI matrices for {ttype}")

    # Between-type
    for t1, t2 in combinations(trial_types, 2):
        pairs = list(product(type_to_indices[t1], type_to_indices[t2]))
        print(f"Between-type {t1}-{t2} ROI: {len(pairs)} pairs")
        if pairs:
            start_time = time.time()
            sim_matrices = Parallel(n_jobs=4, verbose=10)(
                delayed(compute_roi_pair)(i, j, bold_4d, combined_atlas_aligned, combined_roi_labels)
                for i, j in pairs
            )
            print(f"ROI similarity for {t1}-{t2} took {time.time() - start_time:.2f} seconds")
            sim_matrices = [m for m in sim_matrices if m is not None]
            if sim_matrices:
                avg_sim_matrix = np.nanmean(np.stack(sim_matrices, axis=0), axis=0)
                df = roi_dfs[f"between-{t1}-{t2}"]
                for i in range(n_rois):
                    for j in range(n_rois):
                        df.iloc[i, j] = avg_sim_matrix[i, j]
            else:
                print(f"No valid ROI matrices for {t1}-{t2}")

    # Save ROI DataFrames
    for df_name, df in roi_dfs.items():
        output_path = os.path.join(output_dir, 'roi', f"sub-{sub}_task-{task}_{df_name}.csv")
        print(f"Saving ROI to {output_path}")
        try:
            df.to_csv(output_path, index=True, index_label='ROI1')
            print(f"Saved {df_name} ROI similarities")
        except Exception as e:
            print(f"Error saving ROI {output_path}: {e}")

if __name__ == '__main__':
    if args.profile:
        print("Running with cProfile")
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        try:
            stats.dump_stats('profile.out')
            print("Profiling stats saved to profile.out")
        except Exception as e:
            print(f"Error saving profile.out: {e}")
    else:
        main()