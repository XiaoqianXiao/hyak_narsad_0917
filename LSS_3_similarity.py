import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations, product
from nilearn.image import load_img, index_img, new_img_like, resample_to_img
import nibabel as nib
from similarity import searchlight_similarity, roi_similarity, load_roi_names, get_roi_labels
from joblib import Parallel, delayed
import cProfile
import pstats
import time
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

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--subject', required=True)
parser.add_argument('--task', required=True)
parser.add_argument('--mask_img_path', required=True)
parser.add_argument('--combined_atlas_path', required=True)
parser.add_argument('--roi_names_file', required=True)
parser.add_argument('--analysis_type', choices=['searchlight', 'roi', 'both'], default='both',
                    help='Type of analysis to run: searchlight, roi, or both')
parser.add_argument('--batch_size', type=int, default=1000, help='Number of voxels per batch for searchlight')
parser.add_argument('--n_jobs', type=int, default=12, help='Number of parallel jobs')
parser.add_argument('--profile', action='store_true', help='Enable cProfile for debugging')
args = parser.parse_args()

logger = setup_logging()

def main():
    sub = args.subject
    task = args.task
    mask_img_path = args.mask_img_path
    combined_atlas_path = args.combined_atlas_path
    roi_names_file = args.roi_names_file
    analysis_type = args.analysis_type
    logger.info(f"Starting processing for sub-{sub}, task-{task}, analysis_type={analysis_type}")
    logger.info(f"Mask path: {mask_img_path}, exists: {os.path.exists(mask_img_path)}")
    logger.info(f"Atlas path: {combined_atlas_path}, exists: {os.path.exists(combined_atlas_path)}")
    logger.info(f"ROI names file: {roi_names_file}, exists: {os.path.exists(roi_names_file)}")

    # Paths
    root_dir = os.getenv('DATA_DIR', '/data')
    project_name = 'NARSAD'
    derivatives_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives')
    behav_dir = os.path.join(root_dir, project_name, 'MRI', 'source_data', 'behav')
    data_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'firstLevel', 'all_subjects')
    output_dir = os.path.join(data_dir, 'similarity')
    try:
        if analysis_type in ['searchlight', 'both']:
            os.makedirs(os.path.join(output_dir, 'searchlight'), exist_ok=True)
        if analysis_type in ['roi', 'both']:
            os.makedirs(os.path.join(output_dir, 'roi'), exist_ok=True)
        logger.info(f"Output directories created: {output_dir}/searchlight, {output_dir}/roi")
    except Exception as e:
        logger.error(f"Error creating output directories: {e}")
        return

    # Load data
    logger.info(f"Loading BOLD data from {data_dir}")
    bold_4d_path = os.path.join(data_dir, f'sub-{sub}_task-{task}.nii')
    logger.info(f"BOLD path: {bold_4d_path}, exists: {os.path.exists(bold_4d_path)}")
    try:
        bold_4d = load_img(bold_4d_path)
        logger.info(f"BOLD shape: {bold_4d.shape}, affine: {bold_4d.affine}")
        bold_data = bold_4d.get_fdata()
        nan_count = np.sum(np.isnan(bold_data))
        zero_count = np.sum(np.all(bold_data == 0, axis=3))
        logger.info(f"BOLD data NaN count: {nan_count}, all-zero voxel count: {zero_count}")
        if nan_count > 0 or zero_count > 0:
            logger.warning("BOLD data contains NaNs or all-zero voxels, which may cause computation issues")
    except Exception as e:
        logger.error(f"Error loading BOLD data {bold_4d_path}: {e}")
        return

    # Load mask
    try:
        mask_img = load_img(mask_img_path)
        logger.info(f"Mask shape: {mask_img.shape}, affine: {mask_img.affine}")
        mask_data = mask_img.get_fdata()
        valid_voxels = np.sum(mask_data > 0)
        nan_count = np.sum(np.isnan(mask_data))
        logger.info(f"Number of voxels in mask: {valid_voxels}, NaN count: {nan_count}")
        if valid_voxels == 0:
            logger.error("Mask contains no valid voxels (all values <= 0)")
            return
        if nan_count > 0:
            logger.warning("Mask contains NaNs, which may cause resampling issues")
    except Exception as e:
        logger.error(f"Error loading mask {mask_img_path}: {e}")
        return

    # Load atlas and ROI names
    if analysis_type in ['roi', 'both']:
        try:
            combined_atlas = load_img(combined_atlas_path)
            logger.info(f"Atlas shape: {combined_atlas.shape}, affine: {combined_atlas.affine}")
            atlas_data = combined_atlas.get_fdata()
            nan_count = np.sum(np.isnan(atlas_data))
            unique_labels = np.unique(atlas_data[atlas_data > 0])
            logger.info(f"Atlas NaN count: {nan_count}, unique positive labels: {len(unique_labels)}")
            if nan_count > 0:
                logger.warning("Atlas data contains NaNs, which may cause resampling issues")
            if len(unique_labels) == 0:
                logger.error("Atlas contains no valid ROIs (all values <= 0)")
                return
        except Exception as e:
            logger.error(f"Error loading atlas {combined_atlas_path}: {e}")
            return

        combined_roi_labels = get_roi_labels(combined_atlas, 'Schaefer+Tian')
        logger.info(f"ROI labels: {len(combined_roi_labels)} found")
        roi_names = load_roi_names(roi_names_file, combined_roi_labels)
        logger.info(f"Loaded {len(roi_names)} ROI names")

    # Event file
    if sub == 'N202' and task == 'phase3':
        events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_events.csv')
    else:
        events_file = os.path.join(behav_dir, f'task-Narsad_{task}_events.csv')
    logger.info(f"Events file: {events_file}, exists: {os.path.exists(events_file)}")
    try:
        events = pd.read_csv(events_file)
        logger.info(f"Events loaded, shape: {events.shape}, columns: {events.columns}")
    except Exception as e:
        logger.error(f"Error loading events file {events_file}: {e}")
        return
    trial_types = events['trial_type'].unique()
    logger.info(f"Trial types: {trial_types}")
    trial_to_type = dict(enumerate(events['trial_type'].values))
    type_to_indices = {t: [i for i, tt in trial_to_type.items() if tt == t] for t in trial_types}
    logger.info(f"Type to indices: {type_to_indices}")

    # ---- Searchlight Similarity ----
    if analysis_type in ['searchlight', 'both']:
        def compute_searchlight_pair(i, j, bold_4d, mask_img, pair_num, total_pairs):
            try:
                img1 = index_img(bold_4d, i)
                img2 = index_img(bold_4d, j)
                start_time = time.time()
                result = searchlight_similarity(
                    img1, img2, radius=6, mask_img=mask_img, similarity='pearson',
                    n_jobs=args.n_jobs, batch_size=args.batch_size
                ).get_fdata()
                elapsed = time.time() - start_time
                logger.info(f"Searchlight pair {pair_num}/{total_pairs} (trials {i} vs {j}) completed in {elapsed:.2f} seconds")
                return result
            except Exception as e:
                logger.error(f"Error computing searchlight similarity for trials {i} vs {j}: {e}")
                return None

        # Within-type similarity
        for ttype, indices in type_to_indices.items():
            pairs = list(combinations(indices, 2))
            total_pairs = len(pairs)
            logger.info(f"Within-type {ttype}: {total_pairs} pairs")
            if pairs:
                start_time = time.time()
                # Parallelize pair computations
                sim_maps = Parallel(n_jobs=2, verbose=0)(
                    delayed(compute_searchlight_pair)(i, j, bold_4d, mask_img, idx, total_pairs)
                    for idx, (i, j) in enumerate(pairs, 1)
                )
                elapsed = time.time() - start_time
                logger.info(f"Searchlight for {ttype} completed in {elapsed:.2f} seconds")
                sim_maps = [m for m in sim_maps if m is not None]
                if sim_maps:
                    avg_map = np.nanmean(np.stack(sim_maps, axis=0), axis=0)
                    avg_img = new_img_like(mask_img, avg_map)
                    output_path = os.path.join(output_dir, 'searchlight', f"sub-{sub}_task-{task}_within-{ttype}.nii.gz")
                    logger.info(f"Saving searchlight to {output_path}")
                    try:
                        nib.save(avg_img, output_path)
                        logger.info(f"Saved searchlight for {ttype}")
                    except Exception as e:
                        logger.error(f"Error saving searchlight {output_path}: {e}")
                else:
                    logger.warning(f"No valid searchlight maps for {ttype}")

        # Between-type similarity
        for t1, t2 in combinations(trial_types, 2):
            pairs = list(product(type_to_indices[t1], type_to_indices[t2]))
            total_pairs = len(pairs)
            logger.info(f"Between-type {t1}-{t2}: {total_pairs} pairs")
            if pairs:
                start_time = time.time()
                sims = Parallel(n_jobs=2, verbose=0)(
                    delayed(compute_searchlight_pair)(i, j, bold_4d, mask_img, idx, total_pairs)
                    for idx, (i, j) in enumerate(pairs, 1)
                )
                elapsed = time.time() - start_time
                logger.info(f"Searchlight for {t1}-{t2} completed in {elapsed:.2f} seconds")
                sims = [m for m in sims if m is not None]
                if sims:
                    avg_map = np.nanmean(np.stack(sims, axis=0), axis=0)
                    avg_img = new_img_like(mask_img, avg_map)
                    output_path = os.path.join(output_dir, 'searchlight', f"sub-{sub}_task-{task}_between-{t1}-{t2}.nii.gz")
                    logger.info(f"Saving searchlight to {output_path}")
                    try:
                        nib.save(avg_img, output_path)
                        logger.info(f"Saved searchlight for {t1}-{t2}")
                    except Exception as e:
                        logger.error(f"Error saving searchlight {output_path}: {e}")
                else:
                    logger.warning(f"No valid searchlight maps for {t1}-{t2}")

    # ---- ROI-based Similarity ----
    if analysis_type in ['roi', 'both']:
        logger.info(f"Computing ROI similarities for sub-{sub}, task-{task}")

        # Initialize DataFrames
        roi_dfs = {}
        columns = [roi_names[label] for label in combined_roi_labels]
        index = [roi_names[label] for label in combined_roi_labels]
        for ttype in trial_types:
            roi_dfs[f"within-{ttype}"] = pd.DataFrame(index=index, columns=columns)
        for t1, t2 in combinations(trial_types, 2):
            roi_dfs[f"between-{t1}-{t2}"] = pd.DataFrame(index=index, columns=columns)

        # Align atlas with BOLD using mask
        combined_atlas_aligned = combined_atlas
        if not np.allclose(bold_4d.affine, combined_atlas.affine) or bold_4d.shape[:3] != combined_atlas.shape:
            logger.info(f"Resampling atlas to match BOLD data space using mask")
            try:
                logger.info(f"Using mask as reference: shape={mask_img.shape}, affine={mask_img.affine}")
                combined_atlas_aligned = resample_to_img(combined_atlas, mask_img, interpolation='nearest')
                logger.info(f"Resampled atlas shape: {combined_atlas_aligned.shape}, affine: {combined_atlas_aligned.affine}")
                atlas_data = combined_atlas_aligned.get_fdata()
                nan_count = np.sum(np.isnan(atlas_data))
                unique_labels = np.unique(atlas_data[atlas_data > 0])
                logger.info(f"Resampled atlas NaN count: {nan_count}, unique positive labels: {len(unique_labels)}")
                if nan_count > 0:
                    logger.warning("Resampled atlas contains NaNs")
                if len(unique_labels) == 0:
                    logger.error("Resampled atlas has no valid ROIs (all values <= 0)")
                    return
            except Exception as e:
                logger.error(f"Error resampling atlas: {e}")
                return
        else:
            logger.info("Atlas and BOLD have compatible affines and shapes, using original atlas")

        # Compute ROI similarities
        n_rois = len(combined_roi_labels)
        logger.info(f"Number of ROIs: {n_rois}, pairs: {n_rois * n_rois}")
        # Within-type
        for ttype, indices in type_to_indices.items():
            pairs = list(combinations(indices, 2))
            total_pairs = len(pairs)
            logger.info(f"Within-type {ttype} ROI: {total_pairs} pairs")
            if pairs:
                start_time = time.time()
                sim_matrices = Parallel(n_jobs=args.n_jobs, verbose=0)(
                    delayed(roi_similarity)(
                        index_img(bold_4d, i), index_img(bold_4d, j),
                        combined_atlas_aligned, combined_roi_labels, similarity='pearson', n_jobs=1
                    ) for i, j in pairs
                )
                elapsed = time.time() - start_time
                logger.info(f"ROI similarity for {ttype} completed in {elapsed:.2f} seconds")
                sim_matrices = [m for m in sim_matrices if m is not None]
                if sim_matrices:
                    avg_sim_matrix = np.nanmean(np.stack(sim_matrices, axis=0), axis=0)
                    df = roi_dfs[f"within-{ttype}"]
                    for i in range(n_rois):
                        for j in range(n_rois):
                            df.iloc[i, j] = avg_sim_matrix[i, j]
                else:
                    logger.warning(f"No valid ROI matrices for {ttype}")

        # Between-type
        for t1, t2 in combinations(trial_types, 2):
            pairs = list(product(type_to_indices[t1], type_to_indices[t2]))
            total_pairs = len(pairs)
            logger.info(f"Between-type {t1}-{t2} ROI: {total_pairs} pairs")
            if pairs:
                start_time = time.time()
                sim_matrices = Parallel(n_jobs=args.n_jobs, verbose=0)(
                    delayed(roi_similarity)(
                        index_img(bold_4d, i), index_img(bold_4d, j),
                        combined_atlas_aligned, combined_roi_labels, similarity='pearson', n_jobs=1
                    ) for i, j in pairs
                )
                elapsed = time.time() - start_time
                logger.info(f"ROI similarity for {t1}-{t2} completed in {elapsed:.2f} seconds")
                sim_matrices = [m for m in sim_matrices if m is not None]
                if sim_matrices:
                    avg_sim_matrix = np.nanmean(np.stack(sim_matrices, axis=0), axis=0)
                    df = roi_dfs[f"between-{t1}-{t2}"]
                    for i in range(n_rois):
                        for j in range(n_rois):
                            df.iloc[i, j] = avg_sim_matrix[i, j]
                else:
                    logger.warning(f"No valid ROI matrices for {t1}-{t2}")

        # Save ROI DataFrames
        for df_name, df in roi_dfs.items():
            output_path = os.path.join(output_dir, 'roi', f"sub-{sub}_task-{task}_{df_name}.csv")
            logger.info(f"Saving ROI to {output_path}")
            try:
                df.to_csv(output_path, index=True, index_label='ROI1')
                logger.info(f"Saved {df_name} ROI similarities")
            except Exception as e:
                logger.error(f"Error saving ROI {output_path}: {e}")

if __name__ == '__main__':
    if args.profile:
        logger.info("Running with cProfile")
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        output_path = os.path.join(
            os.getenv('DATA_DIR', '/data'), 'NARSAD', 'MRI', 'derivatives',
            'fMRI_analysis', 'LSS', 'firstLevel', 'all_subjects', 'similarity', 'searchlight',
            f'sub-{args.subject}_task-{args.task}_profile.prof'
        )
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            profiler.dump_stats(output_path)
            stats = pstats.Stats(output_path)
            stats.sort_stats('cumulative').print_stats(20)
            logger.info(f"Profiling stats saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving profiling stats to {output_path}: {e}")
    else:
        main()