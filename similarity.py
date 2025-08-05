import numpy as np
from nilearn.image import index_img, load_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.input_data import NiftiMasker
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import nibabel as nib
from joblib import Parallel, delayed
import os
import re
import logging

logger = logging.getLogger(__name__)


def searchlight_similarity(img1, img2, radius=6, affine=None, mask_img=None, similarity='pearson', n_jobs=12, batch_size=1000):
    """
    Compute voxel-wise similarity between two 3D or 4D images using a cubic searchlight approach.

    Parameters:
        img1: nib.Nifti1Image - first image (3D or 4D)
        img2: nib.Nifti1Image - second image (must match img1 shape)
        radius: int - half the side length of the cube in mm (cube side = 2 * radius)
        affine: optional affine to transform coordinates (default: from mask_img)
        mask_img: binary Nifti image for where to apply searchlight
        similarity: 'pearson' or 'cosine'
        n_jobs: int - number of parallel jobs for voxel processing
        batch_size: int - number of voxels to process per batch

    Returns:
        similarity_map: nib.Nifti1Image with similarity at each voxel
    """
    logger.info(
        f"Starting searchlight similarity with cube side={2 * radius}mm, similarity={similarity}, n_jobs={n_jobs}")
    try:
        masker = NiftiMasker(mask_img=mask_img)
        masker.fit()
        logger.info(f"Masker fitted, mask shape: {masker.mask_img_.shape}")
        img1_data = masker.transform(img1)  # Cache masked data
        img2_data = masker.transform(img2)
        logger.info(f"Transformed img1 shape: {img1_data.shape}, img2 shape: {img2_data.shape}")
    except Exception as e:
        logger.error(f"Error in masker setup or transform: {e}")
        raise

    coordinates = np.argwhere(masker.mask_img_.get_fdata() > 0)
    logger.info(f"Number of voxels to process: {len(coordinates)}")
    if affine is None:
        affine = masker.mask_img_.affine

    # Get voxel size from affine matrix (assuming isotropic voxels for simplicity)
    voxel_size = np.abs(affine[0, 0])  # Assuming cubic voxels
    half_side_voxels = int(np.round(radius / voxel_size))  # Number of voxels to extend in each direction

    def compute_batch_similarity(coords, img1, img2, mask_img, half_side_voxels, batch_idx, total_batches):
        """
        Compute similarity for a batch of voxels.
        """
        try:
            batch_results = []
            img1_data = img1.get_fdata()
            img2_data = img2.get_fdata()
            mask_data = mask_img.get_fdata()
            img_shape = img1.shape[:3]

            for voxel_num, coord in enumerate(coords, 1):
                x, y, z = coord
                x_min, x_max = max(0, x - half_side_voxels), x + half_side_voxels + 1
                y_min, y_max = max(0, y - half_side_voxels), y + half_side_voxels + 1
                z_min, z_max = max(0, z - half_side_voxels), z + half_side_voxels + 1

                x_max = min(x_max, img_shape[0])
                y_max = min(y_max, img_shape[1])
                z_max = min(z_max, img_shape[2])

                # Extract mask data within the cube
                mask_cube = mask_data[x_min:x_max, y_min:y_max, z_min:z_max]
                valid_voxels = mask_cube > 0
                n_voxels = np.sum(valid_voxels)
                if n_voxels < 2:
                    batch_results.append(np.nan)
                    continue

                # Extract data from img1 and img2 within the cube
                img1_cube = img1_data[x_min:x_max, y_min:y_max, z_min:z_max][valid_voxels].ravel()
                img2_cube = img2_data[x_min:x_max, y_min:y_max, z_min:z_max][valid_voxels].ravel()

                if len(img1_cube) < 2 or np.any(np.isnan(img1_cube)) or np.any(np.isnan(img2_cube)):
                    batch_results.append(np.nan)
                    continue

                if similarity == 'pearson':
                    sim = pearsonr(img1_cube, img2_cube)[0]
                elif similarity == 'cosine':
                    sim = cosine_similarity(img1_cube.reshape(1, -1), img2_cube.reshape(1, -1))[0, 0]
                else:
                    raise ValueError("similarity must be 'pearson' or 'cosine'")
                batch_results.append(sim)

            logger.info(f"Processed batch {batch_idx}/{total_batches}, {len(coords)} voxels")
            return batch_results
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            return [np.nan] * len(coords)

    # Split coordinates into batches
    total_voxels = len(coordinates)
    batches = [coordinates[i:i + batch_size] for i in range(0, total_voxels, batch_size)]
    total_batches = len(batches)
    logger.info(f"Processing {total_voxels} voxels in {total_batches} batches of {batch_size} voxels")

    # Process batches in parallel
    batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(compute_batch_similarity)(batch_coords, img1, img2, mask_img, half_side_voxels, idx + 1, total_batches)
        for idx, batch_coords in enumerate(batches)
    )

    # Flatten results
    similarity_values = [val for batch in batch_results for val in batch]
    logger.info(
        f"Computed {len(similarity_values)} similarity values, skipped {sum(np.isnan(similarity_values))} voxels")

    # Create output similarity map
    similarity_map = np.full(masker.mask_img_.shape, np.nan)
    for i, coord in enumerate(coordinates):
        similarity_map[tuple(coord)] = similarity_values[i]

    return nib.Nifti1Image(similarity_map, masker.mask_img_.affine)


def roi_similarity(img1, img2, atlas_img, roi_labels, similarity='pearson', n_jobs=4):
    """
    Compute pairwise ROI similarities between two images.

    Parameters:
        img1: nib.Nifti1Image - first image
        img2: nib.Nifti1Image - second image
        atlas_img: nib.Nifti1Image - labeled Nifti image (ROIs > 0)
        roi_labels: list - list of valid ROI labels
        similarity: 'pearson' or 'cosine'
        n_jobs: int - number of parallel jobs for ROI pairs

    Returns:
        np.ndarray: Matrix of shape (n_rois, n_rois) with pairwise similarities
    """
    logger.info(f"Starting ROI similarity with {len(roi_labels)} ROIs, similarity={similarity}, n_jobs={n_jobs}")
    try:
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, detrend=False)
        roi_ts1 = masker.fit_transform(img1)  # Cache ROI time-series
        roi_ts2 = masker.transform(img2)
        logger.info(f"ROI time-series shape: ts1={roi_ts1.shape}, ts2={roi_ts2.shape}")
    except Exception as e:
        logger.error(f"Error in ROI masker setup or transform: {e}")
        raise

    n_rois = len(roi_labels)
    sim_matrix = np.zeros((n_rois, n_rois))

    def compute_roi_pair(i, j, ts1, ts2, similarity, pair_num, total_pairs):
        try:
            if similarity == 'pearson':
                sim = pearsonr(ts1[:, i], ts2[:, j])[0]
            elif similarity == 'cosine':
                sim = cosine_similarity(ts1[:, i].reshape(1, -1), ts2[:, j].reshape(1, -1))[0, 0]
            else:
                raise ValueError("similarity must be 'pearson' or 'cosine'")
            logger.debug(f"ROI pair {pair_num}/{total_pairs} ({i} vs {j}) similarity: {sim:.4f}")
            return sim
        except Exception as e:
            logger.error(f"Error computing ROI pair {i} vs {j} ({pair_num}/{total_pairs}): {e}")
            return np.nan

    pairs = [(i, j) for i in range(n_rois) for j in range(n_rois)]
    total_pairs = len(pairs)
    logger.info(f"Computing {total_pairs} ROI pairs")
    sim_values = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(compute_roi_pair)(i, j, roi_ts1, roi_ts2, similarity, idx + 1, total_pairs)
        for idx, (i, j) in enumerate(pairs)
    )

    for idx, (i, j) in enumerate(pairs):
        sim_matrix[i, j] = sim_values[idx]

    return sim_matrix



def load_roi_names(names_file_path: str, roi_labels: list) -> dict:
    """
    Load ROI names from a file with alternating lines:
      - Odd lines: ROI name (e.g., 'HIP-rh', '7Networks_LH_Vis_1')
      - Even lines: 'label R G B A'
    Args:
        names_file_path (str): Path to the ROI names file
        roi_labels (list): List of ROI labels (integers or strings convertible to integers)
    Returns:
        dict: Dictionary mapping integer labels to formatted ROI names
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading ROI names from {names_file_path}")

    def format_name(name: str) -> str:
        """Format ROI name based on specific patterns."""
        s = name.strip()
        m = re.match(r"^(.+)-(rh|lh)$", s, flags=re.IGNORECASE)
        if m:
            region, hemi = m.group(1), m.group(2).lower()
            return f"{hemi}_{region}"
        m = re.match(r"^7Networks_(LH|RH)_(.+)$", s, flags=re.IGNORECASE)
        if m:
            hemi = m.group(1).lower()
            rest = m.group(2)
            m_idx = re.match(r"^(.*)_(\d+)$", rest)
            if m_idx:
                base, idx = m_idx.group(1), m_idx.group(2)
                return f"{hemi}_{base}-{idx}"
            return f"{hemi}_{rest}"
        return s

    default_names = {int(l): f"combined_ROI_{int(l)}" for l in roi_labels if l is not None}

    if not os.path.exists(names_file_path):
        logger.warning(f"ROI names file not found: {names_file_path}. Using numerical labels.")
        return default_names

    intlabel_to_rawname = {}
    try:
        with open(names_file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                logger.warning(f"Incomplete pair at line {i+1}: missing label data")
                continue
            name_line = lines[i]
            nums = lines[i + 1].split()
            if not nums:
                logger.warning(f"Empty label line at {i+2}")
                continue
            try:
                label_int = int(nums[0])
                intlabel_to_rawname[label_int] = name_line
            except (ValueError, IndexError) as e:
                logger.warning(f"Invalid label format at line {i+2}: {lines[i+1]} - {e}")
                continue
    except Exception as e:
        logger.error(f"Error reading ROI names file {names_file_path}: {e}. Using numerical labels.")
        return default_names

    roi_names = {}
    for lab in roi_labels:
        try:
            lab_int = int(lab)
            raw = intlabel_to_rawname.get(lab_int)
            roi_names[lab_int] = format_name(raw) if raw is not None else f"combined_ROI_{lab_int}"
        except (ValueError, TypeError):
            logger.warning(f"Invalid ROI label: {lab}. Skipping.")
            continue

    if not roi_names:
        logger.warning("No valid ROI names loaded. Using numerical labels.")
        return default_names

    logger.info(f"Loaded {len(roi_names)} ROI names. Example: {list(roi_names.items())[:10]}")
    return roi_names


def get_roi_labels(atlas_img, atlas_name):
    logger.info(f"Extracting ROI labels from {atlas_name}")
    atlas_data = atlas_img.get_fdata()
    roi_labels = np.unique(atlas_data)[np.unique(atlas_data) > 0]
    if len(roi_labels) == 0:
        logger.error(f"No valid ROIs found in {atlas_name} atlas (all values <= 0)")
        raise ValueError(f"No valid ROIs found in {atlas_name} atlas")
    logger.info(f"Found {len(roi_labels)} ROI labels")
    return roi_labels
