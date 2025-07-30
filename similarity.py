import numpy as np
from nilearn.image import index_img
from nilearn.maskers import NiftiSpheresMasker
from nilearn.input_data import NiftiMasker
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import nibabel as nib
from nilearn.image import load_img
import os


def searchlight_similarity(img1_path, img2_path, radius=6, affine=None, mask_img=None, similarity='pearson'):
    """
    Compute voxel-wise similarity between two 3D or 4D images using a searchlight approach.

    Parameters:
        img1_path: str or nib.Nifti1Image - first image (3D or 4D)
        img2_path: str or nib.Nifti1Image - second image (must match img1 shape)
        radius: int - radius of the searchlight sphere in mm
        affine: optional affine to transform coordinates (default: from image)
        mask_img: binary Nifti image for where to apply searchlight
        similarity: 'pearson' or 'cosine'

    Returns:
        similarity_map: 3D numpy array with similarity at each voxel
    """
    img1 = load_img(img1_path)
    img2 = load_img(img2_path)

    masker = NiftiMasker(mask_img=mask_img)
    img1_data = masker.fit_transform(img1)
    img2_data = masker.transform(img2)

    coordinates = np.argwhere(masker.mask_img_.get_fdata() > 0)
    world_coords = nib.affines.apply_affine(masker.affine_, coordinates)

    similarity_values = np.zeros(coordinates.shape[0])

    for i, coord in enumerate(world_coords):
        sphere_masker = NiftiSpheresMasker([coord], radius=radius, detrend=False, standardize=False)
        try:
            sphere_ts1 = sphere_masker.fit_transform(img1)
            sphere_ts2 = sphere_masker.transform(img2)
            if sphere_ts1.shape[1] < 2:
                similarity_values[i] = np.nan
                continue

            if similarity == 'pearson':
                sim = pearsonr(sphere_ts1.ravel(), sphere_ts2.ravel())[0]
            elif similarity == 'cosine':
                sim = cosine_similarity(sphere_ts1, sphere_ts2)[0, 0]
            else:
                raise ValueError("similarity must be 'pearson' or 'cosine'")
            similarity_values[i] = sim
        except:
            similarity_values[i] = np.nan

    # Fill 3D image with similarity values
    similarity_map = np.full(masker.mask_img_.shape, np.nan)
    for i, coord in enumerate(coordinates):
        similarity_map[tuple(coord)] = similarity_values[i]

    return similarity_map


from nilearn.maskers import NiftiLabelsMasker


def roi_similarity(img1_path, img2_path, atlas_img, roi_labels, similarity='pearson'):
    """
    Compute pairwise ROI similarities between two images.

    Parameters:
        img1_path: str or nib.Nifti1Image - first image
        img2_path: str or nib.Nifti1Image - second image
        atlas_img: nib.Nifti1Image - labeled Nifti image (ROIs > 0)
        roi_labels: list - list of valid ROI labels
        similarity: 'pearson' or 'cosine'

    Returns:
        np.ndarray: Matrix of shape (n_rois, n_rois) with pairwise similarities
    """
    img1 = load_img(img1_path)
    img2 = load_img(img2_path)

    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, detrend=False)
    roi_ts1 = masker.fit_transform(img1)
    roi_ts2 = masker.transform(img2)

    n_rois = len(roi_labels)
    sim_matrix = np.zeros((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(n_rois):
            ts1 = roi_ts1[:, i]
            ts2 = roi_ts2[:, j]
            if similarity == 'pearson':
                sim = pearsonr(ts1, ts2)[0]
            elif similarity == 'cosine':
                sim = cosine_similarity(ts1.reshape(1, -1), ts2.reshape(1, -1))[0, 0]
            else:
                raise ValueError("similarity must be 'pearson' or 'cosine'")
            sim_matrix[i, j] = sim

    return sim_matrix


import os
import re

def load_roi_names(names_file_path, roi_labels):
    """
    File format: alternating lines
      - Odd lines: ROI name (e.g., 'HIP-rh', '7Networks_LH_Vis_1')
      - Even lines: 'label R G B A'
    Returns: dict with **int** keys and formatted names:
      - 'REGION-rh' / 'REGION-lh'  -> 'rh_REGION' / 'lh_REGION'
      - '7Networks_LH_X_Y_1'       -> 'lh_X_Y-1' (drop '7Networks_', last '_' -> '-' before index)
    """

    def format_name(name: str) -> str:
        s = name.strip()

        # Case A: Subcortical 'REGION-rh' / 'REGION-lh' -> 'rh_REGION' / 'lh_REGION'
        m = re.match(r"^(.+)-(rh|lh)$", s, flags=re.IGNORECASE)
        if m:
            region, hemi = m.group(1), m.group(2).lower()
            return f"{hemi}_{region}"

        # Case B: Schaefer '7Networks_(LH|RH)_(...)'
        m = re.match(r"^7Networks_(LH|RH)_(.+)$", s)
        if m:
            hemi = m.group(1).lower()   # 'lh' or 'rh'
            rest = m.group(2)
            # If ends with '_<number>', convert last '_' to '-' before the index
            m_idx = re.match(r"^(.*)_(\d+)$", rest)
            if m_idx:
                base, idx = m_idx.group(1), m_idx.group(2)
                return f"{hemi}_{base}-{idx}"
            else:
                return f"{hemi}_{rest}"

        # Default: leave as-is (rare)
        return s

    # If file missing: fallback with int keys
    if not os.path.exists(names_file_path):
        print(f"ROI names file not found: {names_file_path}. Using numerical labels.")
        return {int(l): f"combined_ROI_{int(l)}" for l in roi_labels}

    # Parse {int_label: raw_name} from alternating lines
    intlabel_to_rawname = {}
    try:
        with open(names_file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for i in range(0, len(lines), 2):
            name_line = lines[i]
            if i + 1 >= len(lines):
                continue
            nums = lines[i + 1].split()
            try:
                label_int = int(nums[0])
            except (IndexError, ValueError):
                continue
            intlabel_to_rawname[label_int] = name_line

    except Exception as e:
        print(f"Error reading ROI names file: {e}. Using numerical labels.")
        return {int(l): f"combined_ROI_{int(l)}" for l in roi_labels}

    # Build mapping with **int** keys and formatted names
    roi_names = {}
    for lab in roi_labels:
        lab_int = int(lab)  # ensures Python int, not np.float64
        raw = intlabel_to_rawname.get(lab_int)
        roi_names[lab_int] = format_name(raw) if raw is not None else f"combined_ROI_{lab_int}"

    # Debug example
    preview = list(roi_names.items())[:10]
    print(f"Loaded {len(roi_names)} ROI names. Example: {preview}")
    return roi_names



def get_roi_labels(atlas_img, atlas_name):
    atlas_data = atlas_img.get_fdata()
    roi_labels = np.unique(atlas_data)[np.unique(atlas_data) > 0]
    if len(roi_labels) == 0:
        raise ValueError(f"No valid ROIs found in {atlas_name} atlas (all values <= 0)")
    return roi_labels
