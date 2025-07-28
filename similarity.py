import numpy as np
from nilearn.image import index_img
from nilearn.maskers import NiftiSpheresMasker
from nilearn.input_data import NiftiMasker
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import nibabel as nib
from nilearn.image import load_img


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


def roi_similarity(img1_path, img2_path, atlas_img_path, similarity='pearson'):
    """
    Compute similarity between two images within each ROI defined in an atlas.

    Parameters:
        img1_path: str - path to first image
        img2_path: str - path to second image
        atlas_img_path: str - labeled Nifti image (ROIs > 0)
        similarity: 'pearson' or 'cosine'

    Returns:
        dict: {roi_label: similarity_score}
    """
    img1 = load_img(img1_path)
    img2 = load_img(img2_path)
    atlas_img = load_img(atlas_img_path)

    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
    roi_ts1 = masker.fit_transform(img1)
    roi_ts2 = masker.transform(img2)

    roi_similarities = {}
    for i in range(roi_ts1.shape[1]):
        ts1 = roi_ts1[:, i]
        ts2 = roi_ts2[:, i]
        if similarity == 'pearson':
            sim = pearsonr(ts1, ts2)[0]
        elif similarity == 'cosine':
            sim = cosine_similarity(ts1.reshape(1, -1), ts2.reshape(1, -1))[0, 0]
        else:
            raise ValueError("similarity must be 'pearson' or 'cosine'")
        roi_similarities[f'ROI_{i + 1}'] = sim

    return roi_similarities
