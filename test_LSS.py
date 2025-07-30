#%%
import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations, product
from nilearn.image import load_img, index_img, new_img_like
import nibabel as nib
from similarity import searchlight_similarity
from similarity import load_roi_names
from similarity import get_roi_labels
#%%
combined_atlas_path = '/Users/xiaoqianxiao/tool/parcellation/Tian/3T/Cortex-Subcortex/MNIvolumetric/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_3T_MNI152NLin2009cAsym_2mm.nii.gz'  # Update with actual path
combined_atlas = load_img(combined_atlas_path)
# Validate and extract ROI labels

combined_roi_labels = get_roi_labels(combined_atlas, 'Schaefer+Tian')

roi_names_file = '/Users/xiaoqianxiao/tool/parcellation/Tian/3T/Cortex-Subcortex/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_label.txt'
roi_names = load_roi_names(roi_names_file, combined_roi_labels)