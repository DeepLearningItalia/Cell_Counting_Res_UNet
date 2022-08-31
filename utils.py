import pickle
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.segmentation import watershed
from scipy import ndimage
from math import hypot
import numpy as np
import torch


def write_config(cfg: dict, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(cfg, f)
        
        
def load_config(load_path: str):
    with open(load_path, "rb") as f:
        return pickle.load(f)
        
        
def thresh_format(pred, t=0.5):
    tr = torch.zeros_like(pred)
    tr[:, 0, :, :] = tr[:, 0, :, :].add(t-0.5)
    tr[:, 1, :, :] = tr[:, 1, :, :].add(0.5-t)
    return pred + tr


def mask_post_processing(thresh_image, area_threshold=50, min_obj_size=10, max_dist=8, foot=8):

    # Find object in predicted image
    labels_pred, nlabels_pred = ndimage.label(thresh_image)
    
    # remove holes
    processed = remove_small_holes(labels_pred, area_threshold=area_threshold, connectivity=1,
                                   in_place=False)
    
    # remove small objects
    processed = remove_small_objects(
        processed, min_size=min_obj_size, connectivity=1, in_place=False)
    labels_bool = processed.astype(bool)

    
    # watershed
    distance = ndimage.distance_transform_edt(processed)

    maxi = ndimage.maximum_filter(distance, size=max_dist, mode='constant')
    local_maxi = peak_local_max(maxi, indices=False, footprint=np.ones((foot, foot)),
                                exclude_border=False,
                                labels=labels_bool)

    local_maxi = remove_small_objects(
        local_maxi, min_size=5, connectivity=1, in_place=False)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=labels_bool,
                       compactness=1, watershed_line=True)

    return(labels.astype("uint8")*255)
