import pickle
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.segmentation import watershed
from scipy import ndimage
from math import hypot
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def write_config(cfg: dict, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(cfg, f)
        
        
def load_config(load_path: str):
    with open(load_path, "rb") as f:
        return pickle.load(f)
        
        
EPS = 10**(-2)
def F1Score(metrics):
    # compute performance measure for the current quantile filter
    tot_tp_test = metrics["TP"].sum()
    tot_fp_test = metrics["FP"].sum()
    tot_fn_test = metrics["FN"].sum()
    tot_abs_diff = abs(metrics["Target_count"] - metrics["Predicted_count"])
    tot_perc_diff = (metrics["Predicted_count"] -
                     metrics["Target_count"])/(metrics["Target_count"]+EPS)
    accuracy = (tot_tp_test + EPS)/(tot_tp_test +
                                      tot_fp_test + tot_fn_test + EPS)
    precision = (tot_tp_test + EPS)/(tot_tp_test + tot_fp_test + EPS)
    recall = (tot_tp_test + EPS)/(tot_tp_test + tot_fn_test + EPS)
    F1_score = 2*precision*recall/(precision + recall)
    MAE = tot_abs_diff.mean()
    MedAE = tot_abs_diff.median()
    MPE = tot_perc_diff.mean()

    return(F1_score, MAE, MedAE, MPE, accuracy, precision, recall)
    
    
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
    

def plot_MAE(test_metrics):
    '''Plot mean absolute error distribution based on pandas dataframe. Return None.'''
    
    sns.set_style('whitegrid')
    
    # N.B. the dataframe must contain true and predicted counts in two columns named as follows
    mae_list = list(abs(test_metrics.Target_count - test_metrics.Predicted_count))
    
    fig = plt.figure(figsize=(15,6))
    suptit = plt.suptitle("Absolute Error Distribution")
    
    color = 'blue'
    
    MAX = max(mae_list)
    
    sb = plt.subplot(1,2,1)
    box=plt.boxplot(mae_list,vert=0,patch_artist=True, labels=[""])
    plt.xlabel("Absolute Error")
    plt.ylabel("MAE")
    
    t = plt.text(2, 1.15, 'Mean Abs. Err.: {:.2f}\nMedian Abs. Err.: {:.2f}\nStd. Dev.: {:.2f}'.format(
    np.array(mae_list).mean(), np.median(np.array(mae_list)), np.array(mae_list).std()),
            bbox={'facecolor': color, 'alpha': 0.5, 'pad': 5})
    
    for patch, color in zip(box['boxes'], color):
        patch.set_facecolor(color)
    _ = plt.xticks(range(0,MAX, 5))
    
    sb = plt.subplot(1,2,2)
    
    dens = sns.distplot(np.array(mae_list), bins = 20, color=color, hist=True, norm_hist=False)
    _ = plt.xlim(0,MAX)
    _ = dens.axes.set_xticks(range(0,max(mae_list),5))
    _ = plt.axvline(np.mean(mae_list), 0,1, color="firebrick", label = "Mean Abs. Err.")
    _ = plt.axvline(np.median(mae_list), 0,1, color="goldenrod", label = "Median Abs. Err.")
    
    # Plot formatting
    leg = plt.legend(title="Model")
    xlab = plt.xlabel('Absolute Error')
    ylab = plt.ylabel('Density')
    
    plt.show()
    return(None)
    
    
def plot_MPE(test_metrics):
    '''Plot mean percentage error distribution based on pandas dataframe. Return None.'''
    
    sns.set_style('whitegrid')
    
    # N.B. the dataframe must contain true and predicted counts in two columns named as follows
    mpe_list = list((test_metrics.Predicted_count - test_metrics.Target_count)/(test_metrics.Target_count + EPS))

    fig = plt.figure(figsize=(15,6))
    suptit = plt.suptitle("Percentage Error Distribution")
    
    color = 'green'
    
    MIN = min(mpe_list)
    MAX = max(mpe_list)
    
    sb = plt.subplot(1,2,1)
    box=plt.boxplot(mpe_list,vert=0,patch_artist=True, labels=[""])
    plt.xlabel("Percentage Error")
    plt.ylabel("MPE")
    
    t = plt.text(-0.9, 1.15, 'Mean Perc. Err.: {:.2f}\nMedian Perc. Err.: {:.2f}\nStd. Dev.: {:.2f}'.format(
    np.array(mpe_list).mean(), np.median(np.array(mpe_list)), np.array(mpe_list).std()),
            bbox={'facecolor': color, 'alpha': 0.5, 'pad': 5})
    
    for patch, color in zip(box['boxes'], color):
        patch.set_facecolor(color)
    # _ = plt.xticks(range(0,MAX, 5))
    
    sb = plt.subplot(1,2,2)
    
    dens = sns.distplot(np.array(mpe_list), bins = 20, color=color, hist=True, norm_hist=False)
    _ = plt.xlim(MIN,MAX)
    # _ = dens.axes.set_xticks(range(0,max(mae_list),5))
    _ = plt.axvline(np.mean(mpe_list), 0,1, color="firebrick", label = "Mean Perc. Err.")
    _ = plt.axvline(np.median(mpe_list), 0,1, color="goldenrod", label = "Median Perc. Err.")
    
    # Plot formatting
    leg = plt.legend(title="Model")
    xlab = plt.xlabel('Percentage Error')
    ylab = plt.ylabel('Density')
    
    plt.show()
    return(None)
