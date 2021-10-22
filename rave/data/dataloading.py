import os
import pickle as pkl
import numpy as np
from rave.data.utils import normalize_trace


def data_loader_bc(
        data_directory,
        franke_file="bc_franke_NEW.pkl",
        szatko_file="bc_szatko_NEW.pkl"
):
    with open(os.path.join(data_directory, franke_file), 'rb') as handle:
        bc_franke = pkl.load(handle)
    with open(os.path.join(data_directory, szatko_file), 'rb') as handle:
        bc_szatko = pkl.load(handle)
    return bc_franke, bc_szatko


def data_loader_sim(sim_data, ipl_depths, random_seed=8000):
    """
    Create a custom Dataset from simulated data
    :param sim_data: dictionary with key preds_local, preds_global (predictions
    to old chirp version) and preds_local_new, preds_global_new (predictions
    to new chirp version); values are of shape
    n_simulated_neurons (1000) x n_types (14) x time (960)
    :return:
    """
    ### reshape from neurons x types into neurons*types
    ipl_depths = ipl_depths.reshape(-1, order="F")
    gc_old = sim_data["preds_global"].reshape(-1, 960, order="F")
    np.random.seed(random_seed)
    shuffling_idx = np.arange(gc_old.shape[0], dtype=int)
    np.random.shuffle(shuffling_idx)
    lc_old = sim_data["preds_local"].reshape(-1, 960, order="F")
    gc_new = sim_data["preds_global_new"].reshape(-1, 960, order="F")
    lc_new = sim_data["preds_local_new"].reshape(-1, 960, order="F")
    type_labels = np.arange(0, 14, 1)
    type_labels = np.repeat(type_labels, 1000)
    type_labels_new = np.arange(1, 15, 1)
    type_labels_new = np.repeat(type_labels_new, 1000)
    type_labels_new *= -1
    all_old = np.concatenate([
        normalize_trace(gc_old[shuffling_idx]),
        normalize_trace(lc_old[shuffling_idx]),
    ], 1)
    all_new = np.concatenate([
        normalize_trace(gc_new[shuffling_idx]),
        normalize_trace(lc_new[shuffling_idx]),
    ], 1)
    scan_labels = np.concatenate([
        np.zeros(all_old.shape[0]), np.ones(all_new.shape[0])
    ])
    type_labels = np.concatenate([type_labels[shuffling_idx],
                                  type_labels_new[shuffling_idx]])
    X = np.concatenate([all_old, all_new], 0)
    all_ipl_depths = np.concatenate([ipl_depths[shuffling_idx], ipl_depths[shuffling_idx]])
    return X, scan_labels, all_ipl_depths, type_labels