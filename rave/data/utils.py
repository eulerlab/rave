import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold


def normalize_trace(data):
    tmp = data.copy()
    tmp -= np.mean(tmp, 1, keepdims=True)
    tmp /= np.std(tmp, 1, keepdims=True)
    return tmp


def normalize_feature(data):
    tmp = data.copy()
    tmp -= np.mean(tmp, 0, keepdims=True)
    tmp /= np.std(tmp, 0, keepdims=True)
    return tmp


def combine_data_rgc(rgc_old, rgc_new):
    all_old = np.concatenate([
        normalize_trace(rgc_old['chirp']),
        normalize_trace(rgc_old['bar']),
        # normalize_feature(rgc_old['roi_size']),
        # normalize_feature(rgc_old['dsp'])
    ], 1)
    all_new = np.concatenate([
        normalize_trace(rgc_new['chirp']),
        normalize_trace(rgc_new['bar']),
        # normalize_feature(rgc_new['roi_size']),
        # normalize_feature(rgc_new['dsp'])
    ], 1)
    scan_label = np.concatenate([
        np.zeros(all_old.shape[0]), np.ones(all_new.shape[0])
    ])
    type_label = np.zeros_like(scan_label)
    type_label[scan_label == 1] = -1  #set all new type labels to -1
    type_label[scan_label == 0] = np.squeeze(rgc_old["label"]) - 1
    return np.concatenate([all_old, all_new], 0), all_old, all_new, scan_label, type_label


def combine_data_bc(bc_old, bc_new):
    lc_old = bc_old['local_chirp']
    gc_old = bc_old['global_chirp']
    roi_depth = bc_old['ipl_depth']
    label = bc_old['label']
    kl_lc = bc_new['local_chirp']
    kl_gc = bc_new['global_chirp']
    kl_all_depths = bc_new['ipl_depth']
    all_depths = np.concatenate([
        np.squeeze(roi_depth), np.squeeze(kl_all_depths)
    ])

    all_old = np.concatenate([
        normalize_trace(gc_old),
        normalize_trace(lc_old),
    ], 1)
    all_new = np.concatenate([
        normalize_trace(kl_gc),
        normalize_trace(kl_lc),
    ], 1)
    scan_label = np.concatenate([
        np.zeros(all_old.shape[0]), np.ones(all_new.shape[0])
    ])
    type_label = np.zeros_like(scan_label)
    type_label[scan_label == 1] = -1  # set all new type labels to -1
    type_label[scan_label == 0] = np.squeeze(label) - 1
    return (np.concatenate([all_old, all_new], 0), all_old, all_new,
            scan_label, type_label, all_depths)


def get_splits(X, Y=None, num_split=5, seed=42):
    np.random.seed(seed)
    if Y is not None:
        skf = StratifiedKFold(
            n_splits=num_split, shuffle=True, random_state=seed)
        dum = skf.split(X, Y)
    else:
        skf = KFold(
            n_splits=num_split, shuffle=True, random_state=seed)
        dum = skf.split(X)
    return list(dum)