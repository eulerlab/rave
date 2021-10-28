from sklearn.metrics import accuracy_score as acc
from copy import deepcopy
import numpy as np


def print_result_dictionary(result_dictionary, dataset):
    type_names = ['1', '2', '3a', '3b', '4', '5t', '5o', '5i', 'X', '6', '7', '8', '9', 'R']
    print("Mean +- sd domain accuracy: {:.2f} +- {:.3f}".format(
        result_dictionary["acc_dom"].mean(), result_dictionary["acc_dom"].std()))

    print("Mean +- sd type accuracy for dataset 0: {:.2f} +- {:.3f}".format(
        result_dictionary["acc_type"].mean(), result_dictionary["acc_type"].std()))

    if len(np.unique(dataset.Y_type_test[dataset.Y_scan_test == 1])) > 1:
        # cell type labels available for dataset 1 (sim data)
        # the y type predictions are of shape cells x seeds -> for each cell,
        # get the type label that has been predicted in most of the seeds
        y_type_most_numerous = np.asarray([np.argmax(np.bincount(preds.astype(int), minlength=14))
                                           for preds in result_dictionary["y_type_test_post"]]
                                          )
        bool_mask = dataset.Y_scan_test==1
        ground_truth_type_labels = deepcopy(dataset.Y_type_test)
        ground_truth_type_labels[bool_mask] *= -1
        ground_truth_type_labels[bool_mask] -= 1
        acc_type_d1 = acc(ground_truth_type_labels[bool_mask],
                          y_type_most_numerous[bool_mask])
        print("type accuracy for dataset 1: {:.2f}".format(
            acc_type_d1))

    print("Mean +- SD; min JS for dataset 1 per type \n:")
    for i, tn in enumerate(type_names):
        print("{}:  {:.3f} +- {:.2f}; {:.3f}".format(
            tn, result_dictionary["js_depth"][:, i].mean(),
            result_dictionary["js_depth"][:, i].std(),
            result_dictionary["js_depth"][:, i].min(),
        ))

    print("domain accuracy: {:.2f} +- {:.3f}".format(result_dictionary["acc_dom"].mean(),
                                                     result_dictionary["acc_dom"].std()))

    print("Mean +- SD ari cross for dataset 1 (consistency between cell type"
          "labels assigned by 10 different type clfs): {:.3f} +- {:.2f}".format(
            result_dictionary["ari_cross"].mean(), result_dictionary["ari_cross"].std()))


def get_sim_ground_truth_labels(y_type_train, y_scan_train, y_type_val,
                                y_scan_val, y_type_test, y_scan_test):
    """
    Only for simulated data with ground truth type labels for dataset B:
    take type labels ranging from -1 to -14 (to be compatible with the
    dataset function) and map them to labels ranging from 0 to 13 (-1 -> 0,
    -2 -> 1, ..., -14 -> 13)
    :param y_type_test:
    :param y_scan_test:
    :return:
    """
    y_type_test_gt = deepcopy(y_type_test)
    y_type_test_gt[y_scan_test == 1] *= -1
    y_type_test_gt[y_scan_test == 1] -= 1

    y_type_val_gt = deepcopy(y_type_val)
    y_type_val_gt[y_scan_val == 1] *= -1
    y_type_val_gt[y_scan_val == 1] -= 1

    y_type_train_gt = deepcopy(y_type_train)
    y_type_train_gt[y_scan_train == 1] *= -1
    y_type_train_gt[y_scan_train == 1] -= 1
    assert np.all(y_type_test_gt[y_type_test==-14] == 13), "Conversion failed"
    return y_type_train_gt, y_type_val_gt, y_type_test_gt
