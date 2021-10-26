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
