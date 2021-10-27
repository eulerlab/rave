import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from rave.data.dataloading import data_loader_bc, data_loader_sim
from rave.data.utils import combine_data_bc, get_splits


def get_bc_data(data_dir, franke_file_name="bc_franke.pickle",
                szatko_file_name="bc_szatko.pickle", verbose=True):
    bc_old, bc_new = data_loader_bc(data_dir, franke_file_name, szatko_file_name)
    X, _, _, scan_label, type_label, ipl_depths = combine_data_bc(bc_old, bc_new)
    dataset = Dataset(X, Y_scan=scan_label, Y_type=type_label,
                      ipl_depths=ipl_depths, verbose=verbose)
    return dataset


def get_bc_sim_data(sim_data, ipl_depths, random_seed=8000):
    """
    Create a custom Dataset from simulated data
    :param sim_data: dictionary with key preds_local, preds_global (predictions
    to old chirp version) and preds_local_new, preds_global_new (predictions
    to new chirp version); values are of shape
    n_simulated_neurons (1000) x n_types (14) x time (960)
    """
    return BipCellDataset(*data_loader_sim(sim_data, ipl_depths, random_seed))


class BipCellDataset(Dataset):
    def __init__(
            self,
            X,
            Y_scan,
            ipl_depths,
            Y_type=None,  # -1 if not available
            seed=42,
            k=5,  # k-fold cross-validation
            test_frac=10,  # use 1/test_frac for testing
            verbose=True,
    ):
        self.seed = seed
        self.k = k
        self.N, self.D = X.shape
        self.scan_label = np.unique(Y_scan)
        self.verbose = verbose

        self.X = X.copy()
        self.Y_scan = Y_scan.copy()

        if Y_type is not None:
            if self.verbose:
                print('Splitting by Type Labels, all type labels must be in first scan')
            self.Y_type = Y_type.copy()
            typed_ind = Y_type > -1
            untyped_ind = Y_type <= -1
            # check that all type labels are in first scan:
            assert np.all(Y_scan[typed_ind] == 0)
            self.type_label = np.unique(Y_type[typed_ind])
            # split by types
            typed_train_ind, typed_test_ind = get_splits(
                X[typed_ind], Y_scan[typed_ind],
                num_split=test_frac, seed=seed)[0]
            untyped_train_ind, untyped_test_ind = get_splits(
                X[untyped_ind], num_split=test_frac, seed=seed)[0]
            self.X_train = np.concatenate([
                X[typed_ind][typed_train_ind],
                X[untyped_ind][untyped_train_ind]
            ], 0).copy()
            self.Y_scan_train = np.concatenate([
                Y_scan[typed_ind][typed_train_ind],
                Y_scan[untyped_ind][untyped_train_ind]
            ], 0).copy()
            self.Y_type_train = np.concatenate([
                Y_type[typed_ind][typed_train_ind],
                Y_type[untyped_ind][untyped_train_ind]
            ], 0).copy()
            self.ipl_depth_train = np.concatenate([
                ipl_depths[typed_ind][typed_train_ind],
                ipl_depths[untyped_ind][untyped_train_ind]
            ], 0).copy()
            self.X_test = np.concatenate([
                X[typed_ind][typed_test_ind],
                X[untyped_ind][untyped_test_ind]
            ], 0).copy()
            self.Y_scan_test = np.concatenate([
                Y_scan[typed_ind][typed_test_ind],
                Y_scan[untyped_ind][untyped_test_ind]
            ], 0).copy()
            self.Y_type_test = np.concatenate([
                Y_type[typed_ind][typed_test_ind],
                Y_type[untyped_ind][untyped_test_ind]
            ], 0).copy()
            self.ipl_depth_test = np.concatenate([
                ipl_depths[typed_ind][typed_test_ind],
                ipl_depths[untyped_ind][untyped_test_ind]
            ], 0).copy()

            # k-fold cross-validation splits
            typed_ind = self.Y_type_train > -1
            untyped_ind = self.Y_type_train <= -1
            typed_val_splits = get_splits(
                self.X_train[typed_ind], self.Y_scan_train[typed_ind],
                num_split=k, seed=seed)
            untyped_val_splits = get_splits(
                self.X_train[untyped_ind], num_split=k, seed=seed)
            num_typed = np.sum(typed_ind)
            self.val_splits = []
            for t, u in zip(typed_val_splits, untyped_val_splits):
                self.val_splits.append([
                    np.concatenate([t[0], u[0] + num_typed], 0),
                    np.concatenate([t[1], u[1] + num_typed], 0)
                ])
            if self.verbose:
                print('%s-Fold Cross-Validation Train=%s / Val=%s' % (
                    k,
                    self.val_splits[0][0].shape[0],
                    self.val_splits[0][1].shape[0]))

            # compute baseline accuracies
            classes, counts = np.unique(self.Y_scan_train, return_counts=True)
            biggest_class = classes[np.argmax(counts)]
            if self.verbose:
                print('baseline adversarial accuracy (biggest class %s)' % biggest_class,
                      np.mean(self.Y_scan_train == biggest_class))

            classes, counts = np.unique(
                self.Y_type_train[typed_ind], return_counts=True)
            biggest_class = classes[np.argmax(counts)]
            if self.verbose:
                print('baseline cell type accuracy (biggest class %s)' % biggest_class,
                      np.mean(self.Y_type_train[typed_ind] == biggest_class))

        else:
            self.Y_type = None
            if self.verbose:
                print('Splitting by Scan Labels')
            train_ind, test_ind = get_splits(
                X, Y_scan, num_split=test_frac, seed=seed)[0]
            self.X_train = X[train_ind].copy()
            self.Y_scan_train = Y_scan[train_ind].copy()
            self.X_test = X[test_ind].copy()
            self.Y_scan_test = Y_scan[test_ind].copy()
            if self.verbose:
                print('Training Data: %s, Testing Data: %s' % (
                    self.X_train.shape, self.X_test.shape))

            # k-fold cross-validation splits
            self.val_splits = get_splits(
                self.X_train, self.Y_scan_train,
                num_split=k, seed=seed)
            if self.verbose:
                print('%s-Fold Cross-Validation Train=%s / Val=%s' % (
                    k,
                    self.val_splits[0][0].shape[0],
                    self.val_splits[0][1].shape[0]))

            # compute baseline adversarial accuracy
            classes, counts = np.unique(self.Y_scan_train, return_counts=True)
            biggest_class = classes[np.argmax(counts)]
            if self.verbose:
                print('baseline adversarial accuracy (biggest class %s)' % biggest_class,
                      np.mean(self.Y_scan_train == biggest_class))

    def get_split(self, split=0, device="cpu"):
        train_ind, val_ind = self.val_splits[split]

        # train
        x_train = torch.tensor(
            self.X_train[train_ind], dtype=torch.float32).to(device)
        y_scan_train = torch.tensor(
            self.Y_scan_train[train_ind], dtype=torch.long).to(device)

        # validation
        x_val = torch.tensor(
            self.X_train[val_ind], dtype=torch.float32).to(device)
        y_scan_val = torch.tensor(
            self.Y_scan_train[val_ind], dtype=torch.long).to(device)

        output = [x_train, y_scan_train, x_val, y_scan_val]

        if self.Y_type is not None:
            y_type_train = torch.tensor(
                self.Y_type_train[train_ind], dtype=torch.long).to(device)
            y_type_val = torch.tensor(
                self.Y_type_train[val_ind], dtype=torch.long).to(device)

            output += [y_type_train, y_type_val]

        return output

    def get_ipl_split(self, split=0):
        train_ind, val_ind = self.val_splits[split]
        return self.ipl_depth_train[train_ind], self.ipl_depth_train[val_ind]

    def __len__(self):
        train_ind, val_ind = self.val_splits[0]
        return len(train_ind)

    def __getitem__(self, idx):
        train_idx, val_idx = self.val_splits[0]
        if self.Y_type_train is not None:
            return torch.as_tensor(self.X_train[train_idx[idx]],
                                   device="cuda"), \
                   torch.as_tensor(self.Y_scan_train[train_idx[idx]],
                                   dtype=torch.long, device="cuda"), \
                   torch.as_tensor(self.Y_type_train[train_idx[idx]],
                                   dtype=torch.long, device="cuda")
        else:
            return self.X_train[train_idx[idx]].to("cuda"), \
                   self.Y_scan_train[train_idx[idx]].to("cuda")

    def get_split_numpy(self, split=0):
        train_ind, val_ind = self.val_splits[split]

        # train
        x_train = self.X_train[train_ind]
        y_scan_train = self.Y_scan_train[train_ind]

        # validation
        x_val = self.X_train[val_ind]
        y_scan_val = self.Y_scan_train[val_ind]

        output = [x_train, y_scan_train, x_val, y_scan_val]

        if self.Y_type is not None:
            y_type_train = self.Y_type_train[train_ind]
            y_type_val = self.Y_type_train[val_ind]

            output += [y_type_train, y_type_val]

        return output
