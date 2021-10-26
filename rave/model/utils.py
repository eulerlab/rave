import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV


def get_optimizers(model, optimizer, lr_model, lr_discriminator, weight_decay):
    print(type(model))
    model_params = [
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': model.type_classifier.parameters()},
    ]
    if optimizer == 'adam':
        optimizer_model = optim.Adam(
            model_params, lr=lr_model, betas=(0.5, 0.999), weight_decay=weight_decay
        )
        optimizer_discriminator = optim.Adam(
            model.scan_classifier.parameters(),
            lr=lr_discriminator, betas=(0.5, 0.999), weight_decay=weight_decay
        )
    elif optimizer == 'sgd':
        raise ValueError('Not used in a while, needs updating.')
        optimizer_model = torch.optim.SGD(model_params, lr=1e-1)
        optimizer_discriminator = torch.optim.SGD(
            model.discriminator.parameters(), lr=1e-2)

    return optimizer_model, optimizer_discriminator


def check_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                print('NaN in Gradient, skipping step', name)
                return False
    return True


def get_loss(dataset, x, y_scan, y_type, x_rec, y_scan_hat, y_type_hat,
             weight_recon, weight_scan, weight_type,
             reconstruction_loss, type_classification_loss,
             domain_classification_loss, adversarial_training):
    recon_loss = 0
    type_loss = 0
    domain_loss = 0
    for d in dataset.scan_label:
        idxs = y_scan == d
        if torch.any(idxs):
            recon_loss += reconstruction_loss(x_rec[idxs, :], x[idxs, :])
            domain_loss += domain_classification_loss(y_scan_hat[idxs], y_scan[idxs])
    for c in dataset.type_label:
        idxs = y_type == c
        if torch.any(idxs):
            type_loss += type_classification_loss(y_type_hat[idxs], y_type[idxs])
    loss = weight_recon * recon_loss + weight_type * type_loss
    # print(recon_loss.detach().cpu().numpy(), type_loss.detach().cpu().numpy(),
    #       domain_loss.detach().cpu().numpy())
    if adversarial_training:
        loss -= weight_scan * domain_loss
    return loss, recon_loss, type_loss


def get_adv_loss(dataset, y_scan, y_scan_hat, domain_classification_loss):
    domain_loss = 0
    for d in dataset.scan_label:
        idxs = y_scan == d
        if torch.any(idxs):
            domain_loss += domain_classification_loss(y_scan_hat[idxs], y_scan[idxs])
    return domain_loss


def var_exp(total_variance, x, x_rec):
    """
    Calculates the mean variance explained across samples (cells)
    :param total_variance: torch Tensor of shape # samples, 1
    :param x: torch Tensor of shape # samples x
    :param x_rec:
    :return:
    """
    mse = torch.sum(mse_loss(x_rec, x, reduction="none"), dim=1)
    return torch.mean(1 - mse/total_variance).item()


def correlate(a, b):
    x = a.detach().cpu().numpy()
    y = b.detach().cpu().numpy()
    corrs = [spearmanr(i, j) for i, j in zip(x, y)]
    return np.mean(corrs)


def least_squares_decoding(z_train_, x_train_, z_val_, x_val_):
    z_train = z_train_.detach().cpu().numpy()
    x_train = x_train_.detach().cpu().numpy()
    z_val = z_val_.detach().cpu().numpy()
    x_val = x_val_.detach().cpu().numpy()
    X = z_train.copy()
    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(x_train))
    x_val_ = z_val.dot(beta)
    res = x_val - x_val_
    var_exp = 1 - np.sum(res**2) / np.sum(x_val**2)
    corrs = [spearmanr(i, j) for i, j in zip(x_val, x_val_)]
    return var_exp, np.mean(corrs)


def final_evaluate(net, dataset, x_train, x_val, total_variance_val, y_scan_train,
                   y_type_train, y_scan_val, y_type_val):
    # make predictions
    net.eval()
    z, y_scan, y_type, x_rec = net(x_train)
    z_val, y_scan_val_pred, y_type_val_pred, x_rec_val = net(x_val)

    res_val = (x_val - x_rec_val) ** 2
    corr_val = correlate(x_val, x_rec_val)
    ve_val = torch.mean(
        1 - torch.sum(res_val, 1) / total_variance_val).item()
    _, predicted = torch.max(y_scan_val_pred, 1)
    accuracy_scan = (predicted == y_scan_val).double().mean().item()
    _, predicted = torch.max(y_type_val_pred, 1)
    accuracy_type = (predicted == y_type_val).double().mean().item()

    # also do least_squares_decoding
    lsq_var_exp, lsq_corrs = least_squares_decoding(z, x_train, z_val, x_val)

    # use latents for all models:
    model_output_train = z.detach().cpu().numpy()
    model_output_val = z_val.detach().cpu().numpy()
    y_scan_train = y_scan_train.detach().cpu().numpy()
    y_scan_val = y_scan_val.detach().cpu().numpy()
    y_type_train = y_type_train.detach().cpu().numpy()
    y_type_val = y_type_val.detach().cpu().numpy()

    assert not net.training, "net is in training mode"

    # Grid search
    train_idx, val_idx = dataset.val_splits[0]
    clf_input = np.zeros(
        (model_output_train.shape[0] + model_output_val.shape[0],
         model_output_train.shape[1]))
    clf_scan_target = np.zeros(
        model_output_train.shape[0] + model_output_val.shape[0])
    clf_type_target = np.zeros(
        model_output_train.shape[0] + model_output_val.shape[0])
    clf_input[train_idx] = model_output_train
    clf_input[val_idx] = model_output_val
    clf_scan_target[train_idx] = y_scan_train
    clf_scan_target[val_idx] = y_scan_val
    # get Franke idxs into train+val dataset
    franke_only = np.where(clf_scan_target == 0)[0]
    # get those training indexes that are Franke samples
    train_idx_franke = list(set(franke_only).intersection(set(train_idx)))
    # get those validation indexes that are Franke samples
    val_idx_franke = list(set(franke_only).intersection(set(val_idx)))
    clf_type_target[train_idx] = y_type_train
    clf_type_target[val_idx] = y_type_val

    # Train classifiers first
    scan_clf_param_grid = dict(
        n_estimators=[5, 10, 20, 30], max_depth=[5, 10, 15, 20, None],
        ccp_alpha=[0, 0.001, 0.01], max_samples=[0.5, 0.7, 0.9, 1]
    )
    grid_search_scan = GridSearchCV(estimator=RFC(),
                                    param_grid=scan_clf_param_grid,
                                    cv=[dataset.val_splits[0]], n_jobs=1,
                                    refit=False)
    grid_search_scan.fit(clf_input, clf_scan_target)
    scan_classifier = RFC(**grid_search_scan.best_params_)
    scan_classifier.fit(model_output_train, y_scan_train)
    final_scan_acc = scan_classifier.score(model_output_val, y_scan_val)

    ### train type classifier
    grid_search_type = GridSearchCV(estimator=RFC(),
                                    param_grid=scan_clf_param_grid,
                                    cv=[[train_idx_franke, val_idx_franke]],
                                    n_jobs=1, refit=False)
    grid_search_type.fit(clf_input, clf_type_target)
    type_classifier = RFC(**grid_search_type.best_params_)
    type_classifier.fit(
        model_output_train[y_scan_train == 0], y_type_train[y_scan_train == 0]
    )
    final_type_acc = type_classifier.score(
        model_output_val[y_scan_val == 0], y_type_val[y_scan_val == 0]
    )
    res_dict = dict(val_corr=corr_val,
                    var_exp=ve_val,
                    type_acc=accuracy_type,
                    adv_acc=accuracy_scan,
                    final_type_acc=final_type_acc,
                    final_scan_acc=final_scan_acc,
                    lsq_var_exp=lsq_var_exp,
                    lsq_corrs=lsq_corrs)
    return res_dict, grid_search_type, grid_search_scan


class ModelOutputs:
    def get_model_outputs(self, model, my_dataset, device="cuda"):
        if model.training:
            "Model still in training mode! Running model.eval() to set training=False..."
        model.eval()
        [x_train, y_scan_train, x_val, y_scan_val, y_type_train, y_type_val] = my_dataset.get_split(0, device=device)
        z_train, y_scan_train_hat, y_type_train_hat, x_rec_train = model(x_train)
        z_val, y_scan_val_hat, y_type_val_hat, x_rec_val = model(x_val)
        z_test, y_scan_test_hat, y_type_test_hat, x_rec_test = model(torch.tensor(
            my_dataset.X_test, dtype=torch.float32).to(device))
        z_train = z_train.detach().cpu().numpy()
        z_val = z_val.detach().cpu().numpy()
        z_test = z_test.detach().cpu().numpy()
        self.z_train = z_train
        self.y_scan_train_hat = np.argmax(y_scan_train_hat.detach().cpu().numpy(), axis=-1)
        self.y_type_train_hat = np.argmax(y_type_train_hat.detach().cpu().numpy(), axis=-1)
        self.x_rec_train = x_rec_train.detach().cpu().numpy()

        self.z_val = z_val
        self.y_scan_val_hat = np.argmax(y_scan_val_hat.detach().cpu().numpy(), axis=-1)
        self.y_type_val_hat = np.argmax(y_type_val_hat.detach().cpu().numpy(), axis=-1)
        self.x_rec_val = x_rec_val.detach().cpu().numpy()

        self.z_test = z_test
        self.y_scan_test_hat = np.argmax(y_scan_test_hat.detach().cpu().numpy(), axis=-1)
        self.y_type_test_hat = np.argmax(y_type_test_hat.detach().cpu().numpy(), axis=-1)
        self.x_rec_test = x_rec_test.detach().cpu().numpy()

        train_ind, val_ind = my_dataset.val_splits[0]
        self.ipl_depth_train = my_dataset.ipl_depth_train[train_ind]
        self.ipl_depth_val = my_dataset.ipl_depth_train[val_ind]
        self.ipl_depth_test = my_dataset.ipl_depth_test
        self.train_ind = train_ind
        self.val_ind = val_ind

