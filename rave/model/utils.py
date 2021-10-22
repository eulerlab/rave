import torch
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr


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
        recon_loss += reconstruction_loss(x_rec[idxs, :], x[idxs, :])
        domain_loss += domain_classification_loss(y_scan_hat[idxs], y_scan[idxs])
    for c in dataset.type_label:
        idxs = y_type == c
        type_loss += type_classification_loss(y_type_hat[idxs], y_type[idxs])
    loss = weight_recon * recon_loss + weight_type * type_loss
    # print(recon_loss.detach().cpu().numpy(), type_loss.detach().cpu().numpy(),
    #       domain_loss.detach().cpu().numpy())
    if adversarial_training:
        loss -= weight_scan * domain_loss
    return loss


def get_adv_loss(dataset, y_scan, y_scan_hat, weight_scan, domain_classification_loss):
    domain_loss = 0
    for d in dataset.scan_label:
        idxs = y_scan == d
        domain_loss += domain_classification_loss(y_scan_hat[idxs], y_scan[idxs])
    loss = weight_scan * domain_loss
    return loss


def correlate(a, b):
    x = a.detach().cpu().numpy()
    y = b.detach().cpu().numpy()
    corrs = [spearmanr(i, j) for i, j in zip(x, y)]
    return np.mean(corrs)


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

