import os
import pickle as pkl
import numpy as np
from ray import tune
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import ParameterGrid, GridSearchCV
from scipy.stats import spearmanr

from rave.model.architectures import Autoencoder, ResidualModel, CustomDataParallel
from rave.model.utils import *
from typing import Dict, AnyStr
from rave.utils.import_helpers import dynamic_import, split_module_name
from rave.data.datasets import get_bc_data, get_bc_sim_data


def train_model(config: Dict, hypersearch=False):
    """
    Trains the model
    :param config:
    :param checkpoint_dir:
    :param hypersearch:
    :return:
    """
    dataset = config.get("dataset", False)
    if not dataset:
        datatype = config.get("datatype", "bc")
        if datatype == "bc":
            dataset = get_bc_data(
                config['data_dir'], **config["data_kwargs"])
        elif datatype == "sim":
            with open(os.path.join(config["data_dir"], "processed_sim_data.pkl"), "rb") as f:
                temp = pkl.load(f)
            out = get_bc_sim_data(**temp)
            dataset = out[-1]
        else:
            raise NotImplementedError("Datatype {} is not supported yet. Please implement"
                            "dataloading for your datatype".format(datatype))

    if config["model_architecture"] == "autoencoder":
        net = Autoencoder(input_shape=dataset.D, **config["model"],
                          device=config["device"])
    elif config["model_architecture"] == "resnet":
        net = ResidualModel(input_shape=dataset.D, **config["model"])
    else:
        raise NotImplementedError("Model {} not implemented yet".format(
            config["model_architecture"]
        ))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            net = CustomDataParallel(net)
    net.to(device)
    if dataset.Y_type is not None:
        [x_train, y_scan_train, x_val, y_scan_val, y_type_train,
         y_type_val] = dataset.get_split(config["training"]["split"],
                                         device=config["device"])
    else:
        x_train, y_scan_train, x_val, y_scan_val = \
            dataset.get_split(config["training"]["split"],
                              device=config["device"])

    dis_warm_cool_steps = config["training"].get("dis_warm_cool_steps")
    optimizer = config["training"].get("optimizer")
    lr_model = config["training"].get("lr_model")
    lr_discriminator = config["training"].get("lr_discriminator")
    weight_decay = config["training"].get("weight_decay")
    train_model_every = config["training"].get("train_model_every")
    num_epochs = config["training"].get("num_epochs")
    log_dir = config["training"].get("log_dir")
    log_steps = config["training"].get("log_steps")
    weight_recon = config["training"]["weight_recon"]
    weight_type = config["training"]["weight_type"]
    weight_scan = config["training"]["weight_scan"]
    batch_size = config["training"]["batch_size"]
    reconstruction_loss_str = config["training"]["reconstruction_loss"]
    type_classification_loss_str = config["training"]\
        ["type_classification_loss"]
    domain_classification_loss_str = config["training"]\
        ["domain_classification_loss"]
    reconstruction_loss_fn = dynamic_import(
        *split_module_name(reconstruction_loss_str)
    )
    type_classification_loss_fn = dynamic_import(
        *split_module_name(type_classification_loss_str)
    )
    domain_classification_loss_fn = dynamic_import(
        *split_module_name(domain_classification_loss_str)
    )

    adversarial_training = True
    if dis_warm_cool_steps > 0:
        adversarial_training = False

    if config["supervised"]:
        weight_recon = 0.0
    else:
        weight_type = 0.0

    if train_model_every > 1:
        num_epochs *= train_model_every

    var_exp_train, var_exp_val = [], []
    adv_acc_train, adv_acc_val = [], []
    type_acc_train, type_acc_val = [], []
    steps_val = []
    run_ve = 0.0
    run_adv_acc = 0.0
    run_type_acc = 0.0

    optimizer_model, optimizer_discriminator = get_optimizers(
        net, optimizer, lr_model, lr_discriminator, weight_decay)
    optim_step_count = 1
    # ToDo: Figure out what optim_step_count should be
    print("adversarial training: {}".format(adversarial_training))
    for epoch in range(num_epochs):
        # for batch_no, (x, y_scan, y_type) in enumerate(
        #         DataLoader(dataset, batch_size, shuffle=True, num_workers=10)
        # ):
        #     x = x.to(config["device"])
        #     y_scan = y_scan.to(config["device"]).long()
        #     y_type = y_type.to(config["device"]).long()
        #     z, y_scan_hat, y_type_hat, x_rec = net(x)
        #     loss = get_loss(dataset, x, y_scan, y_type, x_rec, y_scan_hat, y_type_hat,
        #                     weight_recon, weight_scan, weight_type,
        #                     reconstruction_loss_fn,
        #                     type_classification_loss_fn,
        #                     domain_classification_loss_fn,
        #                     adversarial_training
        #                     )
        #     loss.backward()
        #     if not epoch % train_model_every and check_grad(net):
        #         # if (batch_no+1) % optim_step_count == 0:
        #         optimizer_model.step()
        #         optimizer_model.zero_grad()
        #
        #     z, y_scan_hat, y_type_hat, x_rec = net(x)
        #     loss_adv = get_adv_loss(dataset, y_scan, y_scan_hat, weight_scan,
        #                             domain_classification_loss_fn)
        #     loss_adv.backward()
        #     if check_grad(net):
        #         optimizer_discriminator.step()
        #         optimizer_discriminator.zero_grad()

        optimizer_discriminator.zero_grad()
        optimizer_model.zero_grad()
        z, y_scan_hat, y_type_hat, x_rec = net(x_train)
        loss = get_loss(dataset, x_train, y_scan_train, y_type_train, x_rec, y_scan_hat, y_type_hat,
                        weight_recon, weight_scan, weight_type,
                        reconstruction_loss_fn,
                        type_classification_loss_fn,
                        domain_classification_loss_fn,
                        adversarial_training
                        )
        loss.backward()
        if not epoch % train_model_every and check_grad(net):
            # if (batch_no+1) % optim_step_count == 0:
            optimizer_model.step()
            optimizer_model.zero_grad()

        z, y_scan_hat, y_type_hat, x_rec = net(x_train)
        loss_adv = get_adv_loss(dataset, y_scan_train, y_scan_hat, weight_scan,
                                domain_classification_loss_fn)
        loss_adv.backward()
        if check_grad(net):
            optimizer_discriminator.step()
            optimizer_discriminator.zero_grad()
        if not epoch % log_steps:
            print(epoch)
            torch.save(net.state_dict(), os.path.join(log_dir, 'model.pth'))
            net.eval()
            z_val, y_scan_val_pred, y_type_val_pred, x_rec_val = \
                net(x_val.to(config["device"]))
            corr_val = correlate(x_val, x_rec_val)
            predicted = torch.argmax(y_scan_val_pred, dim=-1)
            accuracy_scan = (predicted == y_scan_val).double().mean().item()
            adv_acc_val.append(accuracy_scan)
            predicted = torch.argmax(y_type_val_pred, dim=-1)
            accuracy_type = (
                    predicted[y_scan_val==0] == y_type_val[y_scan_val==0]
            ).double().mean().item()

            accuracy_type_unsup = (
                    predicted[y_scan_val == 1] == (y_type_val[y_scan_val == 1])*-1 -1
            ).double().mean().item()

            type_acc_val.append(accuracy_type)
            print("corr: {:.2f}, scan_acc: {:.2f}, type_acc: {:.2f}, type_acc B:{:.2f}".format(
                corr_val, accuracy_scan, accuracy_type, accuracy_type_unsup
            ))
            if hypersearch:
                tune.report(
                    iteration=epoch,
                    var_exp=1,  # ToDo fill in
                    corr_val=corr_val,
                    adv_acc=accuracy_scan,
                    type_acc=accuracy_type,
                    final_scan_acc=0,
                    final_type_acc=0,
                    lsq_var_exp=0,
                    lsq_corrs=0
                )
            net.train()

        if epoch > dis_warm_cool_steps:
            adversarial_training = True

    return net
















