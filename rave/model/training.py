import os
import pickle as pkl
import numpy as np
from ray import tune
import torch
import time
from torch import nn
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import ParameterGrid, GridSearchCV
from scipy.stats import spearmanr

from rave.model.architectures import Autoencoder, ResidualModel, CustomDataParallel
from rave.model.utils import *
from typing import Dict, AnyStr
from rave.utils.import_helpers import dynamic_import, split_module_name
from rave.data.datasets import get_bc_data, get_bc_sim_data


def custom_collate_fn(input_list):
    return [el for el in input_list[0]]


def train_model(config: Dict):
    """
    Trains the model
    :param config:
    :param checkpoint_dir:
    :param hypersearch:
    :return:
    """
    hypersearch = config.get("hypersearch", False)
    dataset = config.get("dataset", False)
    if not dataset:
        datatype = config.get("datatype", "bc")
        if datatype == "bc":
            dataset = get_bc_data(
                config['data_dir'], **config["data_kwargs"])
        elif datatype == "sim":
            with open(os.path.join(config["data_dir"], config["dataset_fname"]), "rb") as f:
                dataset = pkl.load(f)

        else:
            raise NotImplementedError("Datatype {} is not supported yet. Please implement"
                            "dataloading for your datatype".format(datatype))

    if config["model_architecture"] == "autoencoder":
        net = Autoencoder(input_shape=dataset.D, **config["model"],
                          device=config["device"])
    elif config["model_architecture"] == "resnet":
        raise NotImplementedError()
        # net = ResidualModel(input_shape=dataset.D, **config["model"])
    else:
        raise NotImplementedError("Model {} not implemented yet".format(
            config["model_architecture"]
        ))

    device = config["device"]
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
        raise NotImplementedError
        # x_train, y_scan_train, x_val, y_scan_val = \
        #     dataset.get_split(config["training"]["split"],
        #                       device=config["device"])
    total_variance_train = torch.sum(x_train ** 2, 1).to(device)
    total_variance_val = torch.sum(x_val ** 2, 1).to(device)

    dis_warm_cool_steps = config["training"].get("dis_warm_cool_steps")
    optimizer = config["training"].get("optimizer")
    lr_model = config["training"].get("lr_model")
    lr_discriminator = config["training"].get("lr_discriminator")
    weight_decay = config["training"].get("weight_decay")
    train_model_every = config["training"].get("train_model_every")
    num_iter = config["training"].get("num_epochs")
    log_dir = config["training"].get("log_dir")
    log_steps = config["training"].get("log_steps")
    weight_mse = config["training"]["weight_recon"]
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
    no_adv = False
    adversarial_training = True
    if dis_warm_cool_steps > 0:
        adversarial_training = False
        no_adv = True
    if config["supervised"]:
        weight_mse = 0.0
    else:
        weight_type = 0.0

    if train_model_every > 1:
        num_iter *= train_model_every

    var_exp_train, var_exp_val = [], []
    adv_acc_train, adv_acc_val = [], []
    type_acc_train, type_acc_val = [], []
    optimizer_model, optimizer_discriminator = get_optimizers(
        net, optimizer, lr_model, lr_discriminator, weight_decay)
    optim_step_count = 1
    # ToDo: Figure out what optim_step_count should be
    print("adversarial training: {}".format(adversarial_training))
    t0 = time.time()
    for i in range(num_iter + 1):
        # Model Training
        for batch_no, (x_train, y_scan_train, y_type_train) in enumerate(
                DataLoader(dataset,
                           sampler=BatchSampler(SequentialSampler(dataset),
                                                batch_size, drop_last=False),
                           shuffle=False,
                           collate_fn=custom_collate_fn)
        ):
            optimizer_model.zero_grad()
            z, y_scan, y_type, x_rec = net(x_train)
            loss, loss_mse, loss_type = get_loss(dataset, x_train, y_scan_train, y_type_train,
                            x_rec,
                            y_scan, y_type, weight_mse, weight_scan,
                            weight_type, reconstruction_loss_fn,
                            type_classification_loss_fn,
                            domain_classification_loss_fn, adversarial_training)
            loss.backward()
            if not i % train_model_every and check_grad(net):
                optimizer_model.step()

            # Adversarial Discriminator
            optimizer_discriminator.zero_grad()
            z, y_scan, y_type, x_rec = net(x_train)
            # compute Adversarial Classification Loss per Scan:

            loss_adv = get_adv_loss(dataset, y_scan_train, y_scan, domain_classification_loss_fn)
            loss_adv.backward()
            if check_grad(net):
                optimizer_discriminator.step()

        if not (i % log_steps):
            torch.save(net.state_dict(), os.path.join(log_dir, 'model.pth'))

            # run_type_acc += accuracy
            print('step=%s, total time=%.2fm' % (i, (time.time() - t0) / 60))
            # run_ve /= log_steps
            # run_adv_acc /= log_steps
            # run_type_acc /= log_steps

            # validate
            # steps_val.append(i)
            net.eval()
            z, y_scan, y_type, x_rec = net(x_val)
            res = (x_val - x_rec) ** 2
            ve = torch.mean(1 - torch.sum(res, 1) / total_variance_val).item()
            var_exp_val.append(ve)
            _, predicted = torch.max(y_scan, 1)
            accuracy_scan = (predicted == y_scan_val).double().mean().item()
            adv_acc_val.append(accuracy_scan)
            _, predicted = torch.max(y_type, 1)
            accuracy_type = (predicted == y_type_val).double().mean().item()
            type_acc_val.append(accuracy_type)
            # log = 'Train: var.exp.=%.2f, acc_adv=%.2f, acc_type=%.2f; mse=%.2f' % (
            #     run_ve, run_adv_acc, run_type_acc, loss_mse)
            log = ' Val: var.exp.=%.2f, acc_adv=%.2f, acc_type=%.2f; mse=%.2f' % (
                ve, accuracy_scan, accuracy_type, loss_mse)
            print(log)
            run_ve = 0.0
            run_adv_acc = 0.0
            run_type_acc = 0.0
            net.train()

        if i > dis_warm_cool_steps:
            if num_iter - dis_warm_cool_steps > i:
                if no_adv:
                    print('\nBeginning 2 player game:\n')
                    no_adv = False
                    optimizer_model, optimizer_discriminator = get_optimizers(
                        net, optimizer, lr_model, lr_discriminator, weight_decay)
            else:
                if not no_adv:
                    print('\nCooling down (no adversarial training):\n')
                    no_adv = True
                    weight_type = 0
                    weight_mse = 0
                    weight_scan = 0
                    optimizer_net, optimizer_discriminator = get_optimizers(
                        net, optimizer, lr_model, lr_discriminator,
                        weight_decay)

    net.eval()
    if hypersearch:
        final_eval_results, grid_search_type, grid_search_scan = final_evaluate(
            net, dataset, x_train, x_val, total_variance_val, y_scan_train,
            y_type_train, y_scan_val, y_type_val)

        # final_eval_results.update(dict(iteration=epoch))
        # tune.report(iteration=epoch,
        #             var_exp=final_eval_results["var_exp"],
        #             val_corr=final_eval_results["val_corr"],
        #             adv_acc=final_eval_results["adv_acc"],
        #             type_acc=final_eval_results["type_acc"],
        #             final_scan_acc=final_eval_results["final_scan_acc"],
        #             final_type_acc=final_eval_results["final_type_acc"],
        #             lsq_var_exp=final_eval_results["lsq_var_exp"],
        #             lsq_corrs=final_eval_results["lsq_corrs"])
        # with open(os.path.join(log_dir, 'grid_search_type.pkl'), 'wb') as handle:
        #     pkl.dump(grid_search_type.best_params_,
        #              handle, protocol=pkl.HIGHEST_PROTOCOL)
        # with open(os.path.join(log_dir, 'grid_search_scan.pkl'), 'wb') as handle:
        #     pkl.dump(grid_search_scan.best_params_,
        #              handle, protocol=pkl.HIGHEST_PROTOCOL)

    results = {
        'var_exp_train': var_exp_train,
        'var_exp_val': var_exp_val,
        'adv_acc_train': adv_acc_train,
        'adv_acc_val': adv_acc_val,
        'type_acc_train': type_acc_train,
        'type_acc_val': type_acc_val,
    }

    with open(os.path.join(log_dir, 'train_log.pkl'), 'wb') as handle:
        pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)

    if not hypersearch:
        return net
















