import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
import seaborn as sns
from numpy.random import choice


def get_kde_per_type(ipl_em, depth_sampling_points,
                     ipl_depth, type_labels, plot=True, ppath=None,
                     palette="tab20b", bw_method=None, kde_weights=None,
                     type_names=None, violin_kwargs=dict()):
    """
    Inputs:
    ipl_em: len(depth_sampling_points) x n_types array, summing to 1 across axis=0
            giving the probability p(depth|type)
    depth_sampling_points: array giving 400 equally space points at which ipl_em was
            evaluated
    ipl_depth: n_samples array giving the ipl depths of our BCs
    type_labels: n_samples array giving the type labels of our BCs

    """
    if type_names is None:
        type_names = ["1", "2", "3a", "3b", "4", "5t", "5o", "5i", "X", "6", "7", "8", "9", "R"]
    # Estimate Gaussian KDE of our data per type
    n_types = ipl_em.shape[-1]
    density_at_sampling_points_per_type = np.zeros_like(ipl_em)
    for i in range(n_types):
        # Estimate Gaussian KDE of our data per type
        n_cells_per_type = len(type_labels[type_labels==i])
        if n_cells_per_type > 5:
            if kde_weights is None:
                weights = np.ones_like(type_labels[type_labels == i]) * 1/n_cells_per_type
            else:
                weights = kde_weights[type_labels == i]
            kde = gaussian_kde(ipl_depth[type_labels == i],
                               bw_method=bw_method, weights=weights)
            # evaluate KDE at depth sampling points
            density_at_sampling_points_per_type[:, i] = kde.evaluate(depth_sampling_points)
            # normalize such that it sums to 1 across depth samplint points
            density_at_sampling_points_per_type[:, i] /= density_at_sampling_points_per_type[:, i].sum()
        else:
            density_at_sampling_points_per_type[:, i] = 1/density_at_sampling_points_per_type.shape[0]

    if plot:
        numeric_types = np.arange(0, 14)
        df = pd.DataFrame(dict(depth=np.concatenate(
            [np.tile(depth_sampling_points, n_types), np.tile(depth_sampling_points, n_types)]),
                                            density=np.concatenate(
        [density_at_sampling_points_per_type.reshape(-1, order="F"), -1*ipl_em.reshape(-1, order="F")]),
                                            types=np.concatenate(
                                                [np.repeat(numeric_types, len(depth_sampling_points)),
                                                 np.repeat(numeric_types, len(depth_sampling_points))]),
                                            dataset=np.concatenate([["dataset B" for i in range(14*400)],
                                                                   ["em" for i in range(14*400)]])
                                            ))
        sns.lineplot(data=df,
                     y="density", x="depth", hue="types", hue_order=numeric_types, style="dataset",
                     palette=palette, hue_norm=[0, 13], dashes=[[2, 2, 10, 2], [6, 1]], legend=False)
        plt.gca().set_xlim(2, -2)
        if ppath is not None:
            plt.savefig(ppath, dpi=300)
            plt.savefig(ppath[:-3]+'svg', dpi=300)

        ### generate depth samples per type from em distribution
        depth_samples_em = []
        type_labels_em = []
        for i in range(n_types):
            depth_samples = choice(depth_sampling_points, size=len(ipl_depth[type_labels == i]),
                   p=ipl_em[:, i]
                   )
            depth_samples_em.append(depth_samples)
            type_labels_em.append(np.ones(len(depth_samples))*i)

        depth_samples_em = np.concatenate(depth_samples_em)
        depth_samples_all = np.concatenate([ipl_depth.reshape(-1, order="F"),
                                           depth_samples_em])
        dataset = np.repeat(["dataset B", "em"], repeats=len(ipl_depth))
        type_labels_ = np.concatenate([type_labels, np.concatenate(type_labels_em)])

        df_violin = pd.DataFrame(dict(depth=depth_samples_all,
                                      dataset=dataset,
                                      types=type_labels_))
        plt.figure()
        sns.violinplot(data=df_violin,
                     x="types", y="depth", hue="dataset",
                       split=True,
                     palette=palette, **violin_kwargs)
        if ppath is not None:
            ppath = ppath[:-4]+"_violin.png"
            plt.savefig(ppath, dpi=300)
    if ppath is not None:
        plt.savefig(ppath, dpi=300)
    return density_at_sampling_points_per_type


def get_js_div(ipl_em, kde_per_type):
    js = np.zeros(ipl_em.shape[-1])
    js_normalized = np.zeros(ipl_em.shape[-1])
    uniform = np.ones(ipl_em.shape[0]) * 1/ipl_em.shape[0]
    for t in range(len(js)):
        js[t] = jensenshannon(ipl_em[:, t]+10e-10, kde_per_type[:, t]+10e-10)
        js_normalized[t] = js[t]/jensenshannon(ipl_em[:, t], uniform)
    return js, js_normalized
