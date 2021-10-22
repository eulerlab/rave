import matplotlib as mpl
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd

from rave.model.utils import ModelOutputs
from rave.evaluation.embed import get_embedding


def visualize_embedding_basic(model, dataset, datatier, embedding=None,
                              embedder=None, embedding_settings={},
                              embedding_input_version = "raw", scan_label_vers="raw",
                              type_label_vers="raw",
                              dataset_cmap=mpl.colors.ListedColormap(
                                  ['tab:green', 'darkorange']),
                              type_cmap="tab20b", depth_cmap="RdBu",
                              device="cuda"):
    type_names = ['1', '2', '3a', '3b', '4', '5t', '5o', '5i', 'X', '6', '7', '8', '9', 'R']
    model_outputs = ModelOutputs()
    model_outputs.get_model_outputs(model, dataset, device=device)
    [x_train, y_scan_train, x_val, y_scan_val, y_type_train, y_type_val] = dataset.get_split_numpy(0)
    x_test = dataset.X_test
    ipl_depth_train, ipl_depth_val = dataset.get_ipl_split(0)
    ipl_depth_test = dataset.ipl_depth_test
    y_scan_test = dataset.Y_scan_test
    ''' Get the scan label'''
    if scan_label_vers == "raw":
        if datatier == "train":
            scan_label = y_scan_train
        elif datatier == "val":
            scan_label = y_scan_val
        elif datatier == "test":
            scan_label = dataset.Y_scan_test
    elif scan_label_vers == "rave":
        key = "y_scan_{}_hat".format(datatier)
        scan_label = getattr(model_outputs, key)

    ''' Get the type label '''
    if type_label_vers == "raw":
        if datatier == "train":
            type_label = y_type_train
        elif datatier == "val":
            type_label = y_type_val
        elif datatier == "test":
            type_label = dataset.Y_type_test
        # for dataset creation, the labels of dataset 1 are signflipped and
        # indexing starts at 1; revert this here
        bool_mask = locals()["y_scan_{}".format(datatier)] == 1
        new_type_label = deepcopy(type_label)
        new_type_label[bool_mask] *= -1
        new_type_label[bool_mask] -= 1
    elif type_label_vers == "rave":
        key = "y_type_{}_hat".format(datatier)
        new_type_label = getattr(model_outputs, key)

    ''' Get the embedding '''
    if embedding is None or embedder is None:
        if embedding_input_version == "raw":
            embedding_input = locals()["x_{}".format(datatier)]

        elif embedding_input_version == "rave":
            embedding_input = getattr(model_outputs, "z_{}".format(datatier))
        embedding, embedder = get_embedding(embedding_input,
                                            **embedding_settings)

    ipl_depth = locals()["ipl_depth_{}".format(datatier)]
    df = pd.DataFrame(dict(x=embedding[:, 0], y=embedding[:, 1],
                           scan_label=scan_label,
                           type_label=new_type_label,
                           ipl_depth=locals()["ipl_depth_{}".format(datatier)]))

    ''' Set plot params '''
    fig = plt.figure(constrained_layout=True, figsize=(5, 10))
    mosaic = """
       A
       B
       C
       """
    """
    DATASET
    """
    dataset_norm = Normalize(vmin=0, vmax=1)
    axd = fig.subplot_mosaic(mosaic)
    axd["A"].scatter(*embedding.transpose(), c=scan_label, cmap=dataset_cmap,
                     norm=dataset_norm, marker="."
                     )
    sc_map = ScalarMappable(norm=dataset_norm, cmap=dataset_cmap)
    # cbar = plt.colorbar(sc_map, ax=axd["A"])
    # cbar.set_label("Dataset", rotation=90)
    # cbar.ax.get_yaxis().labelpad = 15
    labels = ["Dataset A", "Dataset B"]
    legend_elements = [Line2D([0], [0], linestyle="", marker='o',
                              color=sc_map.to_rgba(i), label=labels[i],
                              markersize=5) for i in range(len(labels))]
    axd["A"].legend(handles=legend_elements, loc='center left',
                    bbox_to_anchor=(1, 0.5))

    """ CEll types """
    type_norm = Normalize(vmin=0, vmax=13)
    out = axd["B"].scatter(*embedding.transpose(), c=new_type_label, cmap=type_cmap,
                           norm=type_norm, marker=".")
    sc_map = ScalarMappable(norm=type_norm, cmap=type_cmap)
    # cbar = plt.colorbar(out, ax=axd["B"])
    # cbar.set_label("Cell Type", rotation=90)
    # cbar.ax.get_yaxis().labelpad = 15

    labels = type_names
    legend_elements = [Line2D([0], [0], linestyle="", marker='o',
                              color=sc_map.to_rgba(i), label=labels[i],
                              markersize=5) for i in range(len(labels))]
    axd["B"].legend(handles=legend_elements, loc='center left',
                    bbox_to_anchor=(1, 0.5))

    """ IPL depth """
    depth_norm = Normalize(vmin=ipl_depth.min(), vmax=ipl_depth.max())
    axd["C"].scatter(*embedding.transpose(),
                     c=ipl_depth,
                     cmap=depth_cmap,
                     norm=depth_norm,
                     marker=".")
    sc_map = ScalarMappable(norm=depth_norm, cmap=depth_cmap)
    cbar = plt.colorbar(sc_map, ax=axd["C"])
    cbar.set_label("IPL depth", rotation=90)
    cbar.ax.get_yaxis().labelpad = 15
    # sns.scatterplot(data=df, x="x", y="y", hue="scan_label",
    #                 palette=dataset_cmap, ax=axd["A"])
    #
    # sns.scatterplot(data=df, x="x", y="y", hue="new_type_label",
    #                 palette=type_cmap, ax=axd["B"])
    #
    # sns.scatterplot(data=df, x="x", y="y", hue="ipl_depth",
    #                 palette=depth_cmap, ax=axd["C"])
    for ax in axd.values():
        ax.axis("off")
    return fig, embedding, embedder


def visualize_embedding_new(model, dataset, raw_input, type_clf, scan_clf, scan_label_vers,
                            type_label_vers, clf_input_version, embedding_settings,
                            embedding_input_version,
                            datatier, embedding=None, embedder=None,
                            dataset_cmap=mpl.colors.ListedColormap(['tab:green', 'darkorange']),
                            type_cmap="tab20b", depth_cmap="RdBu"):
    """
    visualize embedding of selected type (pca, tsne, umap); color according to
    dataset, cell types and ipl depth, for raw and/or rave output; get labels
    for coloring from original, rave model, or RFC
    :param model: trained model
    :param dataset:
    :param type_clf:
    :param scan_clf:
    :param scan_label_vers:
    :param type_label_vers:
    :param embedding_type:
    :param datatier:
    :return:
    """
    type_names = ['1', '2', '3a', '3b', '4', '5t', '5o', '5i', 'X', '6', '7', '8', '9', 'R']
    model_outputs = ModelOutputs()
    model_outputs.get_model_outputs(model, dataset)
    [x_train, y_scan_train, x_val, y_scan_val, y_type_train, y_type_val] = dataset.get_split_numpy(0)
    x_test = dataset.X_test
    ipl_depth_train, ipl_depth_val = dataset.get_ipl_split(0)
    ipl_depth_test = dataset.ipl_depth_test
    y_scan_test = dataset.Y_scan_test
    ''' Get the scan label'''
    if scan_label_vers == "raw":
        if datatier == "train":
            scan_label = y_scan_train
        elif datatier == "val":
            scan_label = y_scan_val
        elif datatier == "test":
            scan_label = dataset.Y_scan_test
    elif scan_label_vers == "rave":
        key = "y_scan_{}_hat".format(datatier)
        scan_label = getattr(model_outputs, key)
    elif scan_label_vers == "rfc":
        # ToDo: adapt this also for RAVE (in addition to RAVE+)
        if clf_input_version == "raw":
            scan_label = scan_clf.predict(raw_input)
        elif clf_input_version == "rave":
            clf_input = getattr(model_outputs, "z_{}".format(datatier))
            scan_label = scan_clf.predict(clf_input)

    ''' Get the type label '''
    if type_label_vers == "raw":
        if datatier == "train":
            type_label = y_type_train
        elif datatier == "val":
            type_label = y_type_val
        elif datatier == "test":
            type_label = dataset.Y_type_test
        # for dataset creation, the labels of dataset 1 are signflipped and
        # indexing starts at 1; revert this here
        bool_mask = locals()["y_scan_{}".format(datatier)] == 1
        new_type_label = deepcopy(type_label)
        new_type_label[bool_mask] *= -1
        new_type_label[bool_mask] -= 1
    elif type_label_vers == "rave":
        key = "y_type_{}_hat".format(datatier)
        new_type_label = getattr(model_outputs, key)
    elif type_label_vers == "rfc":
        # ToDo: adapt this also for RAVE (in addition to RAVE+)
        if clf_input_version == "raw":
            new_type_label = type_clf.predict(raw_input)
        elif clf_input_version == "rave":
            clf_input = getattr(model_outputs, "z_{}".format(datatier))
            new_type_label = type_clf.predict(clf_input)
    ''' Get the embedding '''
    if embedding is None or embedder is None:
        if embedding_input_version == "raw":
            embedding_input = locals()["x_{}".format(datatier)]

        elif embedding_input_version == "rave":
            embedding_input = getattr(model_outputs, "z_{}".format(datatier))
        embedding, embedder = get_embedding(embedding_input,
                                            **embedding_settings)

    ipl_depth = locals()["ipl_depth_{}".format(datatier)]
    df = pd.DataFrame(dict(x=embedding[:, 0], y=embedding[:, 1],
                           scan_label=scan_label,
                           type_label=new_type_label,
                           ipl_depth=locals()["ipl_depth_{}".format(datatier)]))

    ''' Set plot params '''
    fig = plt.figure(constrained_layout=True, figsize=(5, 10))
    mosaic = """
    A
    B
    C
    """
    """
    DATASET
    """
    dataset_norm = Normalize(vmin=0, vmax=1)
    axd = fig.subplot_mosaic(mosaic)
    axd["A"].scatter(*embedding.transpose(), c=scan_label, cmap=dataset_cmap,
                     norm=dataset_norm, marker="."
                     )
    sc_map = ScalarMappable(norm=dataset_norm, cmap=dataset_cmap)
    # cbar = plt.colorbar(sc_map, ax=axd["A"])
    # cbar.set_label("Dataset", rotation=90)
    # cbar.ax.get_yaxis().labelpad = 15
    labels = ["Dataset A", "Dataset B"]
    legend_elements = [Line2D([0], [0], linestyle="", marker='o',
                              color=sc_map.to_rgba(i), label=labels[i],
                          markersize=5) for i in range(len(labels))]
    axd["A"].legend(handles=legend_elements, loc='center left',
                    bbox_to_anchor=(1, 0.5))

    """ CEll types """
    type_norm = Normalize(vmin=0, vmax=13)
    out = axd["B"].scatter(*embedding.transpose(), c=new_type_label, cmap=type_cmap,
                     norm=type_norm, marker=".")
    sc_map = ScalarMappable(norm=type_norm, cmap=type_cmap)
    # cbar = plt.colorbar(out, ax=axd["B"])
    # cbar.set_label("Cell Type", rotation=90)
    # cbar.ax.get_yaxis().labelpad = 15

    labels = type_names
    legend_elements = [Line2D([0], [0], linestyle="", marker='o',
                              color=sc_map.to_rgba(i), label=labels[i],
                              markersize=5) for i in range(len(labels))]
    axd["B"].legend(handles=legend_elements,  loc='center left',
                    bbox_to_anchor=(1, 0.5))


    """ IPL depth """
    depth_norm = Normalize(vmin=ipl_depth.min(), vmax=ipl_depth.max())
    axd["C"].scatter(*embedding.transpose(),
                     c=ipl_depth,
                     cmap=depth_cmap,
                     norm=depth_norm,
                     marker=".")
    sc_map = ScalarMappable(norm=depth_norm, cmap=depth_cmap)
    cbar = plt.colorbar(sc_map, ax=axd["C"])
    cbar.set_label("IPL depth", rotation=90)
    cbar.ax.get_yaxis().labelpad = 15
    # sns.scatterplot(data=df, x="x", y="y", hue="scan_label",
    #                 palette=dataset_cmap, ax=axd["A"])
    #
    # sns.scatterplot(data=df, x="x", y="y", hue="new_type_label",
    #                 palette=type_cmap, ax=axd["B"])
    #
    # sns.scatterplot(data=df, x="x", y="y", hue="ipl_depth",
    #                 palette=depth_cmap, ax=axd["C"])
    for ax in axd.values():
        ax.axis("off")
    return fig, embedding, embedder