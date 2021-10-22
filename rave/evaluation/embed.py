from openTSNE import TSNE
import umap
from sklearn.decomposition import PCA


def get_embedding(data, embedding_type="pca", do_pca=True, num_pcs=20,
                  pca_kwargs=dict(), embedding_kwargs=dict()):
    """
    This function takes as input the data (can be raw or model output), and
    returns an embedding of the desired type
    :param data: array; of shape n_samples x n_orig_dims
    :param embedding_type: str; one of ["pca", "tsne", "umap"]
    :param do_pca: bool;
    :param num_pcs: how many PCs to keep; obsolete of do_pca=False
    :param embedding_kwargs: optional kwargs passed on to embedding function;
                            obsolete if embedding_type="pca"
    :return:
    the resulting 2-dimensional embedding for visualization (shape n_samples x 2);
    in the case of openTSNE, this is of type openTSNE.TSNEEmbedding; otherwise,
    a numpy array
    """
    if do_pca:
        pca_obj = PCA(random_state=42, n_components=num_pcs, **pca_kwargs)
        embedding_input = pca_obj.fit_transform(data)
    else:
        embedding_input = data
    if embedding_type == "pca":
        embedding = embedding_input[:, :2]
        embedder = pca_obj
    elif embedding_type == "tsne":
        tsne_obj = TSNE(random_state=42, **embedding_kwargs)
        embedding = tsne_obj.fit(embedding_input)
        embedder = tsne_obj
    elif embedding_type == "umap":
        init = embedding_kwargs.pop("initialization", "spectral")
        if init == "pca":
            if do_pca:
                init = embedding_input[:, :2]
            else:
                pca_obj = PCA(random_state=42, n_components=2, **pca_kwargs)
                init = pca_obj.fit_transform(data)
            embedding_kwargs.update(dict(init=init))
        reducer = umap.UMAP(random_state=42, **embedding_kwargs)
        embedding = reducer.fit_transform(embedding_input)
        embedder = reducer
    return embedding, embedder
