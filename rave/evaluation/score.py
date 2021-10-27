import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import adjusted_rand_score as ari
from scipy import io
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

from rave.evaluation.metrics import get_js_div, get_kde_per_type

def get_all_scores(
        dataset,
        model_output_train,
        model_output_val,
        model_output_test,
        y_scan_train,
        y_scan_val,
        y_scan_test,
        y_type_train,
        y_type_val,
        y_type_test,
        ipl_depth_test,
        ppath=None,
        depth_sampling_points=None,
        ipl_depth_per_type_em=None,
        grid_search_kwargs_scan=dict(),
        grid_search_kwargs_type=dict(),
        n_jobs=20,
        seeds=None,
        conf_weighted_kde_estimate=False,
        ipl_file1='/your/path/to/ipl.mat',
        ipl_file2='/your/path/to/BC_Profiles_Helmstaedter.txt',
        correct_label_mixup=True
        ):
    """
    Evaluates all scores on input
    Takes as input:
    model_output_{train, val, test}:      array     model output on data n_samples x n_features_red
    y_scan_{train, val, test}:     array n_samples        binary array giving scan identity of all
                                        samples (Franke=0, Szatko=1)
    y_type_{train, val, test}: array n_samples       array giving original type labels for
                                        Franke (0-13) and dummy type label -1 for
                                        Szatko
    ipl_depth_test  array n_samples   array giving ipl depth
    ppath:  string          full path (incl figname and file extension) for
                            saving the ipl depth profiles

    depth_sampling_points: array n_depth_samples=400; equally spaced depth
                            sampling points from EM data
    ipl_depth_per_type_em:  array n_depth_samples x n_BC_types (=14)
                            distribution p_EM(depth|type)
    grid_search_kwargs_{scan, type}:    dict    keyword arguments to be passed to grid search
    n_jobs:     int     pass to random forest classifier (RFC)
    seeds:      list of ints to be used as random seeds for the RFC
    conf_weighted_kde_estimate:     bool      whether or not to weight depth values according to classifier confidence
                                              when estimate kernel density
    ipl_file1:  str     path to ipl mat file
    ipl_file2:  str     path to ipl txt file
    correct_label_mixup:    bool    indicating whether or not to correct the label
                                    mixup present in the Franke data file;
                                    not necessary if corrected before (as in sim
                                    data)
    returns:
    result_dictionary   dict containing all results
    acc_dom     array of size n_seeds    performance on test set of scan classifier
                                        trained on model output
    acc_type    array of size n_seeds    performance on Franke test set of type classifier
                                        trained on model output
    ari_dom     array of size n_seeds    ARI on original scan labels and scan labels
                                        predicted by scan classifier trained on model output
    f1          array of size n_seeds    F1 score as in Tran et al. (2020). Genome Biology
    asw_batch   array of size n_seeds   average silhouette width for domain dimension
    js_depth    array of size n_seeds x n_types    Jensen Shannon distance per seed and type
                                                    between ipl depth distribution taken from EM and
                                                    ipl depth distribution according to predicted
                                                    type labels; evaluated on test set of dataset B
    js_normalized array of size n_seeds x n_types   JS distance per seed and type, normalized to JS
                                                    between uniform and EM
    kde_per_type_all    array   n_seeds x n_depth_sampling_points x n_types   KDE estimate per type across depth for 10 RFC predictions
    y_type_test_post    array of size n_samples x n_seeds   predicted type labels from 10 differently initialized RFCs
    y_scan_test_post    array of size n_samples X n_seeds   predicted scan labels from 10 differently initialized RFCs
    confidence          array of size n_samples x n_seeds   conf. of predicted type labels for 10 RFCs
    y_type_test_rl      array of size n_samples             ground truth type labels, potentially corrected for label mixup
    ari_cross           array of size (n_seeds)*(n_seeds-1)/2 indicating the ARI between all possible combinations of
                                                            label predictions of 10 different RFCs
    best_scan_clf       sklearn.ensemble.RandomForestClassifier     best perfoming Scan RFC
    best_type_clf       sklearn.ensemble.RandomForestClassifier     best perfoming type RFC
    """
    if correct_label_mixup:
        y_type_train_rl = deepcopy(y_type_train)
        y_type_val_rl = deepcopy(y_type_val)
        y_type_test_rl = deepcopy(y_type_test)
        index_R = np.where(y_type_train_rl == 11)
        index_8 = np.where(y_type_train_rl == 12)
        index_9 = np.where(y_type_train_rl == 13)
        y_type_train_rl[index_R] = 13
        y_type_train_rl[index_8] = 11
        y_type_train_rl[index_9] = 12

        index_R = np.where(y_type_val_rl == 11)
        index_8 = np.where(y_type_val_rl == 12)
        index_9 = np.where(y_type_val_rl == 13)
        y_type_val_rl[index_R] = 13
        y_type_val_rl[index_8] = 11
        y_type_val_rl[index_9] = 12
        orig_test = deepcopy(y_type_test)
        index_R = np.where(y_type_test_rl == 11)
        index_8 = np.where(y_type_test_rl == 12)
        index_9 = np.where(y_type_test_rl == 13)
        y_type_test_rl[index_R] = 13
        y_type_test_rl[index_8] = 11
        y_type_test_rl[index_9] = 12
    else:
        y_type_test_rl = deepcopy(y_type_test)
        y_type_train_rl = deepcopy(y_type_train)
        y_type_val_rl = y_type_val
        orig_test = deepcopy(y_type_test)

    assert correct_label_mixup != (np.all(orig_test==y_type_test_rl)), "inconsistent type labels"
    # if EM ipl depth and sampling points not provide, load from file
    if depth_sampling_points is None:
        with open(ipl_file1, 'rb') as f:
            ipl_dict = io.loadmat(f)
        depth_sampling_points = np.squeeze(ipl_dict["d"])
    if ipl_depth_per_type_em is None:
        with open(ipl_file2, 'rb') as f:
            ipl_depth_per_type_em = np.loadtxt(f)

    train_idx, val_idx = dataset.val_splits[0]
    clf_input = np.zeros((model_output_train.shape[0] + model_output_val.shape[0],
                          model_output_train.shape[1]))
    clf_scan_target = np.zeros(model_output_train.shape[0] + model_output_val.shape[0])
    clf_type_target = np.zeros(model_output_train.shape[0] + model_output_val.shape[0])
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
    clf_type_target[train_idx] = y_type_train_rl
    clf_type_target[val_idx] = y_type_val_rl

    # Train classifiers first
    ###  train scan classifier
    if seeds is None:
        seeds = [0, 42, 1067, 99, 50, 100, 200, 300, 2020, 2021]
    y_scan_test_post = np.zeros((model_output_test.shape[0], len(seeds)))
    confidence = np.zeros((model_output_test.shape[0], len(seeds)))
    acc_dom = np.zeros(len(seeds))
    ari_dom = np.zeros(len(seeds))
    print("start grid search")
    scan_clf_param_grid = dict(class_weight=["balanced"],
        n_estimators=[5, 10, 20, 30],
                               max_depth=[5, 10, 15, 20, None],
                               ccp_alpha=[0, 0.001, 0.01],
                               max_samples=[0.5, 0.7, 0.9, 1],
                               random_state=[2021]
                               )
    grid_search_scan = GridSearchCV(estimator=RFC(), param_grid=scan_clf_param_grid,
                                    cv=[dataset.val_splits[0]], n_jobs=n_jobs, refit=False,
                                    **grid_search_kwargs_scan)

    grid_search_scan.fit(clf_input, clf_scan_target)
    best_params_scan = deepcopy(grid_search_scan.best_params_)
    print("sklearn grid search identified best scan RFC parameters: \n",
          best_params_scan)
    print("val set accuracy: {:.3f}".format(grid_search_scan.best_score_))
    scan_clfs = []
    for s, seed in enumerate(seeds):
        best_params_scan.update(dict(random_state=seed))
        print(best_params_scan)
        scan_clf = RFC(**best_params_scan, n_jobs=20)
        scan_clf.fit(model_output_train, y_scan_train)
        train_score = scan_clf.score(model_output_train, y_scan_train)
        print("scan train score seed {}: {:.3f}".format(s, train_score))
        y_scan_test_post[:, s] = scan_clf.predict(model_output_test)
        confidence_temp = scan_clf.predict_proba(model_output_test)
        confidence[:, s] = confidence_temp.max(axis=-1)
        acc_dom[s] = scan_clf.score(model_output_test, y_scan_test)
        ari_dom[s] = ari(y_scan_test, y_scan_test_post[:, s])
        scan_clfs.append(scan_clf)
    best_scan_seed_idx = np.argmax(acc_dom)
    print("Best achieved test set scan accuracy \n",
          acc_dom[best_scan_seed_idx])
    best_scan_clf = scan_clfs[best_scan_seed_idx]


    ### train type classifier
    y_type_test_post = np.zeros((model_output_test.shape[0], len(seeds)))
    y_type_train_post = np.zeros((model_output_train.shape[0], len(seeds)))
    acc_type = np.zeros(len(seeds))
    grid_search_type = GridSearchCV(estimator=RFC(),
                                    param_grid=scan_clf_param_grid,
                                    cv=[[train_idx_franke, val_idx_franke]],
                                    n_jobs=n_jobs, refit=False,
                                    **grid_search_kwargs_type)

    grid_search_type.fit(clf_input, clf_type_target)
    best_params_type = deepcopy(grid_search_type.best_params_)
    print("sklearn grid search identified best type RFC parameters: \n",
          best_params_type)
    type_clfs = []
    for s, seed in enumerate(seeds):
        best_params_type.update(dict(random_state=s))
        type_clf = RFC(**best_params_type, n_jobs=20)
        type_clf.fit(model_output_train[y_scan_train == 0],
                     y_type_train_rl[y_scan_train == 0])
        train_score = type_clf.score(model_output_train[y_scan_train == 0],
                     y_type_train_rl[y_scan_train == 0])
        print("type train score seed {}: {:.3f}".format(s, train_score))
        y_type_test_post[:, s] = type_clf.predict(model_output_test)
        y_type_train_post[:, s] = type_clf.predict(model_output_train)
        acc_type[s] = type_clf.score(model_output_test[y_scan_test == 0],
                                     y_type_test_rl[y_scan_test == 0])
        type_clfs.append(type_clf)
    best_type_seed_idx = np.argmax(acc_type)
    print("Best achieved test set type accuracy \n",
          acc_type[best_type_seed_idx])
    best_type_clf = type_clfs[best_type_seed_idx]

    js_depth = np.zeros((len(seeds), ipl_depth_per_type_em.shape[-1]))
    js_normalized = np.zeros((len(seeds), ipl_depth_per_type_em.shape[-1]))
    kde_per_type_all = np.zeros((len(seeds), 400, 14))
    if conf_weighted_kde_estimate:
        for i in range(len(seeds)):
            kde_per_type = get_kde_per_type(ipl_depth_per_type_em,
                                            depth_sampling_points,
                                            ipl_depth_test[y_scan_test == 1],
                                            y_type_test_post[y_scan_test == 1, i],
                                            kde_weights=confidence[y_scan_test == 1, i],
                                            plot=False)
            js_depth[i, :], js_normalized[i, :] = get_js_div(ipl_depth_per_type_em, kde_per_type)
            kde_per_type_all[i] = kde_per_type

    else:
        for i in range(len(seeds)):
            kde_per_type = get_kde_per_type(ipl_depth_per_type_em,
                                            depth_sampling_points,
                                            ipl_depth_test[y_scan_test == 1],
                                            y_type_test_post[y_scan_test == 1, i],
                                            kde_weights=None,
                                            plot=False)
            kde_per_type_all[i] = kde_per_type
            js_depth[i, :] = get_js_div(ipl_depth_per_type_em, kde_per_type)
    plot = True
    if ppath is None:
        plot = False
    min_js_idx = np.argmin(js_depth.mean(axis=-1))
    best_kde_per_type = get_kde_per_type(ipl_depth_per_type_em,
                                    depth_sampling_points,
                                    ipl_depth_test[y_scan_test == 1],
                                    y_type_test_post[y_scan_test == 1, min_js_idx],
                                    plot=plot, ppath=ppath)
    print(np.all(orig_test==y_type_test_rl))
    ari_cross = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            temp = adjusted_rand_score(y_type_test_post[y_scan_test == 1, i],
                                       y_type_test_post[y_scan_test == 1, j],
                                       )
            ari_cross.append(temp)
    ari_cross = np.asarray(ari_cross)
    result_dictionary = dict(acc_dom=acc_dom, acc_type=acc_type,
                             ari_dom=ari_dom, js_depth=js_depth,
                             js_normalized=js_normalized,
                             min_js_idx=min_js_idx,
                             kde_per_type_all=kde_per_type_all,
                             y_type_test_post=y_type_test_post,
                             y_type_train_post=y_type_train_post,
                             y_scan_test_post=y_scan_test_post,
                             confidence=confidence, y_type_test_rl=y_type_test_rl,
                             ari_cross=ari_cross, best_scan_clf=best_scan_clf,
                             best_type_clf=best_type_clf
                            )
    print(np.unique(y_type_test_post[y_scan_test == 1]))
    return result_dictionary, [acc_dom, acc_type, ari_dom, \
           js_depth, js_normalized, kde_per_type_all, y_type_test_post, y_scan_test_post, \
           confidence, y_type_test_rl, ari_cross, best_scan_clf, best_type_clf]


def get_type_js(dataset, ipl_dict, result_dict):
    """
    Calculates the Jensen-Shannon distance for distributions across types p(type),
    estimated from EM prior + depth sampling distribution on the one hand
    (p_EM(type) = p_EM(type|depth) * p_scan(depth)
    and from cell type assignments by given method on the other hand
    (p_method(type)).

    Inputs:
    dataset: Dataset object (rave.data.datasets)
    ipl_dict: load from '/gpfs01/euler/data/Resources/Classifier/data/ipl.mat'
    result_dict: result dictionary output by get_all_scores for current method
    """
    # get KDE estimator for sampling depth from dataset B (Szatko scan)
    szatko_bool = dataset.Y_scan_test == 1
    depth_kde_estimator = gaussian_kde(dataset.ipl_depth_test[szatko_bool])
    # evaluate KDE across the depth range known for EM
    kde_across_depth = depth_kde_estimator.evaluate(ipl_dict["d"])
    # normalize
    kde_across_depth /= kde_across_depth.sum()
    # p(type) =  p(depth) * p(type|depth)
    p_type_em = np.dot(kde_across_depth, ipl_dict["prior"])
    # get probability dist. across types for current method
    n_classifier_seeds = result_dict["y_type_test_post"][szatko_bool].shape[1]
    bin_counts = np.asarray(
        [np.bincount(
            result_dict["y_type_test_post"][szatko_bool][:, s].astype(int),
            minlength=14
        ) for s in range(n_classifier_seeds)])
    bin_counts_normed = np.asarray([bc / bc.sum() for bc in bin_counts])
    mean_bin_counts = bin_counts_normed.mean(axis=0)
    js_em = jensenshannon(p_type_em, mean_bin_counts)
    return kde_across_depth, p_type_em, js_em