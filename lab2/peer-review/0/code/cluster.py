from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.linalg import svd
import numpy as np
import pandas as pd

import figures

RANDOM_STATE = 5


def preprocess_data(
    ling_location,
    scale_by_number_of_people_in_cell: bool = True,
    mean_center: bool = True,
    sd_scale: bool = False,
):
    # scale by people in cell, optional mean center and sd scale

    answer_cols = ling_location.columns[3:]

    if scale_by_number_of_people_in_cell:
        X = ling_location[answer_cols].div(ling_location["Number of people in cell"], axis=0)
    else:
        X = ling_location[answer_cols]

    X = (X - X.mean()) if mean_center else X

    X = X / X.std() if sd_scale else X

    return X


def dim_red(
    ling_location,
    filter_ak_hi: bool = True,
    plot: bool = True,
    scale_by_number_of_people_in_cell: bool = True,
    mean_center: bool = True,
    sd_scale: bool = False,
):
    # PCA via svd, scree plots and PC maps, returns first 3 PC scores

    ling_location = (
        ling_location[(ling_location["Longitude"] > -125) & (ling_location["Latitude"] > 24)]
        if filter_ak_hi
        else ling_location
    )

    # First just plot exmaple scree without centering
    X = preprocess_data(
        ling_location,
        scale_by_number_of_people_in_cell=scale_by_number_of_people_in_cell,
        mean_center=mean_center,
        sd_scale=sd_scale,
    )
    U, d, V = np.linalg.svd(X, full_matrices=False)
    if plot:
        figures.plot_scree(d=d, title="scree_not_mean_centered")

    # Then run with centering and continue results
    X_centered = preprocess_data(
        ling_location,
        scale_by_number_of_people_in_cell=scale_by_number_of_people_in_cell,
        mean_center=mean_center,
        sd_scale=sd_scale,
    )
    U, d, V = np.linalg.svd(X_centered, full_matrices=False)
    if plot:
        figures.plot_scree(d=d)

    pc_scores = pd.DataFrame(X_centered.values @ V.T[:, :3], columns=["PC1", "PC2", "PC3"], index=ling_location.index)
    pc_scores["n"] = ling_location["Number of people in cell"].values
    pc_scores["lat"] = ling_location["Latitude"].values
    pc_scores["long"] = ling_location["Longitude"].values

    if plot:
        figures.plot_pc_on_map(pc_scores, "PC1", "pc1_map")
        figures.plot_pc_on_map(pc_scores, "PC2", "pc2_map")
        figures.plot_pc_on_map(pc_scores, "PC3", "pc3_map")

    return pc_scores


def choose_k(X, k_range=range(2, 11), plot: bool = True):
    # elbow and silhouette values for k-means over k range
    wss = [KMeans(n_clusters=k, n_init="auto", random_state=215).fit(X).inertia_ for k in k_range]
    sil = [silhouette_score(X, KMeans(n_clusters=k, n_init="auto", random_state=215).fit_predict(X)) for k in k_range]
    if plot:
        figures.plot_choose_k(k_range, wss, sil)


def cluster_kmeans(pc_scores, k=6, plot: bool = True):
    # k-means clusters on first 3 PCs
    pc_scores = pc_scores.copy()
    X = pc_scores[["PC1", "PC2", "PC3"]]

    choose_k(X, plot=plot)

    pc_scores["kmeans"] = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE).fit_predict(X) + 1

    if plot:
        figures.plot_clusters_on_map(pc_scores, "kmeans", k)

    return pc_scores


def cluster_hierarchical(pc_scores, k=6, plot: bool = True):
    # ward hierarchical clusters on first 3 PCs
    pc_scores = pc_scores.copy()
    X = pc_scores[["PC1", "PC2", "PC3"]]

    if plot:
        figures.plot_dendrogram(X, k)

    pc_scores["hierarchical"] = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X) + 1

    if plot:
        figures.plot_clusters_on_map(pc_scores, "hierarchical", k)

    return pc_scores


def cluster(pc_scores, plot: bool = True):
    # run k-means and hierarchical, print crosstab and ARI
    pc_scores = cluster_kmeans(pc_scores, k=6, plot=plot)
    pc_scores = cluster_hierarchical(pc_scores, k=6, plot=plot)

    print(pd.crosstab(pc_scores["kmeans"].values, pc_scores["hierarchical"].values))
    print("ARI:", adjusted_rand_score(pc_scores["kmeans"], pc_scores["hierarchical"]))

    return pc_scores


def perturb_data(ling_location):
    # just bootstrap here
    return ling_location.sample(ling_location.shape[0], replace=True, random_state=RANDOM_STATE)


def compare_clusters(original, other):
    # use the RAND score to compare for perterbations
    other = other[~other.index.duplicated()]
    return {
        method: adjusted_rand_score(original.loc[other.index, method], other[method])
        for method in ["kmeans", "hierarchical"]
    }
