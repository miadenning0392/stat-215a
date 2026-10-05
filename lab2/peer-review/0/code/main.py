import pandas as pd
from pyreadr import read_r

import clean
import figures
import cluster


def main():
    # load data, clean, figures, PCA, clustering, stability evaluation
    ling_data = pd.read_csv("../data/lingData.txt", sep="\\s+")
    ling_location = pd.read_csv("../data/lingLocation.txt", sep="\\s+")
    question_data = read_r("../data/question_data.RData")

    print("Printing raw figures")
    figures.exploratory_figures_raw(ling_data=ling_data)
    ling_data = clean.clean_ling_data(ling_data=ling_data)

    figures.exploratory_figures_cleaned(ling_data, question_data)

    print(f"Now doing dim red with PCA")
    pc_scores = cluster.dim_red(ling_location)

    pc_scores = cluster.cluster(pc_scores)

    # stability evaluation
    print(f"Running stability evaluation cleaning ")
    pc_scores_uncenter = cluster.dim_red(ling_location, mean_center=False, plot=False)
    pc_scores_uncenter = cluster.cluster(pc_scores_uncenter, plot=False)
    print(cluster.compare_clusters(pc_scores, pc_scores_uncenter))

    # stability evaluation
    print(f"Running stability evaluation sampling")
    ling_location_perterb = cluster.perturb_data(ling_location)
    pc_scores_perterb = cluster.dim_red(ling_location_perterb, plot=False)
    pc_scores_perterb = cluster.cluster(pc_scores_perterb, plot=False)
    print(cluster.compare_clusters(pc_scores, pc_scores_perterb))


if __name__ == "__main__":
    main()
