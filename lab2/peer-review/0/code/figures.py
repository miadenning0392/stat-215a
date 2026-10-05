import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from scipy.cluster.hierarchy import linkage, dendrogram

FIGS_DIR = "../figs"

STATE_DF = gpd.read_file("../data/shapefiles")
STATE_DF = STATE_DF[STATE_DF["iso_a2"] == "US"]
STATE_DF = STATE_DF[~STATE_DF["postal"].isin(["AK", "HI"])]

STATE_POP = pd.read_csv("state_population.csv")


def my_map_theme():
    # remove axes for map plots
    plt.axis("off")


def plot_on_map(data_to_plot, title, fig_name, s=10, alpha=0.5, palette=None, continuous=False, size=None):
    # scatter of answers by lat and long over state boundaries
    plt.figure(figsize=(10, 8))

    norm = None
    if continuous:
        v = data_to_plot["ans"].abs().max()
        norm = plt.Normalize(-v, v)

    ax = sns.scatterplot(
        data=data_to_plot,
        x="long",
        y="lat",
        hue="ans",
        s=s,
        alpha=alpha,
        palette=palette,
        hue_norm=norm,
        legend=False if continuous else "auto",
        size=size,
        sizes=(25, 200),
    )
    STATE_DF.boundary.plot(ax=ax, color="gray", linewidth=0.5)

    if continuous:
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        plt.colorbar(sm, ax=ax, shrink=0.5)
    else:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None)

    my_map_theme()
    plt.title(title)
    plt.savefig(FIGS_DIR + f"/{fig_name}.png", bbox_inches="tight")


def plot_missing_responses(ling_data):
    # histogram of missing responses per respondent
    q_cols = [c for c in ling_data.columns if c.startswith("Q")]
    n_skipped = (ling_data[q_cols] == 0).sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(n_skipped[n_skipped > 0], bins=range(0, 69))
    ax.set_xlabel("Missing responses")
    ax.set_ylabel("Number of respondants")
    ax.set_title("Distribution of total missing reponses per respondant")
    plt.savefig(FIGS_DIR + "/missing_responses.png", bbox_inches="tight")


def plot_state_map(states, column, title, fig_name):
    # state choropleth for given column
    fig, ax = plt.subplots(figsize=(10, 8))
    states.plot(
        column=column, ax=ax, legend=True, cmap="viridis", edgecolor="black", linewidth=0.3, legend_kwds={"shrink": 0.5}
    )
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIGS_DIR + f"/{fig_name}.png", bbox_inches="tight")


def plot_response_locations(ling_data):
    # map of all response locations
    fig, ax = plt.subplots(figsize=(10, 8))
    STATE_DF.boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.scatter(ling_data["long"], ling_data["lat"], s=1, alpha=0.2)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.axis("off")
    ax.set_title("Location of each respondent")
    plt.savefig(FIGS_DIR + "/response_locations.png", bbox_inches="tight")


def plot_responses_by_state(ling_data):
    # map of response counts by state
    counts = ling_data["STATE"].value_counts().rename("n_responses")
    states = STATE_DF.merge(counts, left_on="postal", right_index=True, how="left")
    plot_state_map(states, "n_responses", "Number of responses per state", "responses_by_state")


def plot_response_rate_by_state(ling_data):
    # map of responses per 100k population by state
    counts = ling_data["STATE"].value_counts().rename("n_responses")
    states = STATE_DF.merge(counts, left_on="postal", right_index=True, how="left")
    states = states.merge(STATE_POP, on="name")
    states["per_100k"] = states["n_responses"] / states["population"] * 100_000
    plot_state_map(states, "per_100k", "Responses per 100,000 residents", "response_rate_by_state")


def exploratory_figures_raw(ling_data):
    # all exploratory figures for raw ling data
    plot_missing_responses(ling_data)
    plot_response_locations(ling_data)
    plot_responses_by_state(ling_data)
    plot_response_rate_by_state(ling_data)


def plot_last_slice_bread(ling_data, question_data):
    # map of Q111 end of bread answers
    last_slice_bread = ling_data[(ling_data["Q111"].between(1, 8)) & (ling_data["long"] > -125)]
    answers_q111 = question_data["ans.111"]

    # Prepare to join
    answers_q111["Q111"] = (answers_q111.index + 1).astype(str)
    last_slice_bread["Q111"] = last_slice_bread["Q111"].astype(str)
    last_slice_bread = last_slice_bread.merge(answers_q111, on="Q111", how="inner")
    # remove unused categories
    last_slice_bread["ans"] = last_slice_bread["ans"].cat.remove_unused_categories()

    # Plot
    plot_on_map(last_slice_bread, "What do you call the end of a loaf of bread?", "last_slice_bread_dist")


def plot_icing_frosting(ling_data, question_data):
    # map of Q094 icing or frosting answers
    icing_frosting = ling_data[(ling_data["Q094"].between(1, 6)) & (ling_data["long"] > -125)]
    answers_q094 = question_data["ans.94"]

    # Prepare to join
    answers_q094["Q094"] = (answers_q094.index + 1).astype(str)
    icing_frosting["Q094"] = icing_frosting["Q094"].astype(str)
    icing_frosting = icing_frosting.merge(answers_q094, on="Q094", how="inner")
    # remove unused categories
    icing_frosting["ans"] = icing_frosting["ans"].cat.remove_unused_categories()

    # rename the long ones so they plot better
    icing_frosting["ans"] = icing_frosting["ans"].cat.rename_categories(
        {
            "icing is thinner than frosting, white, and/or made of powdered sugar and milk or lemon juice ": "they are different",
        }
    )

    plot_on_map(
        icing_frosting,
        "Do you say 'frosting' or 'icing' for the sweet spread one puts on a cake?",
        "icing_frosting_dist",
    )


def crosstab_bread_icing(ling_data, question_data):
    # Q094 by Q111 row percent crosstab, latex table and heatmap
    labels_94 = dict(enumerate(question_data["ans.94"]["ans"].astype(str).str.strip(), start=1))
    labels_111 = dict(enumerate(question_data["ans.111"]["ans"].astype(str).str.strip(), start=1))
    labels_94[3] = "they are different"

    both = ling_data[ling_data["Q094"].between(1, 6) & ling_data["Q111"].between(1, 8)]
    # ct = pd.crosstab(both["Q094"].map(labels_94), both["Q111"].map(labels_111), normalize="index") * 100
    # ct.round(0).astype(int).to_latex(FIGS_DIR + "/crosstab_icing_bread.tex")
    icing = both["Q094"].map(labels_94)
    bread = both["Q111"].map(labels_111)

    ct = (pd.crosstab(icing, bread, normalize="index") * 100).round(0).astype(int)
    ct["n"] = icing.value_counts()
    ct.to_latex(FIGS_DIR + "/crosstab_icing_bread.tex")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(ct, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_xlabel("End of a loaf of bread")
    ax.set_ylabel("Icing or frosting")
    ax.set_title("% of each row's respondents giving each bread answer")
    plt.savefig(FIGS_DIR + "/crosstab_icing_bread.png", bbox_inches="tight")


def exploratory_figures_cleaned(ling_data, question_data):
    # all exploratory figures for cleaned ling data
    plot_last_slice_bread(ling_data, question_data)
    plot_icing_frosting(ling_data, question_data)

    crosstab_bread_icing(ling_data, question_data)


def plot_scree(d, title="scree"):
    # scree plot of variance explained for first 20 PCs
    prop_var = d**2 / np.sum(d**2)
    pcs = np.arange(1, 21)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pcs, prop_var[:20] * 100, marker="o", color="purple")
    ax.set_xticks(pcs)
    ax.set_xlabel("Principal Component (PC)")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree plot")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.savefig(FIGS_DIR + f"/{title}.png", bbox_inches="tight")


def plot_pc_on_map(pc_scores, pc, fig_name):
    # map of PC scores by location
    data = pc_scores.rename(columns={pc: "ans"})
    plot_on_map(data, f"{pc} score by location", fig_name, s=40, alpha=1, palette="YlGnBu", continuous=True, size="n")


def plot_choose_k(k_range, wss, sil):
    # elbow and silhouette plots for choosing k
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(k_range, wss, marker="o", color="purple")
    axes[0].set_ylabel("Within-cluster sum of squares")
    axes[0].set_title("Elbow plot")
    axes[1].plot(k_range, sil, marker="o", color="purple")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score")
    for ax in axes:
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(list(k_range))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR + "/choose_k.png", bbox_inches="tight")


def plot_clusters_on_map(pc_scores, method, k):
    # map of cluster labels by location
    data = pc_scores.sort_values(method).assign(ans=pc_scores[method].astype(str))
    plot_on_map(data, f"{method} Clusters (k={k})", f"{method}_map", s=40, alpha=1, palette="viridis", size="n")


def plot_dendrogram(X, k, fig_name="dendrogram"):
    # ward dendrogram with cut line at k clusters
    Z = linkage(X, method="ward")
    cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, color_threshold=cut, above_threshold_color="gray")
    ax.axhline(cut, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Ward distance")
    ax.set_title(f"Hierarchical clustering dendrogram (cut at k={k})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.savefig(FIGS_DIR + f"/{fig_name}.png", bbox_inches="tight")
