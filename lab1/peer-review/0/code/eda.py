import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap 

def create_ct_tbi_outcomes_fig(clean_train):
    """
    Create CT and TBI Outcomes figure.
    """
        
    counts = [len(clean_train),
              (clean_train["ct_planned"] == 1).sum(),
              (clean_train["ct_performed"] == 1).sum(),
              (clean_train["tbi_on_ct"] == 1).sum(),
              (clean_train["ci_tbi"] == 1).sum()]

    bars = plt.bar(["total patients", "CT planned", "CT performed","TBI on CT", "clinically important TBI"], 
                   counts, color = "lightseagreen")

    plt.bar_label(bars, padding = -1)
    plt.ylabel("Number of patients")
    plt.title("CT and TBI Outcomes")
    plt.xticks(rotation = 40, ha = "right")
    plt.tight_layout()

    return plt.gcf()


def create_top_ct_indicators_fig(clean_train):
    """
    Create Top Indications Influencing the Decision to Obtain a CT figure.
    """
    
    ct_planned_vars = {
        "ct_planned_bc_young_age": "young age",
        "ct_planned_bc_amnesia": "amnesia",
        "ct_planned_bc_mental_stat": "decreased mental status",
        "ct_planned_bc_skull_frac": "skull fracture",
        "ct_planned_bc_headache": "headache",
        "ct_planned_bc_hematoma": "scalp hematoma",
        "ct_planned_bc_loss_cons": "loss of consciousness",
        "ct_planned_bc_mech": "injury mechanism",
        "ct_planned_bc_neuro_deficit": "neurological deficit",
        "ct_planned_bc_md_rqst": "MD request",
        "ct_planned_bc_parent_rqst": "parent request",
        "ct_planned_bc_trauma_team_rqst": "trauma team request",
        "ct_planned_bc_seizure": "seizure",
        "ct_planned_bc_vomiting": "vomiting",
        "ct_planned_bc_xray": "x-ray",
        "ct_planned_bc_other": "other"
    }

    ct_planned_data = clean_train.loc[clean_train["ct_planned"] == 1,
                                      ["age_less_two"] + list(ct_planned_vars.keys())]

    ct_planned_data[list(ct_planned_vars.keys())] = (ct_planned_data[list(ct_planned_vars.keys())] == 1)

    ct_planned_plot = (ct_planned_data
                       .groupby("age_less_two")[list(ct_planned_vars.keys())]
                       .mean()
                       .transpose()
                       .rename_axis(columns = "ct_planned_ind"))

    top_under_two = ct_planned_plot[1].sort_values(ascending = False).head(5)
    top_over_two = ct_planned_plot[2].sort_values(ascending = False).head(5)

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 5), sharey = True)

    # < 2 years
    axes[0].bar([ct_planned_vars[var] for var in top_under_two.index], top_under_two, color = "lightseagreen")
    axes[0].set_title("<2 years")
    axes[0].set_ylabel("Proportion")
    plt.setp(axes[0].get_xticklabels(), rotation = 40, ha = "right")

    # >= 2 years
    axes[1].bar([ct_planned_vars[var] for var in top_over_two.index], top_over_two, color = "lightseagreen")
    axes[1].set_title("≥2 years")
    plt.setp(axes[1].get_xticklabels(), rotation = 40, ha = "right")

    fig.suptitle("Top Indications Influencing the Decision to Obtain a CT")
    fig.tight_layout()

    return fig


def create_injury_mech_fig(clean_train):
    """
    Create Injury Mechanisms Influencing the Decision to Obtain a CT figure.
    """

    injury_mech_levels = {
        1: "motor vehicle collision",
        2: "pedestrian struck by vehicle",
        3: "biker struck by vehicle",
        4: "bike collision/fall",
        5: "other wheeled transport crash",
        6: "fall from standing",
        7: "ran into stationary object",
        8: "fall from elevation",
        9: "fall down stairs",
        10: "sports",
        11: "assault",
        12: "object struck head",
        13: "other"
    }

    ct_planned_mech = (clean_train.loc[(clean_train["ct_planned"] == 1) &
                                       (clean_train["ct_planned_bc_mech"] == 1), "injury_mech"]
                       .value_counts(normalize = True)
                       .sort_values(ascending = False))

    plt.bar([injury_mech_levels[lev] for lev in ct_planned_mech.index], 
            ct_planned_mech.values, color = "lightseagreen")

    plt.ylabel("Proportion")
    plt.title("Injury Mechanisms Influencing the Decision to Obtain a CT")
    plt.xticks(rotation = 40, ha = "right")
    plt.tight_layout()

    return plt.gcf()


def create_injury_mech_severity_fig(clean_train):
    """
    Create Clinically Important TBI by Injury Mechanism Severity figure.
    """

    injury_mech_levels = {
        1: "motor vehicle collision",
        2: "pedestrian struck by vehicle",
        3: "biker struck by vehicle",
        8: "fall from elevation",
        10: "sports",
        11: "assault",
        12: "object struck head",
        13: "other"
    }

    mech_sev_data = (clean_train.loc[(clean_train["gcs_total"] >= 14) &
                                     (~clean_train["injury_mech"].isin([4, 5, 6, 7, 9]))]
                    .groupby(["injury_mech", "injury_mech_severity"])["ci_tbi"]
                    .apply(lambda x: (x == 1).mean())
                    .reset_index(name = "proportion"))

    order_mechs = (mech_sev_data
                   .groupby("injury_mech", observed = True)["proportion"]
                   .agg(lambda x: x.max() - x.min())
                   .sort_values()
                   .index)

    fig, ax = plt.subplots(figsize=(8, 5))

    for prop, mech in enumerate(order_mechs):

        inj_mech = mech_sev_data.loc[mech_sev_data["injury_mech"] == mech]
        moderate = inj_mech.loc[inj_mech["injury_mech_severity"] == 2, "proportion"].iloc[0]
        severe = inj_mech.loc[inj_mech["injury_mech_severity"] == 3, "proportion"].iloc[0]

        ax.plot([moderate, severe], [prop, prop], color = "darkgrey",
                linewidth = 2.5 if moderate > severe else 1)

        ax.scatter(moderate, prop, color = "cornflowerblue", s = 80, zorder = 3)
        ax.scatter(severe, prop, color = "firebrick", s = 80, zorder = 3)

    ax.scatter([], [], color = "cornflowerblue", s = 80, label = "Moderate severity")
    ax.scatter([], [], color = "firebrick", s = 80, label = "High severity")
    ax.legend()
    ax.set_yticks(range(len(order_mechs)))
    ax.set_yticklabels([injury_mech_levels[mech] for mech in order_mechs])
    ax.set_xlabel("Proportion with clinically important TBI")
    ax.set_title("Clinically Important TBI by Injury Mechanism Severity")
    plt.tight_layout()

    return plt.gcf()


def create_vomiting_fig(clean_train):
    """
    Create Clinically Important TBI by Vomiting Characteristics figure.
    """

    vomit_data = clean_train.loc[(clean_train["gcs_total"] >= 14) &
                                 (clean_train["vomiting"] == 1)].copy()

    heatmap_under_two = (vomit_data.loc[vomit_data["age_less_two"] == 1]
                        .groupby(["vomiting_ep_count", "vomiting_start"])["ci_tbi"]
                        .apply(lambda x: (x == 1).mean())
                        .unstack()
                        .reindex(index=[3, 2, 1]))

    heatmap_over_two = (vomit_data.loc[vomit_data["age_less_two"] == 2]
                        .groupby(["vomiting_ep_count", "vomiting_start"])["ci_tbi"]
                        .apply(lambda x: (x == 1).mean())
                        .unstack()
                        .reindex(index=[3, 2, 1]))

    vmin = min(heatmap_under_two.min().min(), heatmap_over_two.min().min())
    vmax = max(heatmap_under_two.max().max(), heatmap_over_two.max().max())
    gray_red = LinearSegmentedColormap.from_list("gray_red", ["lightgray", "firebrick"])

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(14, 4), sharey = True)

    # < 2 years
    im = axes[0].imshow(heatmap_under_two, aspect = "equal", cmap = gray_red, vmin = vmin, vmax = vmax)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(["before injury", "within 1 hour", "1-4 hours after", ">4 hours after"], rotation = 30, ha = "right")
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels([">2", "2", "1"])

    axes[0].set_xlabel("Vomiting start")
    axes[0].set_ylabel("Number of vomiting episodes")
    axes[0].set_title("<2 years")

    # >= 2 years
    axes[1].imshow(heatmap_over_two, aspect = "equal", cmap = gray_red, vmin = vmin, vmax = vmax)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(["before injury", "within 1 hour", "1-4 hours after", ">4 hours after"], rotation = 30, ha = "right")
    axes[1].set_xlabel("Vomiting start")
    axes[1].set_title("≥2 years")

    for i in range(3):

        for j in range(4):

            prop_under_two = heatmap_under_two.iloc[i, j]
            axes[0].text(j, i, f"{prop_under_two:.3f}", ha = "center", va = "center")

            prop_over_two = heatmap_over_two.iloc[i, j]
            axes[1].text(j, i, f"{prop_over_two:.3f}", ha = "center", va = "center")

    fig.colorbar(im, ax = axes, label = "Proportion with clinically important TBI")
    fig.suptitle("Clinically Important TBI by Vomiting Characteristics")

    return plt.gcf()


def create_vomiting_stability_fig(clean_train, perturbed_train):
    """
    Create Stability Check for Clinically Important TBI by Vomiting Characteristics figure.
    """

    heatmaps = []

    for data in [clean_train, perturbed_train]:

        vomit_data = data.loc[(data["gcs_total"] >= 14) &
                              (data["vomiting"] == 1)].copy()

        heatmap_under_two = (vomit_data.loc[vomit_data["age_less_two"] == 1]
                            .groupby(["vomiting_ep_count", "vomiting_start"])["ci_tbi"]
                            .apply(lambda x: (x == 1).mean())
                            .unstack()
                            .reindex(index=[3, 2, 1]))

        heatmap_over_two = (vomit_data.loc[vomit_data["age_less_two"] == 2]
                            .groupby(["vomiting_ep_count", "vomiting_start"])["ci_tbi"]
                            .apply(lambda x: (x == 1).mean())
                            .unstack()
                            .reindex(index=[3, 2, 1]))

        heatmaps.append((heatmap_under_two, heatmap_over_two))

    vmin = min(h.min().min() for pair in heatmaps for h in pair)
    vmax = max(h.max().max() for pair in heatmaps for h in pair)
    gray_red = LinearSegmentedColormap.from_list("gray_red", ["lightgray", "firebrick"])

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (14, 8), sharex = True, sharey = True)

    for row, (heatmap_under_two, heatmap_over_two) in enumerate(heatmaps):

        im = axes[row, 0].imshow(heatmap_under_two, aspect = "equal", cmap = gray_red, vmin = vmin, vmax = vmax)
        axes[row, 1].imshow(heatmap_over_two, aspect = "equal", cmap = gray_red, vmin = vmin, vmax = vmax)

        for i in range(3):

            for j in range(4):

                prop_under_two = heatmap_under_two.iloc[i, j]
                axes[row, 0].text(j, i, f"{prop_under_two:.3f}", ha = "center", va = "center")

                prop_over_two = heatmap_over_two.iloc[i, j]
                axes[row, 1].text(j, i, f"{prop_over_two:.3f}", ha = "center", va = "center")

    axes[0, 0].set_title("<2 years")
    axes[0, 1].set_title("≥2 years")
    axes[0, 0].set_ylabel("Before perturbation\n\nNumber of vomiting episodes")
    axes[1, 0].set_ylabel("After perturbation\n\nNumber of vomiting episodes")
    axes[0, 0].set_yticks(range(3))
    axes[0, 0].set_yticklabels([">2", "2", "1"])

    for ax in axes[1, :]:
        ax.set_xticks(range(4))
        ax.set_xticklabels(["before injury", "within 1 hour", "1-4 hours after", ">4 hours after"],
                           rotation = 30, ha = "right")
        ax.set_xlabel("Vomiting start")

    fig.colorbar(im, ax = axes, label = "Proportion with clinically important TBI")
    fig.suptitle("Stability Check for Clinically Important TBI by Vomiting Characteristics")

    return fig


def plot_logistic_coefs(usual_coef_data, perturbed_coef_data, title):
    """
    Create logistic regression coefficient figure.
    """

    plt.figure(figsize=(8, 5))

    var_names = {
        "loc_gt_5sec": "loss of consciousness >= 5 sec",
        "gcs_14": "GCS score = 14",
        "scalp_hematoma": "scalp hematoma",
        "skull_fracture": "palpable skull fracture",
        "vomiting_gt_2": "vomiting episodes >2",
        "severe_mech": "severe mechanism of injury",
        "alt_mental_stat": "altered mental status",
        "struck_by_vehicle": "stuck by vehicle",
        "acting_normal": "acting normal"
    }

    coef_data = (pd.merge(usual_coef_data, perturbed_coef_data, on = "variable", suffixes=("_usual", "_perturbed"))
                .sort_values("coefficient_usual")
                .reset_index(drop=True))

    coef_data["variable"] = coef_data["variable"].map(var_names)
    y_position = np.arange(len(coef_data))

    for i in range(len(coef_data)):

        plt.plot([coef_data.loc[i, "coefficient_usual"],
                coef_data.loc[i, "coefficient_perturbed"]],
                [i, i], color = "darkgray", linewidth = 1.5, zorder = 1)

    plt.scatter(coef_data["coefficient_usual"], y_position, 
                label = "Before perturbation", color = "rebeccapurple", s = 40)

    plt.scatter(coef_data["coefficient_perturbed"], y_position, 
                label = "After perturbation", color = "darkorange", s = 40)

    plt.axvline(0, color = "darkgrey", linestyle = "--", linewidth = 1)
    plt.yticks(y_position, coef_data["variable"])
    plt.xlabel("Logistic regression coefficient")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()