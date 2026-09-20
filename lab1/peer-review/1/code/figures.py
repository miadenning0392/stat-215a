from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from clean import clean_data, load_data

# paths
CODE_DIR = Path(__file__).resolve().parent
FIGS_DIR = CODE_DIR.parent / "figs"
DATA_PATH = CODE_DIR.parent / "data" / "TBI PUD 10-08-2013.csv"

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "pink": "#CC79A7",
    "black": "#000000",
    "gray": "#7F7F7F",
}

COLOR_AGE_UNDER2 = PALETTE["orange"]
COLOR_AGE_OVER2 = PALETTE["blue"]

COLOR_PRESENT = PALETTE["vermillion"]
COLOR_ABSENT = PALETTE["blue"]
COLOR_NEUTRAL = PALETTE["gray"]
COLOR_ACCENT = PALETTE["green"]


CONTINUOUS_CMAP = "cividis"


def configure_style():
    """Configure global plotting styles."""
    # Use TeX for rendering.
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Helvetica"

    # Make ticks face inwards and appear on the top and bottom of plots
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["axes.linewidth"] = 0.9

    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"


def save_figure(fig, name):
    """Save a figure as a PDF in the figures directory."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_DIR / f"{name}.pdf")
    plt.close(fig)


# defined based on data documentation
RACE_LABELS = {
    1: "White",
    2: "Black",
    3: "Asian",
    4: "Am. Indian/\nAlaskan Native",
    5: "Pacific\nIslander",
    90: "Other",
}
MECH_LABELS = {
    1: "Motor vehicle collision occupant",
    2: "Pedestrian struck",
    3: "Bike rider hit by auto",
    4: "Bike crash/fall",
    5: "Other wheeled transport",
    6: "Fall (ground level)",
    7: "Walked/ran into object",
    8: "Fall from elevation",
    9: "Fall down stairs",
    10: "Sports",
    11: "Assault",
    12: "Object struck head",
    90: "Other",
}


def build_eda_df(train_df):
    """Add labels and age groups for exploratory plots."""
    df = train_df.copy()

    df["age_group"] = np.where(df["age"] < 2, "<2 years", ">=2 years")
    df["age_group"] = pd.Categorical(
        df["age_group"], categories=["<2 years", ">=2 years"], ordered=True
    )
    df["race_label"] = df["race"].astype(float).map(RACE_LABELS)
    df["mechanism_label"] = df["injury_mechanism"].astype(float).map(MECH_LABELS)

    return df


def fig_age_distribution(df):
    """Plot the patient age distribution."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(df["age"], bins=36, color=COLOR_AGE_OVER2, edgecolor="white", linewidth=0.4)
    ax.axvline(2, color=PALETTE["black"], linestyle="--", linewidth=1)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of patients")
    ax.set_title("Age distribution")
    fig.tight_layout()
    save_figure(fig, "eda_age_distribution")


def fig_gcs_score(df):
    """Plot GCS score counts with ciTBI rates overlaid."""
    # number of patients at each GCS score
    gcs_counts = df["gcs_total"].value_counts().sort_index()

    # ciTBI rate at each GCS score
    gcs_rate = (
        df.groupby("gcs_total")["clinically_important_tbi"]
        .mean()
        .reindex(gcs_counts.index)
    )

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    # bars is number of patients
    ax1.bar(
        gcs_counts.index.astype(str),
        gcs_counts.values,
        color=COLOR_NEUTRAL,
        width=0.6,
        label="Number of patients",
    )
    ax1.set_xlabel("Total GCS score")
    ax1.set_ylabel("Number of patients")

    # second y-axis is for ciTBI rate
    ax2 = ax1.twinx()

    ax2.plot(
        gcs_counts.index.astype(str),
        gcs_rate.values,
        marker="o",
        color=COLOR_PRESENT,
        linewidth=2,
        label="ciTBI rate",
    )

    ax2.set_ylabel("ciTBI rate (in percent)")
    ax2.set_ylim(0, gcs_rate.max() * 1.2)

    # format as percentages
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    ax1.set_title("GCS score and ciTBI rate")

    fig.tight_layout()
    save_figure(fig, "eda_gcs_score")


def fig_injury_mechanism(df):
    """Plot injury mechanism counts and ciTBI rates by age group."""

    for age_group in ["<2 years", ">=2 years"]:

        # only keep patients in this age group
        d = df[df["age_group"] == age_group]

        # number of patients at each injury mechanism
        mech_counts = d["mechanism_label"].value_counts().sort_values()

        # ciTBI rate at each injury mechanism
        mech_rate = (
            d.groupby("mechanism_label")["clinically_important_tbi"]
            .mean()
            .reindex(mech_counts.index)
        )

        fig, ax1 = plt.subplots(figsize=(6.5, 5))

        # bars = number of patients
        ax1.barh(
            mech_counts.index,
            mech_counts.values,
            color=COLOR_NEUTRAL,
            label="Number of patients",
        )

        ax1.set_xlabel("Number of patients")
        ax1.set_ylabel("Injury mechanism")

        # second x-axis = ciTBI rate
        ax2 = ax1.twiny()

        ax2.plot(
            mech_rate.values,
            mech_rate.index,
            marker="o",
            color=COLOR_PRESENT,
            linewidth=2,
            label="ciTBI rate (percent)",
        )

        ax2.set_xlabel("ciTBI rate (percent)")

        # format as percentages
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2%}"))

        display_age = {"<2 years": "Under 2 years", ">=2 years": "2 years and older"}[
            age_group
        ]

        ax1.set_title(f"Injury mechanism and ciTBI rate ({display_age})")

        fig.tight_layout()
        save_figure(
            fig,
            f"eda_injury_mechanism_{age_group.replace(' ', '_').replace('<', 'under').replace('>=', 'over')}",
        )


def fig_outcome_rates(df):
    """Plot overall ciTBI and CT-use rates."""
    fig, ax = plt.subplots(figsize=(4, 4.5))
    rates = pd.Series(
        {
            "ciTBI": df["clinically_important_tbi"].mean(),
            "CT performed": df["ct_done"].mean(),
        }
    )
    bars = ax.bar(
        rates.index, rates.values, color=[COLOR_PRESENT, COLOR_ABSENT], width=0.55
    )
    for bar, v in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.015,
            f"{v:.2}",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of patients")
    ax.set_title("Overall ciTBI rate vs. CT use")
    fig.tight_layout()
    save_figure(fig, "eda_outcome_rates")


def fig_ct_use_by_race_age(df):
    """Plot CT-use rates by race and age group."""
    d = df.dropna(subset=["race_label"])  # drop patients with missing race value
    grp = (  # group by race and age
        d.groupby(["race_label", "age_group"])["ct_done"]
        .agg(k="sum", n="count")
        .reset_index()
    )
    grp["rate"] = grp["k"] / grp["n"]

    race_order = (  # race categories ordered by total sample size (largest first)
        grp.groupby("race_label")["n"].sum().sort_values(ascending=False).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    width = 0.38
    x = np.arange(len(race_order))
    colors = {"<2 years": COLOR_AGE_UNDER2, ">=2 years": COLOR_AGE_OVER2}

    legend_labels = {"<2 years": r"$<$ 2 years", ">=2 years": r"$\geq$ 2 years"}
    for i, ag in enumerate(["<2 years", ">=2 years"]):
        sub = (
            grp[grp["age_group"] == ag].set_index("race_label").reindex(race_order)
        )  # put the races in the correct order
        offset = (i - 0.5) * width
        ax.bar(
            x + offset, sub["rate"], width, label=legend_labels[ag], color=colors[ag]
        )
        for xi, rate, n in zip(
            x + offset, sub["rate"], sub["n"]
        ):  # write the sample sizes for clarity
            ax.text(
                xi,
                0.02,
                f"n={int(n)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(race_order)
    ax.set_ylabel("Proportion receiving a CT scan")
    ax.set_xlabel("Race")
    ax.set_ylim(0, 0.85)
    ax.set_title(
        "Non-black children and children over the age of 2 tend to have higher CT usage rates than their counterparts."
    )
    ax.legend(title="Age group", loc="upper right")

    fig.tight_layout()
    save_figure(fig, "ct_use_by_race_age")


def fig_clinical_risk_factors(df, perturb=False):
    """Plot ciTBI rates for clinical findings when present or absent."""

    indicators = {
        "Any loss of consciousness": df["loss_of_consciousness"].map(
            {0: 0, 1: 1, 2: 1}
        ),
        "Vomiting": df["vomiting"].astype("boolean"),
        "Seizure": df["seizure"].astype("boolean"),
        "Altered mental status": df["altered_mental_status"].astype("boolean"),
        "Palpable skull fracture": df["palpable_skull_fracture"].astype("boolean"),
        "Basilar skull fracture signs": df["basilar_skull_fracture"].astype("boolean"),
        "Scalp hematoma": df["hematoma"].astype("boolean"),
        "Headache": df["headache"].map(
            {0: 0, 1: 1}
        ),  # excludes pre-verbal/non-verbal (91)
        "Amnesia": df["amnesia"].map(
            {0: 0, 1: 1}
        ),  # excludes pre-verbal/non-verbal (91)
        "Not acting normally": (~df["acting_normally"].astype("boolean")).astype(
            "boolean"
        ),
        "High-severity injury mechanism": (df["injury_mechanism_severity"] == 3),
    }

    rows = []
    for label, ind in indicators.items():
        tmp = pd.DataFrame(
            {"ind": ind, "y": df["clinically_important_tbi"].astype(float)}
        ).dropna()  # we drop rows where either value is missing
        for level, status in [(1, "present"), (0, "absent")]:
            sub = tmp[tmp["ind"] == level]
            n, k = (
                len(sub),
                sub["y"].sum(),
            )  # compute number of patients in that group and also number of patients who had ciTBI
            rows.append(
                {
                    "finding": label,
                    "status": status,
                    "n": n,
                    "rate": k / n if n else np.nan,
                }
            )
    res = pd.DataFrame(rows)

    present = res[res["status"] == "present"].set_index("finding")
    absent = res[res["status"] == "absent"].set_index("finding")
    order = present["rate"].sort_values(ascending=False).index.tolist()
    present, absent = present.reindex(order), absent.reindex(order)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    y = np.arange(len(order))

    for yi, (r_absent, r_present) in enumerate(zip(absent["rate"], present["rate"])):
        ax.plot(
            [r_absent, r_present],
            [yi, yi],
            color=COLOR_NEUTRAL,
            linewidth=1.6,
            zorder=1,
        )

    ax.scatter(
        absent["rate"], y, color=COLOR_ABSENT, label="Finding absent", zorder=2, s=45
    )
    ax.scatter(
        present["rate"], y, color=COLOR_PRESENT, label="Finding present", zorder=3, s=45
    )

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xscale("log")
    ax.set_xlabel("Proportion with clinically-important TBI (log scale)")
    ax.set_ylabel("Clinical finding")
    ax.set_title(
        "Signs of skull fracture and altered mental status are associated with higher empirical ciTBI rates than vomiting, hematoma, or headache"
    )
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    ax.grid(axis="x", which="both", linestyle=":", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    if perturb:
        save_figure(fig, "clinical_risk_factors_perturbed")
    else:
        save_figure(fig, "clinical_risk_factors")


def main():
    """Generate and save all exploratory figures."""
    configure_style()

    raw = load_data(DATA_PATH)
    train_df, _ = clean_data(raw)
    df = build_eda_df(train_df)

    fig_age_distribution(df)
    fig_gcs_score(df)
    fig_injury_mechanism(df)
    fig_outcome_rates(df)
    fig_ct_use_by_race_age(df)
    fig_clinical_risk_factors(df)

    raw = load_data(DATA_PATH)
    train_df, _ = clean_data(
        raw, palpable_skull_fracture_handling="palpable_skull_fracture_missing"
    )
    df = build_eda_df(train_df)
    fig_clinical_risk_factors(df, perturb=True)


if __name__ == "__main__":
    main()
