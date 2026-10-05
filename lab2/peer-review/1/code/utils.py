from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_states(data_dir=DATA_DIR):
    '''US state outlines, including AK and HI'''
    states = gpd.read_file(data_dir / "shapefiles")
    return states[states["iso_a2"] == "US"]

def make_us_map(states, figsize=(12, 6)):
    fig = plt.figure(figsize=figsize)
    axes = {
        "contiguous": fig.add_axes([0, 0.25, 1, 0.75]),
        "alaska": fig.add_axes([0, 0, 0.2, 0.25]),
        "hawaii": fig.add_axes([0.2, 0, 0.15, 0.25]),
    }
    map_limits = {
        "contiguous": ((-125, -66), (24, 50)),
        "alaska": ((-170, -140), (54, 72)),
        "hawaii": ((-161, -154), (18.5, 22.5)),
    }
    for region, ax in axes.items():
        states.boundary.plot(ax=ax, color="gray", linewidth=0.5, zorder=2)
        xlim, ylim = map_limits[region]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect(1.3)
        ax.axis("off")
    return fig, axes

def draw_bins(ax, lat, long, colors):
    '''Draw one square per bin'''
    for y, x, color in zip(lat, long, colors):
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="none", zorder=1))

def map_top_label(clean_df, labels, names, states, min_respondents=5):
    '''Map of the most common label (an answer code, a cluster, ...) in each bin.
    names maps each label to its legend text. Bins with a weak majority are faded.'''
    df = pd.DataFrame({"lat": np.floor(clean_df["lat"]),
                       "long": np.floor(clean_df["long"]),
                       "label": labels})

    # label counts per bin: one row per bin, one column per label
    counts = df.groupby(["lat", "long"])["label"].value_counts().unstack(fill_value=0)
    counts = counts[counts.sum(axis=1) >= min_respondents]
    percents = counts.div(counts.sum(axis=1), axis=0) * 100
    top = percents.idxmax(axis=1)
    top_percent = percents.max(axis=1)
    lat = top.index.get_level_values(0)
    long = top.index.get_level_values(1)

    # one color per answer that wins at least one bin
    palette = plt.get_cmap("tab10").colors
    winners = sorted(top.unique())
    color_of = {code: palette[i] for i, code in enumerate(winners)}
    # opacity = percentage giving the top answer (as a 0-1 alpha), at least 0.15 so bins stay visible
    colors = [color_of[code] + (max(p / 100, 0.15),) for code, p in zip(top, top_percent)]

    fig, axes = make_us_map(states)
    for ax in axes.values():
        draw_bins(ax, lat, long, colors)

    handles = [Rectangle((0, 0), 1, 1, color=color_of[code]) for code in winners]
    axes["contiguous"].legend(handles, [names[code] for code in winners],
                              loc="lower left", fontsize=11, frameon=False)
    return fig, axes

def map_values(clean_df, values, states, min_respondents=5, cmap="viridis", label=None):
    '''Map of a per-person number (a PC score, 1/0 for an answer, ...) averaged in each bin'''
    df = pd.DataFrame({"lat": np.floor(clean_df["lat"]),
                       "long": np.floor(clean_df["long"]),
                       "value": values})
    df = df.dropna()
    means = df.groupby(["lat", "long"])["value"].agg(["mean", "count"])
    means = means[means["count"] >= min_respondents]
    lat = means.index.get_level_values(0)
    long = means.index.get_level_values(1)

    norm = plt.Normalize(means["mean"].min(), means["mean"].max())
    colors = plt.get_cmap(cmap)(norm(means["mean"]))

    fig, axes = make_us_map(states)
    for ax in axes.values():
        draw_bins(ax, lat, long, colors)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes["contiguous"],
                 shrink=0.6, label=label)
    return fig, axes
