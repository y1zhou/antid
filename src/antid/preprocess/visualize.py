"""Functions for visualizing sequence/structure-related datasets."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from antid.utils.constant import AA1_BY_PROP, AA1_GROUP_LABEL


def rearrange_amino_acids(df: pl.DataFrame) -> pl.DataFrame:
    """Reorder amino acids by their physiochemical properties.

    Args:
        df: DataFrame with columns as amino acids (1-letter codes).

    Returns:
        20 x n DataFrame with rows ordered according to `AA1_BY_PROP`.
    """
    for aa in AA1_BY_PROP:
        if aa not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(aa))
    return (
        df.with_row_index(name="Position", offset=1)
        .unpivot(
            index="Position", on=AA1_BY_PROP, variable_name="aa", value_name="value"
        )
        .with_columns(pl.col("aa").cast(pl.Enum(AA1_BY_PROP)))
        .sort("aa", descending=True)
        .pivot(on="aa", index="Position", values="value")
        .sort("Position")
    )


def mutational_heatmap(
    df: pl.DataFrame,
    wt_seq: str,
    value_label: str = "Score",
    cmap: str = "icefire",
    value_center: float | None = 0.0,
    shrink_cbar: float = 0.6,
    xlabel: str = "Position",
    df_se: pl.DataFrame | None = None,
    df_highlight: pl.DataFrame | None = None,
    highlight_color: str = "yellow",
    subplots_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    """Make a heatmap from the given DataFrame.

    Args:
        df: Each row should be a site in the sequence, and each column should
            be one of the 20 amino acids, * (stop codon in the DNA sequence),
            or X (N in DNA sequence).
        wt_seq: Wild-type sequence whose length should equal the number of
            rows in ``df``.
        value_label: Annotation of what the values represent in ``df``.
            This is used as the label for the color bar.
        cmap: Color map to use. Set to ``magma`` for sequential scores, and
            ``icefire`` for diverging scores.
        value_center: The value at which to center the colormap when plotting
            divergent data.
        shrink_cbar: scale of the coloar bar relative to the heatmap.
        xlabel: Text label of the x-axis.
        df_se: Same shape as ``df``, but with values indicating the standard
            errors of each score in ``df``. This is used to draw the diagonal
            lines in each cell.
        df_highlight: Highlight cells with ``True`` values.
        highlight_color: Color of the box highlight.
        subplots_kwargs: Additional keyword arguments passed to ``plt.subplots()``.

    Returns:
        The matplotlib Figure and Axes objects.
    """
    # Reorder rows in df to match the amino acid properties
    dat = rearrange_amino_acids(df)

    # If df_se is given, ensure same ordering as dat
    draw_error_bars = False
    if isinstance(df_se, pl.DataFrame):
        draw_error_bars = True
        dat_se = rearrange_amino_acids(df_se)
        if dat.shape != dat_se.shape:
            raise ValueError(
                "Shape mismatch between df and df_se", dat.shape, dat_se.shape
            )

        # Rescale the SE's onto 0 .. 0.98
        # because 0 .. 1.0 causes the corners to look funny
        dat_se_max = np.abs(dat_se.drop("Position").to_numpy()).max().item()
        dat_se = (
            dat_se.select(
                "Position", pl.exclude("Position") / pl.lit(dat_se_max) * pl.lit(0.98)
            )
            .to_pandas()
            .set_index("Position")
            .T
        )

    draw_boxes = False
    if isinstance(df_highlight, pl.DataFrame):
        draw_boxes = True
        dat_hl = rearrange_amino_acids(df_highlight)
        if dat.shape != dat_hl.shape:
            raise ValueError(
                "Shape mismatch between df and df_se", dat.shape, dat_hl.shape
            )
        dat_hl = dat_hl.to_pandas().set_index("Position").T

    # Make heatmap
    default_kwargs: dict[str, Any] = {
        "figsize": (10, 8),
        "dpi": 200,
        "layout": "constrained",
    }
    if subplots_kwargs is None:
        plt_kwargs = default_kwargs
    else:
        plt_kwargs = default_kwargs | subplots_kwargs

    fig, ax = plt.subplots(**plt_kwargs)
    pos_with_wt_aa = [
        f"{aa}{i}" for i, aa in zip(dat.get_column("Position"), wt_seq, strict=True)
    ]
    g = sns.heatmap(
        dat.with_columns(pl.Series("Position", pos_with_wt_aa))
        .to_pandas()
        .set_index("Position")
        .T,
        cmap=cmap,
        center=value_center,
        square=True,
        linecolor="gray",
        linewidths=0.5,
        cbar_kws={"label": value_label, "shrink": shrink_cbar, "pad": 0.02},
        ax=ax,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=-90, va="top", ha="center", font="monospace"
    )
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_facecolor("tab:gray")
    g.set(xlabel=xlabel)

    # Label amino acid properties
    num_aa = dat.width - 1  # Exclude "Position" column
    for label, start, end in AA1_GROUP_LABEL:
        ax.text(
            -1.5,
            num_aa - ((end - start + 1) / 2.0 + start),
            label,
            fontsize=9,
            rotation="vertical",
            verticalalignment="center",
            horizontalalignment="right",
            transform=ax.transData,
        )
        y_start, y_end = num_aa - (start + 0.125), num_aa - (end + 1 - 0.125)
        bar = Line2D(
            [-1.25, -1.25],
            [y_start, y_end],
            transform=ax.transData,
            color="grey",
        )
        bar.set_clip_on(False)
        ax.add_line(bar)
        ax.axhline(y=y_end - 0.1, color="white", alpha=0.7, linewidth=2.0)

    # Highlight WT sequence (dot) and outline high-freq amino acids
    aa = dat.drop("Position").columns
    for j in range(dat.height):
        wt_aa = wt_seq[j]

        for i, mut_aa in enumerate(aa):
            if wt_aa == mut_aa:
                ax.add_patch(
                    Circle(
                        (j + 0.5, i + 0.5),
                        0.1666,
                        fill=True,
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.8,
                    )
                )

            # Draw error bars if values for the mutant aren't too small
            if draw_error_bars and (dat_se.iloc[i, j] >= 0.02) and (wt_aa != mut_aa):
                corner_dist = (1.0 - dat_se.iloc[i, j]) / 2.0
                diag = Line2D(
                    [j + corner_dist, j + 1 - corner_dist],
                    [i + corner_dist, i + 1 - corner_dist],
                    transform=ax.transData,
                    color="grey",
                    linewidth=0.8,
                )
                ax.add_line(diag)

            # Highlight significant p-values
            if draw_boxes and (dat_hl.iloc[i, j]) and (wt_aa != mut_aa):
                ax.add_patch(
                    Rectangle((j, i), 1, 1, fill=False, edgecolor=highlight_color, lw=1)
                )

    return fig, ax
