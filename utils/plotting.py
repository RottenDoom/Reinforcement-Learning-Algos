"""Matplotlib figure functions for the RL Gridworld project.

All public functions accept an optional `ax` argument so callers can compose
figures freely. `save_figure` is the single exit point for file output.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project constants
# --------------------------------------------------------------------------- #

_INV_ST: list[int] = [9, 17, 25, 33, 34, 42, 50, 12, 20, 28, 29, 30, 38, 46, 54]
_TERMINAL: int = 36
_VALID_STATES: list[int] = [s for s in range(64) if s not in _INV_ST and s != _TERMINAL]

FIGURES_DIR: Path = Path(__file__).resolve().parent.parent / "report" / "figures"

# 9 actions: N NE E SE S SW W NW Stay
_ACTION_SYMBOLS: list[str] = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖", "●"]


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #

def save_figure(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    """Save *fig* to report/figures/{name}.png and .pdf.  Returns the PNG path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{name}.png"
    pdf = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png


# --------------------------------------------------------------------------- #
# Learning-curve helpers
# --------------------------------------------------------------------------- #

def plot_learning_curves(
    results: dict,
    metric: str = "abs_err",
    label: str = "",
    ax: plt.Axes | None = None,
    color: str | None = None,
    smooth: int = 1,
) -> plt.Axes:
    """Plot mean ± 1 std of *metric* across runs for one algorithm.

    Args:
        results:  dict loaded from a .npy file (keys: signed_err, abs_err,
                  policy_err, checkpoint_eps, n_runs, n_episodes).
        metric:   "signed_err" | "abs_err" | "policy_err".
        smooth:   running-mean window applied before plotting (1 = no smoothing).
    """
    if ax is None:
        _, ax = plt.subplots()

    if metric == "policy_err":
        x    = results["checkpoint_eps"] + 1   # 1-indexed
        data = results["policy_err"]
    else:
        x    = np.arange(1, int(results["n_episodes"]) + 1)
        data = results[metric]

    mean = data.mean(axis=0)
    std  = data.std(axis=0)

    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        mean = np.convolve(mean, kernel, mode="same")
        std  = np.convolve(std,  kernel, mode="same")

    kw: dict = {} if color is None else {"color": color}
    ax.plot(x, mean, label=label, **kw)
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, **kw)
    return ax


def plot_multi_curves(
    results_map: dict[str, dict],
    metric: str = "abs_err",
    title: str = "",
    xlabel: str = "Episode",
    ylabel: str | None = None,
    figsize: tuple[int, int] = (9, 5),
    colors: list[str] | None = None,
    smooth: int = 1,
) -> plt.Figure:
    """Plot *metric* curves for multiple algorithms on one figure.

    Args:
        results_map:  {display_name: results_dict}.
        metric:       "signed_err" | "abs_err" | "policy_err".
    """
    _ylabel_defaults = {
        "abs_err":    "Mean Absolute Relative Error",
        "signed_err": "Mean Signed Relative Error",
        "policy_err": "Policy Evaluation Error",
    }
    ylabel = ylabel or _ylabel_defaults.get(metric, metric)

    fig, ax = plt.subplots(figsize=figsize)
    palette = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, res) in enumerate(results_map.items()):
        plot_learning_curves(
            res, metric=metric, label=name, ax=ax,
            color=palette[i % len(palette)], smooth=smooth,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Grid-world visualisation helpers
# --------------------------------------------------------------------------- #

def _grid_mask() -> np.ndarray:
    """Return (8, 8) bool mask — True where state is blocked or terminal."""
    m = np.zeros(64, dtype=bool)
    for s in _INV_ST:
        m[s] = True
    m[_TERMINAL] = True
    return m.reshape(8, 8)


def plot_value_heatmap(
    V_est: np.ndarray,
    title: str = "V(s)",
    ax: plt.Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar: bool = True,
) -> plt.Axes:
    """Plot *V_est* (shape 64) as an 8×8 heatmap.

    Blocked and terminal cells are coloured gray and gold respectively.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    grid = V_est.reshape(8, 8).astype(float).copy()
    grid[_grid_mask()] = np.nan

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#cccccc")

    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Highlight terminal cell
    tr, tc = divmod(_TERMINAL, 8)
    ax.add_patch(
        plt.Rectangle((tc - 0.5, tr - 0.5), 1, 1, color="gold", alpha=0.7, zorder=2)
    )
    ax.text(tc, tr, "G", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_title(title, fontsize=11)
    return ax


def plot_vstar_heatmap(ax: plt.Axes | None = None) -> plt.Axes:
    """Convenience: plot the known optimal V* (Fox 2016) as a heatmap."""
    from utils.metrics import V_STAR
    return plot_value_heatmap(V_STAR, title="Optimal V* (Fox 2016)", ax=ax, vmin=0, vmax=6)


def plot_policy_arrows(
    policy: np.ndarray,
    title: str = "Greedy policy",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualise a (64, 9) policy as arrow glyphs on the 8×8 grid.

    The highest-probability action is shown at each valid state.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    inv_set = set(_INV_ST)

    for r in range(8):
        for c in range(8):
            s = r * 8 + c
            if s in inv_set:
                ax.add_patch(
                    plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#cccccc", zorder=1)
                )

    # Terminal cell
    tr, tc = divmod(_TERMINAL, 8)
    ax.add_patch(
        plt.Rectangle((tc - 0.5, tr - 0.5), 1, 1, color="gold", alpha=0.7, zorder=2)
    )
    ax.text(tc, tr, "G", ha="center", va="center", fontsize=9, fontweight="bold")

    for r in range(8):
        for c in range(8):
            s = r * 8 + c
            if s in inv_set or s == _TERMINAL:
                continue
            best_a = int(np.argmax(policy[s, :]))
            ax.text(c, r, _ACTION_SYMBOLS[best_a], ha="center", va="center", fontsize=10)

    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)   # row 0 at top
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_title(title, fontsize=11)
    ax.grid(True, color="lightgray", linewidth=0.5)
    return ax


# --------------------------------------------------------------------------- #
# Summary / comparison figures
# --------------------------------------------------------------------------- #

def plot_summary_bar(
    final_errors: dict[str, float],
    metric_label: str = "Final Policy Eval Error",
    title: str = "Algorithm comparison",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Bar chart comparing algorithms on a single scalar metric."""
    fig, ax = plt.subplots(figsize=figsize)
    names = list(final_errors.keys())
    vals  = [final_errors[n] for n in names]
    colors = plt.cm.tab10.colors[:len(names)]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_multi_heatmap_grid(
    agents_values: list[tuple[str, np.ndarray]],
    n_cols: int = 3,
    vmin: float = 0.0,
    vmax: float = 6.0,
    suptitle: str = "Value Function V(s)",
    figsize_per_cell: tuple[int, int] = (4, 4),
) -> plt.Figure:
    """Lay out multiple value heatmaps in a grid.

    Args:
        agents_values: list of (label, V_est) pairs.
    """
    n = len(agents_values)
    n_rows = (n + n_cols - 1) // n_cols
    fw = figsize_per_cell[0] * n_cols
    fh = figsize_per_cell[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw, fh))
    axes = np.array(axes).reshape(-1)

    for i, (label, V) in enumerate(agents_values):
        plot_value_heatmap(V, title=label, ax=axes[i], vmin=vmin, vmax=vmax, cbar=False)

    for j in range(len(agents_values), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_multi_policy_grid(
    agents_policies: list[tuple[str, np.ndarray]],
    n_cols: int = 3,
    suptitle: str = "Greedy Policies",
    figsize_per_cell: tuple[int, int] = (4, 4),
) -> plt.Figure:
    """Lay out multiple policy-arrow grids in one figure."""
    n = len(agents_policies)
    n_rows = (n + n_cols - 1) // n_cols
    fw = figsize_per_cell[0] * n_cols
    fh = figsize_per_cell[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw, fh))
    axes = np.array(axes).reshape(-1)

    for i, (label, pi) in enumerate(agents_policies):
        plot_policy_arrows(pi, title=label, ax=axes[i])

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
