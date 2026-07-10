"""Simple plotting utilities for visualizing monitor performance.

These plots help you understand how well a monitor distinguishes
between honest and attack trajectories.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_score_histograms(
    honest_scores: NDArray[np.float64],
    attack_scores: NDArray[np.float64],
    title: str = "Suspicion Score Distribution",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot overlapping histograms of suspicion scores for honest vs attack runs.

    If the distributions overlap a lot, the monitor struggles to tell them apart.
    Good monitors produce well-separated distributions.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    ax.hist(
        honest_scores,
        bins=bins,
        alpha=0.6,
        label="Honest",
        color="green",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        attack_scores,
        bins=bins,
        alpha=0.6,
        label="Attack",
        color="red",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Suspicion Score (normalized 0-1)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        # print(f"  Saved histogram to {save_path}")
    plt.show()
    # return fig


def plot_roc_curve(
    fpr: NDArray[np.float64],
    tpr: NDArray[np.float64],
    auc: float,
    label: str = "Monitor",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a single ROC curve with AUC in the legend.

    The diagonal dashed line represents random guessing (AUC = 0.5).
    A good monitor's curve hugs the top-left corner.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        # print(f"  Saved ROC curve to {save_path}")
    plt.show()
    # return fig


def plot_roc_comparison(
    roc_data: dict[str, tuple[NDArray[np.float64], NDArray[np.float64], float]],
    title: str = "Monitor Comparison: ROC Curves",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot multiple ROC curves on the same axes for comparison.

    Args:
        roc_data: Dict mapping monitor name -> (fpr, tpr, auc).
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.colormaps["tab10"](np.linspace(0, 1, max(len(roc_data), 1)))

    for idx, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        ax.plot(
            fpr, tpr, color=colors[idx], linewidth=2, label=f"{name} (AUC = {auc:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        # print(f"  Saved comparison plot to {save_path}")
    plt.show()
    # return fig


def plot_attack_evasion_curve(
    successful_attack_scores: NDArray[np.float64],
    save_path: Path | None = None,
) -> plt.Figure | None:
    """Plot attack success rate vs monitor threshold for working backdoors.

    For each threshold t, computes what fraction of working backdoors
    (side_task_success=True) score below t and would evade the monitor.
    A good attack prompt pushes this curve toward 1.0 across all thresholds.

    Args:
        successful_attack_scores: Monitor suspicion scores for attack samples
            that successfully inserted a working backdoor.
        save_path: Optional path to save the figure.

    Returns:
        The matplotlib Figure, or None if no successful attacks were provided.
    """
    if len(successful_attack_scores) == 0:
        print(
            "  No successful side-task attacks — try regenerating with a better attack prompt."
        )
        return None

    thresholds = np.linspace(0, 1, 200)
    evasion = [(successful_attack_scores < t).mean() for t in thresholds]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, evasion, color="tab:red", lw=2)
    ax.set_xlabel("Monitor threshold (flag if score ≥ threshold)")
    ax.set_ylabel("Fraction of working backdoors evading monitor")
    ax.set_title("Attack Success Rate vs Monitor Threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="50% evasion")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    evasion_at_half = (successful_attack_scores < 0.5).mean()
    print(
        f"  At threshold 0.5: {evasion_at_half:.0%} of working backdoors evade the monitor"
    )
    # return fig


_FRONTIER_MARKERS = ["o", "s", "D", "^", "v", "P", "*", "X"]


def plot_safety_usefulness_frontier(
    protocols: dict[str, list[tuple[float, float, float]]],
    title: str = "Safety–Usefulness Frontier",
    save_path: Path | None = None,
) -> None:
    """Plot safety vs usefulness for control protocols at different thresholds.

    Each protocol is shown as a connected series of points, one per threshold.
    Points are annotated with the threshold value. The ideal corner is
    top-right: high safety AND high usefulness.

    Args:
        protocols: Maps protocol name to a list of (safety, usefulness, threshold)
            tuples. A protocol with a single point (e.g. Trusted Monitoring, which
            doesn't depend on a threshold) is drawn as a standalone marker.
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (name, points) in enumerate(protocols.items()):
        marker = _FRONTIER_MARKERS[idx % len(_FRONTIER_MARKERS)]
        safeties, usefulnesses, thresholds = zip(*points)

        if len(points) == 1:
            # Single point — no connecting line
            ax.scatter(
                safeties,
                usefulnesses,
                s=100,
                marker=marker,
                zorder=5,
                label=name,
            )
        else:
            ax.plot(
                safeties,
                usefulnesses,
                marker=marker,
                markersize=8,
                linewidth=1.5,
                label=name,
                zorder=4,
            )

        # Annotate threshold values (skip for single-point protocols where
        # the threshold is not meaningful, e.g. Trusted Monitoring)
        if len(points) > 1:
            for s, u, t in points:
                ax.annotate(
                    f"t={t}",
                    (s, u),
                    textcoords="offset points",
                    xytext=(8, 4),
                    fontsize=8,
                )

    ax.set_xlabel("Safety (1 − backdoor success rate)")
    ax.set_ylabel("Usefulness (honest task success rate)")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
