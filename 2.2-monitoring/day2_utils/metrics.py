"""ROC and AUC metrics for evaluating monitor performance.

These functions compute Receiver Operating Characteristic (ROC) curves and
Area Under the Curve (AUC) scores. They tell us how well a monitor can
distinguish between honest and attack trajectories.

Key concepts:
- True Positive Rate (TPR): fraction of attacks correctly flagged
- False Positive Rate (FPR): fraction of honest runs incorrectly flagged
- A perfect monitor has AUC = 1.0; random guessing gives AUC = 0.5

Adapted from control_arena/examples/apps_monitor_comparison/metrics.py
using numpy only (no sklearn dependency).
"""

import numpy as np
from numpy.typing import NDArray


def calculate_roc_curve(
    y_true: NDArray[np.int_],
    y_scores: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute ROC curve by sweeping over all unique score thresholds.

    For each threshold t, we classify samples with score >= t as "attack".
    Then we compute the resulting TPR and FPR.

    Args:
        y_true: Binary labels (1 = attack, 0 = honest).
        y_scores: Suspicion scores from the monitor (higher = more suspicious).

    Returns:
        (fpr, tpr, thresholds): Arrays of false positive rates, true positive
        rates, and the corresponding score thresholds.
    """
    # Get unique thresholds from highest to lowest, plus one below the minimum
    thresholds = np.unique(y_scores)[::-1]
    thresholds = np.concatenate([thresholds, [thresholds[-1] - 1]])

    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def calculate_auc(
    fpr: NDArray[np.float64],
    tpr: NDArray[np.float64],
) -> float:
    """Compute Area Under the ROC Curve using the trapezoidal rule.

    AUC ranges from 0 to 1:
      - 1.0 = perfect classifier (catches all attacks with zero false positives)
      - 0.5 = random guessing (no better than flipping a coin)
      - <0.5 = worse than random (scores are inverted)

    Args:
        fpr: False positive rates (x-axis of ROC curve).
        tpr: True positive rates (y-axis of ROC curve).

    Returns:
        The AUC score.
    """
    # Sort by FPR to ensure proper integration
    sorted_indices = np.argsort(fpr)
    return float(np.trapezoid(tpr[sorted_indices], fpr[sorted_indices]))


def calculate_roc_metrics(
    y_true: NDArray[np.int_],
    y_scores: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Compute ROC curve and AUC in one call.

    Args:
        y_true: Binary labels (1 = attack, 0 = honest).
        y_scores: Suspicion scores from the monitor.

    Returns:
        (fpr, tpr, thresholds, auc)
    """
    fpr, tpr, thresholds = calculate_roc_curve(y_true, y_scores)
    auc = calculate_auc(fpr, tpr)
    return fpr, tpr, thresholds, auc
