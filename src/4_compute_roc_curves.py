import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import mean_squared_error
from sklearn.model_selection import KFold
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import rgb_to_hsv
from sklearn.metrics import roc_curve, auc
import pandas as pd

from config import *

METRICS = "../data/reconstruction_metrics.csv"
SAVE_DIR = "../data/roc_curves"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_10fold_roc(df, metric_name, labels_column="Presence"):
    """
    df: dataframe with metrics + a binary labels column
    metric_name: column name of the metric for ROC curve
    labels_column: column name containing the 0/1 ground truth

    Now returns best threshold using Youden's J statistic.
    """

    y = df[labels_column].values
    scores = df[metric_name].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 200)

    # store thresholds from each fold
    fold_thresholds = []

    plt.figure(figsize=(7, 6))

    for train_idx, test_idx in kf.split(scores):
        y_true = y[test_idx]
        y_score = scores[test_idx]

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # ---- Compute best threshold (Youden J) for THIS fold ----
        J = tpr - fpr
        best_idx = np.argmax(J)
        best_thresh = thresholds[best_idx]
        fold_thresholds.append(best_thresh)

        # interpolate for mean ROC curve
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, alpha=0.18, lw=1)

    # ---- Compute MEAN threshold across folds ----
    best_threshold_mean = float(np.mean(fold_thresholds))
    best_threshold_std = float(np.std(fold_thresholds))

    # ---- Mean ROC curve ----
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="blue",
        lw=2.5,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
    )

    # ---- Std band ----
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color="blue", alpha=0.2)

    # diagonal
    plt.plot([0, 1], [0, 1], "k--", lw=1)

    plt.title(f"10-fold ROC Curve — {metric_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save ROC Figure
    save_path = os.path.join(SAVE_DIR, f"ROC_{metric_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # ----- Print threshold result -----
    print(f"\n[Metric: {metric_name}]")
    print(f"  Optimal threshold (mean across folds): {best_threshold_mean:.6f} ± {best_threshold_std:.6f}")
    print(f"  Mean AUC: {mean_auc:.4f} (± {std_auc:.4f})")

    return best_threshold_mean, best_threshold_std


metrics = [
    "emr_dissim",
    "mse_rgb",
    "max_p99",
    "mse_red",
    "mse_hsv_V",
    "mse_hsv_H"
    ]

reconstruction_metrics=pd.read_csv(METRICS)
print(set(reconstruction_metrics["presence"]))

filtered_df = reconstruction_metrics[reconstruction_metrics['presence'] != 0]

results = []

for m in metrics:
    thr_mean, thr_std = plot_10fold_roc(filtered_df, m, labels_column="presence")
    results.append({
        "metric": m,
        "best_threshold_mean": thr_mean,
        "best_threshold_std": thr_std
    })

# ---- Save thresholds ----
thresholds_df = pd.DataFrame(results)
thresholds_df.to_csv("../data/best_thresholds.csv", index=False)

print(thresholds_df)