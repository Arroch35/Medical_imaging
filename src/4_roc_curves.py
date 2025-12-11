import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from config2 import *


SAVE_DIR = "../data/roc_curves"
os.makedirs(SAVE_DIR, exist_ok=True)
Config = "1"


def plot_10fold_roc(df, metric_name, labels_column="Presence"):

    SAVE_SUBDIR = os.path.join(SAVE_DIR, f"VAE_Config{Config}")
    os.makedirs(SAVE_SUBDIR, exist_ok=True)

    y = df[labels_column].values
    scores = df[metric_name].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 200)

    plt.figure(figsize=(7, 6))

    for train_idx, test_idx in kf.split(scores):
        y_true = y[test_idx]
        y_score = scores[test_idx]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, alpha=0.2, lw=1)

    # Mean ROC
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

    # Std band
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color="blue", alpha=0.2)

    plt.plot([0, 1], [0, 1], "k--", lw=1)

    plt.title(f"10-fold ROC Curve — {metric_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(SAVE_SUBDIR, f"ROC_{metric_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(f"reconstruction_metrics{Config}.csv")
    metric_names = [
        "emr_dissim",
        "mse_rgb",
        "max_p99",
        "mse_red",
        "mse_hsv_V",
        "mse_hsv_H"
    ]
    for metric in metric_names:
        plot_10fold_roc(df, metric_name=metric, labels_column="Presence")
        print(f"Plotted ROC curve for {metric}")
