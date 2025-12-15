import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)
from pathlib import Path


PATIENT_ID_COL_DIAG = "CODI"     # same as before
SCORE_COL = "percent_positive"        # from previous step
LABEL_COL = "densitat_int"            # numeric ground truth (-1 / 1)
Config = '3'

SAVE_DIR = "../data/confusion_matrices"
os.makedirs(SAVE_DIR, exist_ok=True)

# df all patient predictions for all thresholds
def load_patient_level_df(path=None):
    df = pd.read_csv("patient_diag_percentage/patient_level_scores_and_labels.csv")
    return df


# df optimal thresholds per fold
def load_thresholds_df(path=None):
    df = pd.read_csv(f"thresholds/optimal_thresholds_patients_VAE_Config{Config}.csv")
    return df


# classify patients for a threshold REVISAR
def classify_patients_for_threshold(df_patient, threshold,
                                    score_col=SCORE_COL):
    preds = np.where(df_patient[score_col].values >= threshold, 1, -1)
    return preds


def plot_and_save_confusion_matrix(fold, cm, labels, title, cmap="Blues"):
    """
    Plots a labeled confusion matrix and saves it.
    """
    SAVE_SUBDIR = os.path.join(SAVE_DIR, f"VAE_Config{Config}")
    os.makedirs(SAVE_SUBDIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True Label',
        xlabel='Predicted Label',
        title=title
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Annotate numbers
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    save_path = os.path.join(SAVE_SUBDIR, f"confusion_matrix_fold{fold}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Confusion matrix → {save_path}")


def evaluate_thresholds_on_all_patients(
    df_patient,
    thresholds_df,
    id_col=PATIENT_ID_COL_DIAG,
    score_col=SCORE_COL,
    label_col=LABEL_COL,
):
    preds_df = df_patient[[id_col, score_col, label_col]].copy()

    y_true_raw = df_patient[label_col].values
    y_true_bin = (y_true_raw == 1).astype(int)   # -1 -> 0, 1 -> 1
    y_score = df_patient[score_col].values

    try:
        auc_overall = roc_auc_score(y_true_bin, y_score)
    except ValueError:
        auc_overall = np.nan

    metrics_rows = []

    for _, row in thresholds_df.iterrows():
        fold = int(row["fold"])
        thr = float(row["best_threshold"])

        col_name = f"pred_fold{fold}"

        # Predictions
        y_pred = classify_patients_for_threshold(df_patient, thr, score_col=score_col)
        preds_df[col_name] = y_pred

        # Confusion matrix with labels [-1, 1]
        cm = confusion_matrix(y_true_raw, y_pred, labels=[-1, 1])

        plot_and_save_confusion_matrix(
            fold=fold,
            cm=cm,
            labels=["Negatiu (-1)", "Positiu (1)"],
            title=f"Confusion Matrix — Fold {fold}",
        )

        #          pred=-1  pred=1
        # true=-1   TN       FP
        # true=1    FN       TP
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp

        accuracy = (tn + tp) / total if total > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = (2 * precision * recall / (precision + recall)) if (
            not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
        ) else np.nan

        sensitivity = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        balanced_acc = (sensitivity + specificity) / 2 if (
            not np.isnan(sensitivity) and not np.isnan(specificity)
        ) else np.nan

        metrics_rows.append({
            "fold": fold,
            "threshold": thr,

            "TN": tn, "FP": fp, "FN": fn, "TP": tp,

            "auc": auc_overall,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,

            "specificity": specificity,
            "balanced_accuracy": balanced_acc,
        })

    Path("patient_final_diagnosis").mkdir(exist_ok=True)
    preds_path = f"patient_final_diagnosis/patient_preds_on_thresholds_Config{Config}.csv"
    metrics_path = f"patient_final_diagnosis/thresholds_patient_level_performance_Config{Config}.csv"

    preds_df.to_csv(preds_path, index=False)
    print(f"Per-patient predictions for all thresholds saved to: {preds_path}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics per fold/threshold saved to: {metrics_path}")

    # --- Overall (across folds): mean of metrics across fold-rows ---
    overall = {
        "Config": Config,
        "mean_auc": metrics_df["auc"].mean(skipna=True),
        "mean_accuracy": metrics_df["accuracy"].mean(skipna=True),
        "mean_precision": metrics_df["precision"].mean(skipna=True),
        "mean_recall": metrics_df["recall"].mean(skipna=True),
        "mean_f1": metrics_df["f1"].mean(skipna=True),
    }
    overall_df = pd.DataFrame([overall])
    overall_path = f"patient_final_diagnosis/overall_patient_level_performance_Config{Config}.csv"
    overall_df.to_csv(overall_path, index=False)
    print(f"Overall mean metrics saved to: {overall_path}")

    return preds_df, metrics_df, overall_df



def select_best_fold_threshold(metrics_df, weights=None):
    """
    Selects the best row (fold/threshold) using a weighted combination of metrics.
    Default: equal weights for accuracy, precision, recall, f1.

    Returns: (best_row_series, metrics_with_score_df)
    """
    if weights is None:
        weights = {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    df = metrics_df.copy()

    # Ensure required columns exist
    required = list(weights.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"metrics_df missing required columns: {missing}")

    # Composite score (ignore NaNs by treating them as very low)
    score = 0.0
    wsum = 0.0
    for col, w in weights.items():
        score += w * df[col].fillna(-1e9)
        wsum += w
    df["composite_score"] = score / wsum

    best_idx = df["composite_score"].idxmax()
    best_row = df.loc[best_idx]
    return best_row, df


def print_best_selection(best_row):
    fold = int(best_row["fold"])
    thr = float(best_row["threshold"])
    print("\n=== BEST FOLD / THRESHOLD (multi-metric) ===")
    print(f"Fold: {fold}")
    print(f"Threshold: {thr:.6f}")
    print(f"AUC: {best_row['auc']:.6f}" if pd.notna(best_row["auc"]) else "AUC: NaN")
    print(f"Accuracy: {best_row['accuracy']:.6f}")
    print(f"Precision: {best_row['precision']:.6f}")
    print(f"Recall: {best_row['recall']:.6f}")
    print(f"F1: {best_row['f1']:.6f}")
    print(f"Composite score: {best_row['composite_score']:.6f}")

if __name__ == "__main__":
    # patients df
    df_patient = load_patient_level_df()

    # thresholds
    thresholds_df = load_thresholds_df()

    # apply thresholds and evaluate
    preds_df, metrics_df, summary_df = evaluate_thresholds_on_all_patients(
        df_patient,
        thresholds_df,
    )

    # best fold/threshold selection
    best_row, metrics_scored_df = select_best_fold_threshold(metrics_df)
    print_best_selection(best_row)
