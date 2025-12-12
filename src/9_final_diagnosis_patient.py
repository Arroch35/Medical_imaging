import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path


PATIENT_ID_COL_DIAG = "CODI"     # same as before
SCORE_COL = "percent_positive"        # from previous step
LABEL_COL = "densitat_int"            # numeric ground truth (-1 / 1)
Config = '1'

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


# classify patients for a threshold
def classify_patients_for_threshold(df_patient, threshold,
                                    score_col=SCORE_COL):
    preds = np.where(df_patient[score_col].values >= threshold, 1, -1)
    return preds


def plot_and_save_confusion_matrix(cm, labels, title, save_path, cmap="Blues"):
    """
    Plots a labeled confusion matrix and saves it.
    """
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
    """
    For each threshold (row) in thresholds_df:
      - classify all patients using percent_positive and that threshold
      - compare predictions with ground truth labels (densitat_num)
      - compute TN, FP, FN, TP, accuracy, sensitivity, specificity

    Returns:
      - patient_level_predictions_all_thresholds.csv
      - thresholds_patient_level_performance.csv
    """
    # new df to store all predictions
    preds_df = df_patient[[id_col, score_col, label_col]].copy()

    metrics_rows = []

    for _, row in thresholds_df.iterrows():
        fold = row["fold"].astype(int)
        thr = row["best_threshold"]

        col_name = f"pred_fold{fold}"

        # 1
        y_pred = classify_patients_for_threshold(
            df_patient, thr, score_col=score_col
        )
        preds_df[col_name] = y_pred

        # 2
        y_true = df_patient[label_col].values

        cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])

        cm_save_path = os.path.join(SAVE_DIR, f"confusion_matrix_fold{fold}.png")
        plot_and_save_confusion_matrix(
            cm,
            labels=["Negatiu (-1)", "Positiu (1)"],
            title=f"Confusion Matrix — Fold {fold}",
            save_path=cm_save_path
        )

        #          pred=-1  pred=1
        # true=-1   TN       FP
        # true=1    FN       TP
        tn, fp, fn, tp = cm.ravel()

        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total if total > 0 else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # recall for class 1
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan  # recall for class -1
        balanced_acc = (sensitivity + specificity) / 2 \
            if not np.isnan(sensitivity) and not np.isnan(specificity) else np.nan

        metrics_rows.append({
            "fold": fold,
            "threshold": thr,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "accuracy": accuracy,
            "sensitivity_pos_class1": sensitivity,
            "specificity_neg_class-1": specificity,
            "balanced_accuracy": balanced_acc,
        })

    # Save predictions for all thresholds
    Path('patient_final_diagnosis').mkdir(exist_ok=True)
    preds_df.to_csv(f"patient_final_diagnosis/patient_preds_on_thresholds{Config}.csv", index=False)
    print(f"Per-patient predictions for all thresholds saved to: /patient_final_diagnosis")

    # Save metrics per threshold
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(f"patient_final_diagnosis/thresholds_patient_level_performance{Config}.csv", index=False)
    print(f"Metrics per threshold saved to: /patient_final_diagnosis")

    return preds_df, metrics_df


if __name__ == "__main__":
    # patients df
    df_patient = load_patient_level_df()

    # thresholds
    thresholds_df = load_thresholds_df()

    # apply thresholds and evaluate
    preds_df, metrics_df = evaluate_thresholds_on_all_patients(
        df_patient,
        thresholds_df,
    )
