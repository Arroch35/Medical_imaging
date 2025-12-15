import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from configVAE import *


PATIENT_COL = "CODI"
DENSITAT_COL = "DENSITAT"
PRED_COL = "all 10fold pred"
SAVE_DIR = "../data/roc_curves_patients"
os.makedirs(SAVE_DIR, exist_ok=True)
Config = '3'


def load_patient_labels(
    patient_diagnosis_dir,
    patient_id_col,
    densitat_col,
):
    """
    Load PatientDiagnosis.csv and convert 'densitat' text labels to numeric:
      negativa = -1
      alta/baixa =  1

    Returns a DF:
      [patient_id_col, densitat_col, 'densitat_int']
    """
    df = pd.read_csv(patient_diagnosis_dir)

    # to chang the text labels to numeric
    mapping = {
        "NEGATIVA": -1,
        "ALTA": 1,
        "BAIXA": 1,
    }

    df["densitat_int"] = df[densitat_col].map(mapping)

    # null check
    if df["densitat_int"].isna().any():
        print("Warning: some 'densitat' values could not be mapped to -1/1:")
        print(df[df["densitat_int"].isna()][[patient_id_col, densitat_col]])

    return df[[patient_id_col, densitat_col, "densitat_int"]]


def compute_patient_percent_positive(
    patient_predictions_dir,
    pred_col,
):
    """
    For each patient:
      - read column pred_col (values -1 or 1)
      - compute percentage of positive (==1)

    Returns DF:
      ['patient_id', 'percent_positive']
    """
    rows = []

    pattern = os.path.join(patient_predictions_dir, "*_predicted.csv")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)

        patient_id = filename.replace("_predicted.csv", "")

        df_pred = pd.read_csv(filepath)
        if pred_col not in df_pred.columns:
            raise ValueError(f"Column '{pred_col}' not found in {filename}")

        preds = df_pred[pred_col].values

        # calculate percentage of positives
        percent_pos = np.mean(preds == 1)

        rows.append({
            "CODI": patient_id,
            "percent_positive": percent_pos,
        })

    patient_scores = pd.DataFrame(rows)
    return patient_scores


def build_patient_level_df(
    preds_dir,
    diagnosis_dir,
    patient_id_col=PATIENT_COL,
    densitat_col=DENSITAT_COL,
):
    """
    Returns DF merged:
      [patient_id_col, 'percent_positive', densitat_col, 'densitat_int']
    """
    labels_df = load_patient_labels(
        patient_diagnosis_dir=diagnosis_dir,
        patient_id_col=patient_id_col,
        densitat_col=densitat_col,
    )

    scores_df = compute_patient_percent_positive(patient_predictions_dir=preds_dir, pred_col=PRED_COL)

    scores_df_renamed = scores_df.rename(columns={"CODI": patient_id_col})

    merged = pd.merge(
        scores_df_renamed,
        labels_df,
        on=patient_id_col,
        how="inner",
    )

    Path('patient_diag_percentage').mkdir(exist_ok=True)

    merged.to_csv("patient_diag_percentage/patient_level_scores_and_labels.csv", index=False)

    return merged


def plot_10fold_roc_patients(
    df,
    metric_name="percent_positive",
    labels_column="densitat_int",
    save_dir=SAVE_DIR,
    curve_name="Patient_Level",
):
    """
    10-fold ROC on patient-level scores.

    df: DataFrame with patient-level [metric_name, labels_column]
    metric_name: e.g. 'percent_positive'
    labels_column: e.g. 'densitat_int' (-1/1)
    """
    os.makedirs(save_dir, exist_ok=True)

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
        lw=2.5,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
    )

    # Std band
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2)

    plt.plot([0, 1], [0, 1], "k--", lw=1)

    plt.title(f"10-fold ROC Curve — {curve_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"ROC_{curve_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"ROC curve saved to: {save_path}")


def get_optimal_thresholds_patients(
    df,
    metric_name="percent_positive",
    labels_column="densitat_int",
    filename=f"optimal_thresholds_patients_VAE_Config{Config}.csv",
):
    """
    10 folds, loop:
    build ROC on that fold
    compute Youden's J = TPR - FPR
    pick the threshold that maximizes J

    Saves CSV:
      [fold, best_threshold, best_tpr, best_fpr, youden_J]
    """
    y = df[labels_column].values
    scores = df[metric_name].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    results = []

    for fold_idx, (_, test_idx) in enumerate(kf.split(scores), start=1):
        y_true = y[test_idx]
        y_score = scores[test_idx]

        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        J = tpr - fpr
        best_idx = np.argmax(J)

        results.append({
            "fold": fold_idx,
            "best_threshold": thresholds[best_idx],
            "best_tpr": tpr[best_idx],
            "best_fpr": fpr[best_idx],
            "youden_J": J[best_idx],
        })

    df_thresh = pd.DataFrame(results)
    Path('thresholds').mkdir(exist_ok=True)

    df_thresh.to_csv(f"thresholds/{filename}", index=False)

    print(f"Optimal thresholds saved")
    return df_thresh


if __name__ == "__main__":
    pred_dir = CLASSFICATION_DIR / f"VAE_Config{Config}"
    df_roc = build_patient_level_df(pred_dir,PATIENT_DIAGNOSIS_FILE, PATIENT_COL, DENSITAT_COL)
    print(df_roc.head(10))
    plot_10fold_roc_patients(
        df_roc,
        metric_name="percent_positive",
        labels_column="densitat_int",
        save_dir=SAVE_DIR,
        curve_name=f"Patient_Level_VAE_Config{Config}",
    )

    get_optimal_thresholds_patients(
        df_roc,
        metric_name="percent_positive",
        labels_column="densitat_int",
    )

