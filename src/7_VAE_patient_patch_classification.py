import pandas as pd
import os

from config2 import *

def apply_thresholds_to_patient(
    patient_csv_path: str,
    thresholds_csv_path: str,
    mse_col: str = "mse_red",
    id_col: str = "imfilename",
    patid_col: str = "PatID",
):
    # Load thresholds (one row per fold)
    opt_df = pd.read_csv(thresholds_csv_path)
    thresholds = dict(zip(opt_df["fold"], opt_df["best_threshold"]))

    # patient diagnosis csv
    df = pd.read_csv(patient_csv_path)

    # new prediction dataframe
    pred_df = pd.DataFrame()
    pred_df[id_col] = df[id_col]

    for fold, thr in thresholds.items():
        col_name = f"fold{fold}"
        # 1 if above threshold, -1 otherwise
        pred_df[col_name] = (df[mse_col] > thr).astype(int)
        pred_df[col_name] = pred_df[col_name].replace({0: -1})

    fold_cols = [c for c in pred_df.columns if c.startswith("fold")]
    vote_sum = pred_df[fold_cols].sum(axis=1)  # sum of -1/1 over folds

    # majority voting
    pred_df["all 10fold pred"] = (vote_sum > 0).astype(int)
    pred_df["all 10fold pred"] = pred_df["all 10fold pred"].replace({0: -1})

    # save the csv
    patient_id = str(df[patid_col].iloc[0])
    csv_dir = os.path.join(CLASSFICATION_DIR, f"VAE_Config{Config}")
    os.makedirs(csv_dir, exist_ok=True)

    out_path = os.path.join(csv_dir, f"{patient_id}_predicted.csv")

    pred_df.to_csv(out_path, index=False)
    return out_path, patient_id


Config = '1'
# load csv thresholds
thresholds = f'thresholds/optimal_thresholds_mse_red_Config{Config}.csv'

patients_folder = 'C:/Users/janaz/Documents/uni/YEAR 4 - S1/vision and learning/Reconstructions/VAE_pat_metrics1/metrics'
all_csvs = [f for f in os.listdir(patients_folder) if f.endswith(".csv")]

for csv_file in all_csvs:

    csv_path = os.path.join(patients_folder, csv_file)

    # check to not process thresholds or already predicted files
    if "optimal_thresholds" in csv_file:
        continue
    if csv_file.endswith("_predicted_presence.csv"):
        continue

    print(f"Processing: {csv_file}")

    try:
        out_path, pat_id = apply_thresholds_to_patient(csv_path, thresholds, mse_col="mse_red")
        print(f"Saved predicted CSV: {pat_id}\n")

    except Exception as e:
        print(f" !! Error processing {csv_file}: {e}\n")

