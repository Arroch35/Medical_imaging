import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import PATIENT_DIAGNOSIS_FILE

def create_patient_splits(
        patient_csv_path,
        output_dir="../data/patient_diagnosis",
        test_size=0.2,
        random_state=42
    ):
    """
    Reads patient diagnosis CSV and returns two stratified DataFrames:
    one for train and one for evaluation.
    
    The CSV must contain at least: PatID, Presence
    Presence ∈ {NEGATIVA, ALTA, BAIXA}.
    
    Output:
        train_df: dataframe of train patients
        eval_df: dataframe of eval patients
    
    Also saves:
        output_dir/train_patients.csv
        output_dir/eval_patients.csv
    """
    exclude_list = [
        'B22-108', 'B22-112', 'B22-142', 'B22-143', 'B22-149', 'B22-150',
        'B22-151', 'B22-152', 'B22-172', 'B22-270', 'B22-94', 'B22-93',
        'B22-61', 'B22-53', 'B22-34', 'B22-284', 'B22-165', 'B22-187',
        'B22-210', 'B22-90'
    ] 
    # --- Load patient diagnosis file ---
    df = pd.read_csv(patient_csv_path)
    df.drop(df[df['CODI'].isin(exclude_list)].index, inplace=True)

    # --- Check for necessary columns ---
    if not {"CODI", "DENSITAT"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: PatID, Presence")

    # --- Convert Presence to binary ---
    # NEGATIVA --> 0
    # ALTA or BAIXA --> 1
    df["DENSITAT"] = df["DENSITAT"].apply(
        lambda x: 0 if x.upper() == "NEGATIVA" else 1
    )

    # --- Stratified split ---
    train_df, eval_df = train_test_split(
        df[["CODI", "DENSITAT"]],
        test_size=test_size,
        stratify=df["DENSITAT"],
        random_state=random_state
    )

    # --- Ensure output directory exists ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Save splits ---
    train_path = os.path.join(output_dir, "train_patients.csv")
    eval_path = os.path.join(output_dir, "eval_patients.csv")

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    print(f"Saved: {train_path}")
    print(f"Saved: {eval_path}")

    return train_df, eval_df

create_patient_splits(
    patient_csv_path=PATIENT_DIAGNOSIS_FILE
)