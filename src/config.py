from pathlib import Path

YOUR_PATH_TO_THE_PROJECT = r"C:\Users\arroc\OneDrive\Escritorio\Apuntes\UAB\4th_year\VAL\github_projects"

ROOT_DIR = Path(YOUR_PATH_TO_THE_PROJECT) / "Medical_imaging"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR = ROOT_DIR / "src"

CROSS_VALIDATION_DIR = DATA_DIR / "CrossValidation"
CROPPED_PATCHES_DIR = CROSS_VALIDATION_DIR / "Cropped"
ANNOTATED_PATCHES_DIR = CROSS_VALIDATION_DIR / "Annotated"
ANNOTATED_METADATA_FILE = CROSS_VALIDATION_DIR / "HP_WSI-CoordAllAnnotatedPatches.xlsx"
PATIENT_DIAGNOSIS_FILE = CROSS_VALIDATION_DIR / "PatientDiagnosis.csv"