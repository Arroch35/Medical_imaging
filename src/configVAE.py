from pathlib import Path

DATA_DIR = Path("C:/Users/janaz/Documents/uni/YEAR 4 - S1/vision and learning")

PROJECT_DIR = DATA_DIR / "Medical_imaging"

CROSS_VAL  = DATA_DIR / "CrossValidation"
CROPPED_PATCHES_DIR = CROSS_VAL / "Cropped"
ANNOTATED_PATCHES_DIR = CROSS_VAL / "Annotated"
ANNOTATED_METADATA_FILE = CROSS_VAL / "HP_WSI-CoordAllAnnotatedPatches.xlsx"
PATIENT_DIAGNOSIS_FILE = CROSS_VAL / "PatientDiagnosis.csv"
RECON_DIR = DATA_DIR / "Reconstructions"
CLASSFICATION_DIR = DATA_DIR / "Patient_Classfication"
ROC_SAVE_DIR = PROJECT_DIR / "data"