import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

PAT_SPLIT_CSV = "../data/patient_diagnosis/train_patients.csv"
SVM_METRICS_CSV = "../data/svm/svm_kernel_metrics.csv"
AE_WEIGHTS = "checkpoints/AE_Config1.pth"
EMBEDDER_WEIGHTS = "checkpoints/CL/triplet_embedder.pt"
SVM_MODEL = "../data/svm/best_svm_auc.pkl"
SCALER_PATH = "../data/svm/scaler.pkl"

OUTPUT_DF_PATH = "../data/patient_diagnosis/train_patient_patch_stats.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------

from Models.AEmodels import AutoEncoderCNN, AEWithLatent
from utils import AEConfigs
from config import CROPPED_PATCHES_DIR
from Models.TripletModels import Embedder
from joblib import load

def load_autoencoder(autoencoder_path):
    inputmodule_paramsEnc = {"num_input_channels": 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs("1", inputmodule_paramsEnc)

    ae = AutoEncoderCNN(
        inputmodule_paramsEnc,
        net_paramsEnc,
        inputmodule_paramsDec,
        net_paramsDec,
    )
    ae.load_state_dict(torch.load(autoencoder_path, map_location=DEVICE))
    model = AEWithLatent(ae).to(DEVICE)
    model.eval()

    return model, ae

# Load AE
ae_with_latent, _ = load_autoencoder(AE_WEIGHTS)

# Load embedder
embedder = Embedder(input_dim=64*64*64).to(DEVICE)
embedder.load_state_dict(torch.load(EMBEDDER_WEIGHTS, map_location=DEVICE))
embedder.eval()

# Load SVM + scaler
svm = load(SVM_MODEL)
scaler = load(SCALER_PATH)

# Load OPTIMAL PATCH THRESHOLD
svm_metrics = pd.read_csv(SVM_METRICS_CSV)
PATCH_THRESHOLD = svm_metrics.loc[0, "Optimal Threshold"]  # first row
print(f"Patch threshold loaded: {PATCH_THRESHOLD:.4f}")


# ---------------------------------------------------------------
# IMAGE → LATENT → EMBEDDING → SVM PROBABILITY
# ---------------------------------------------------------------

import torchvision.transforms as T

transform_t = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

@torch.no_grad()
def get_patch_probability(img_path):
    """Returns SVM probability for a single patch."""
    img = Image.open(img_path).convert("RGB")
    img = transform_t(img).unsqueeze(0).to(DEVICE)

    # AE → latent 
    latent, _ = ae_with_latent(img)   
    latent_flat = latent.flatten().unsqueeze(0)
    # Embedding
    embedding = embedder(latent_flat)             # (1, embed_dim)
    embedding_np = embedding.cpu().numpy()

    # Scale
    embedding_scaled = scaler.transform(embedding_np)

    # Probability from SVM
    prob = svm.predict_proba(embedding_scaled)[0, 1]
    return prob


# ---------------------------------------------------------------
# PROCESS ALL TRAIN PATIENTS
# ---------------------------------------------------------------

train_df = pd.read_csv(PAT_SPLIT_CSV)

patient_rows = []
print("\nProcessing patients...\n")

for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    patient_id = str(row["CODI"])
    presence = row["DENSITAT"]

    # Find the actual folder name by searching for a directory that starts with the patient_id
    patient_dir_name = None
    # Check all items in the parent directory (CROPPED_PATCHES_DIR)
    for folder_name in os.listdir(CROPPED_PATCHES_DIR):
        # Check if the folder name starts with the patient_id followed by '_'
        if folder_name.startswith(f"{patient_id}_"):
            patient_dir_name = folder_name
            break # Found the folder, stop searching

    if patient_dir_name is None:
        print(f"WARNING: No images for patient {patient_id} (missing folder starting with {patient_id}_), skipping.")
        continue
    
    # Construct the full path using the actual folder name
    patient_path = os.path.join(CROPPED_PATCHES_DIR, patient_dir_name)

    image_paths = [
        os.path.join(patient_path, f)
        for f in os.listdir(patient_path)
        if f.lower().endswith((".png"))
    ]

    if len(image_paths) == 0:
        print(f"WARNING: Patient {patient_id} has no images.")
        continue

    # Process images
    positive_count = 0
    total_count = 0

    for img_path in image_paths:
        prob = get_patch_probability(img_path)
        pred = 1 if prob >= PATCH_THRESHOLD else 0

        if pred == 1:
            positive_count += 1
        total_count += 1

    positive_percentage = positive_count / total_count
    negative_percentage = 1 - positive_percentage

    patient_rows.append({
        "patient_id": patient_id,
        "negative_percentage": negative_percentage,
        "positive_percentage": positive_percentage,
        "presence": presence
    })

    # Clear memory for next patient
    torch.cuda.empty_cache()

# Save patient statistics
patient_stats_df = pd.DataFrame(patient_rows)
os.makedirs(os.path.dirname(OUTPUT_DF_PATH), exist_ok=True)
patient_stats_df.to_csv(OUTPUT_DF_PATH, index=False)

print(f"\nSaved patient statistics to: {OUTPUT_DF_PATH}")
print(patient_stats_df.head())


# ---------------------------------------------------------------
# DETERMINE PATIENT-LEVEL THRESHOLD
# ---------------------------------------------------------------

print("\nComputing optimal patient-level threshold...")

y_true = patient_stats_df["presence"].values
scores = patient_stats_df["positive_percentage"].values

# sweep thresholds from 0 → 1
ths = np.linspace(0, 1, 500)
best_th = 0
best_auc = 0

for th in ths:
    preds = (scores >= th).astype(int)
    auc_value = roc_auc_score(y_true, preds)
    if auc_value > best_auc:
        best_auc = auc_value
        best_th = th

print(f"\nOptimal patient-level threshold = {best_th:.4f}")
print(f"AUC at that threshold = {best_auc:.4f}")


# BIIIEEN, FUNCIONA!!! Faltaria hacerlo con 10-fold
# Falta hacer esto pero con evaluation y 10-fold para determinar la accuracy y esas cosas
# Todo el pipeline lo tnedria que hacercon el mejor modelo