# patient_classification_cv_memory_safe.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from joblib import load
import joblib
import torch
import torchvision.transforms as T
import gc

# ---------------------------
# Config (adjust if needed)
# ---------------------------
PAT_SPLIT_CSV = "../data/patient_diagnosis/train_patients.csv"
SVM_METRICS_CSV = "../data/svm/svm_kernel_metrics.csv"
AE_WEIGHTS = "checkpoints/manual_removed/AE_Config1.pth"
EMBEDDER_WEIGHTS = "checkpoints/CL/embedder_triplet_best.pt"
PCA_PATH = "checkpoints/CL/pca_reducer.joblib"
SVM_MODEL = "../data/svm/best_svm.pkl"
SCALER_PATH = "../data/svm/scaler.pkl"

OUT_DIR = "../data/patient_diagnosis"
N_FOLDS = 10
IMG_BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Load models
# ---------------------------
from utils import AEConfigs
from Models.AEmodels import AutoEncoderCNN, AEWithLatent
from Models.TripletModels import EmbedderLarge
from config import CROPPED_PATCHES_DIR

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

print("Loading AE & embedder...")
ae_with_latent, _ = load_autoencoder(AE_WEIGHTS)

pca = joblib.load(PCA_PATH)
print(f"Loaded PCA from {PCA_PATH}")

embedder = EmbedderLarge(input_dim=pca.n_components_).to(DEVICE)
embedder.load_state_dict(torch.load(EMBEDDER_WEIGHTS, map_location=DEVICE))
embedder.eval()
print("Loaded embedder weights.")

# Load SVM + scaler
scaler = joblib.load(SCALER_PATH)
svm = joblib.load(SVM_MODEL)
print("Loaded SVM and scaler.")

# ---------------------------
# Helpers
# ---------------------------
transform_t = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

def collect_patch_paths_for_patient(pid, base_dir):
    """Return list of patch paths for one patient"""
    paths = []
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith(f"{pid}_"):
            folder_path = os.path.join(base_dir, folder_name)
            for f in os.listdir(folder_path):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    paths.append(os.path.join(folder_path, f))
    return paths

@torch.no_grad()
def get_patch_probs_for_patient(patch_paths):
    """Compute SVM probabilities for all patches of one patient in batches"""
    probs = []
    for i in range(0, len(patch_paths), IMG_BATCH):
        batch_paths = patch_paths[i:i+IMG_BATCH]
        imgs = [transform_t(Image.open(p).convert("RGB")) for p in batch_paths]
        batch_tensor = torch.stack(imgs).to(DEVICE)

        # AE latent
        latent, _ = ae_with_latent(batch_tensor)
        latent_flat = latent.view(latent.size(0), -1).cpu().numpy()

        # PCA
        latent_pca = pca.transform(latent_flat)

        # Embedder
        batch_tensor_pca = torch.tensor(latent_pca, dtype=torch.float32).to(DEVICE)
        embedding = embedder(batch_tensor_pca)
        emb_np = embedding.cpu().numpy()

        # Scale + SVM
        emb_scaled = scaler.transform(emb_np)
        prob_batch = svm.predict_proba(emb_scaled)[:, 1]
        probs.extend(prob_batch.tolist())

        # Free memory per batch
        del batch_tensor, latent, latent_flat, latent_pca, batch_tensor_pca, embedding, emb_np, emb_scaled
        gc.collect()
        torch.cuda.empty_cache()
    return probs

def load_patch_threshold(metrics_csv):
    try:
        df = pd.read_csv(metrics_csv)
        th = float(df.loc[0, "CV Optimal Threshold"] if "Optimal Threshold" in df.columns else 0.5)
        print(f"Loaded patch threshold: {th:.4f}")
        return th
    except:
        print("Using default patch threshold 0.5")
        return 0.5

# ---------------------------
# Main
# ---------------------------
def main():
    patients_df = pd.read_csv(PAT_SPLIT_CSV)
    patients_df["CODI"] = patients_df["CODI"].astype(str)

    PATCH_THRESHOLD = load_patch_threshold(SVM_METRICS_CSV)

    patient_ids = []
    patient_labels = []
    patient_pos_pct = []
    patient_num_patches = []

    print("\nProcessing patients individually...\n")
    for _, row in tqdm(patients_df.iterrows(), total=len(patients_df)):
        pid = row["CODI"]
        presence = int(row["DENSITAT"])

        patch_paths = collect_patch_paths_for_patient(pid, CROPPED_PATCHES_DIR)
        if len(patch_paths) == 0:
            continue

        probs = get_patch_probs_for_patient(patch_paths)
        if len(probs) == 0:
            continue

        probs_np = np.array(probs)
        positive_count = np.sum(probs_np >= PATCH_THRESHOLD)
        total_count = len(probs_np)
        pos_pct = positive_count / total_count

        patient_ids.append(pid)
        patient_labels.append(presence)
        patient_pos_pct.append(pos_pct)
        patient_num_patches.append(total_count)

        # Clean memory per patient
        del patch_paths, probs, probs_np
        gc.collect()
        torch.cuda.empty_cache()

    # Save per-patient stats
    patient_df = pd.DataFrame({
        "patient_id": patient_ids,
        "presence": patient_labels,
        "positive_percentage": patient_pos_pct,
        "n_patches": patient_num_patches
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    patient_df.to_csv(os.path.join(OUT_DIR, "patient_positive_percentages_raw_opt_th.csv"), index=False)
    print(f"Saved raw patient stats to {OUT_DIR}/patient_positive_percentages_raw_opt_th.csv")

    # ---------------------------
    # 10-fold CV on patients
    # ---------------------------
    X = np.array(patient_pos_pct).reshape(-1, 1)
    y = np.array(patient_labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    thresholds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X[train_idx].ravel()
        y_train = y[train_idx]
        X_val = X[val_idx].ravel()
        y_val = y[val_idx]

        fpr, tpr, thr = roc_curve(y_train, X_train)
        J = tpr - fpr
        best_i = np.argmax(J)
        best_thr = thr[best_i]
        thresholds.append(best_thr)

        val_preds = (X_val >= best_thr).astype(int)
        val_auc = roc_auc_score(y_val, X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_prec = precision_score(y_val, val_preds)
        val_rec = recall_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds)
        val_cm = confusion_matrix(y_val, val_preds).tolist()

        fold_results.append({
            "fold": fold_idx,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "threshold": float(best_thr),
            "threshold_idx": int(best_i),
            "val_auc": float(val_auc),
            "val_acc": float(val_acc),
            "val_prec": float(val_prec),
            "val_rec": float(val_rec),
            "val_f1": float(val_f1),
            "val_confmat": val_cm
        })

        print(f"Fold {fold_idx}: thr={best_thr:.4f} | AUC={val_auc:.4f} | Acc={val_acc:.4f}")

    thr_mean = float(np.mean(thresholds))
    thr_std  = float(np.std(thresholds, ddof=1)) if len(thresholds) > 1 else 0.0

    results_df = pd.DataFrame(fold_results)
    metrics_summary = {
        "threshold_mean": thr_mean,
        "threshold_std": thr_std,
        "n_folds": N_FOLDS,
        "fold_val_auc_mean": float(results_df["val_auc"].mean()),
        "fold_val_auc_std": float(results_df["val_auc"].std(ddof=1))
    }

    # Save results
    results_df.to_csv(os.path.join(OUT_DIR, "patient_cv_fold_results_opt_th.csv"), index=False)
    pd.DataFrame([metrics_summary]).to_csv(os.path.join(OUT_DIR, "patient_cv_summary_opt_th.csv"), index=False)

    print(f"Threshold mean = {thr_mean:.4f}, std = {thr_std:.4f}")
    print("Patient CV completed.")

if __name__ == "__main__":
    main()
