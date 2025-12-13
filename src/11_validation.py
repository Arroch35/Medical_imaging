import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import joblib
import torch
import torchvision.transforms as T
import gc
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
PAT_SPLIT_CSV = "data/patient_diagnosis/eval_patients.csv"

SVM_METRICS_CSV = "data/svm/svm_kernel_metrics.csv"
PATIENT_CV_THRESHOLD_CSV = "data/patient_diagnosis/patient_cv_summary_opt_th.csv"
EMBEDDER_WEIGHTS = "checkpoints/CL/embedder_triplet_best.pt"
AE_WEIGHTS = "checkpoints/manual_removed/AE_Config1.pth"
PCA_PATH = "checkpoints/CL/pca_reducer.joblib"

SVM_MODEL = "data/svm/best_svm.pkl"
SCALER_PATH = "data/svm/scaler.pkl"

OUT_DIR = "data/patient_diagnosis/test_results"
IMG_BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# LOAD AUTOENCODER
# --------------------------------------------------
from utils import AEConfigs
from Models.AEmodels import AutoEncoderCNN, AEWithLatent
from Models.TripletModels import EmbedderLarge
from config import CROPPED_PATCHES_DIR

def load_autoencoder(path):
    inputmodule_paramsEnc = {"num_input_channels": 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs("1", inputmodule_paramsEnc)

    ae = AutoEncoderCNN(
        inputmodule_paramsEnc,
        net_paramsEnc,
        inputmodule_paramsDec,
        net_paramsDec,
    )

    ae.load_state_dict(torch.load(path, map_location=DEVICE))
    model = AEWithLatent(ae).to(DEVICE)
    model.eval()
    return model

print("Loading AE...")
ae_with_latent = load_autoencoder(AE_WEIGHTS)

pca = joblib.load(PCA_PATH)

embedder = EmbedderLarge(input_dim=pca.n_components_).to(DEVICE)
embedder.load_state_dict(torch.load(EMBEDDER_WEIGHTS, map_location=DEVICE))
embedder.eval()

scaler = joblib.load(SCALER_PATH)
svm = joblib.load(SVM_MODEL)
print("Loaded PCA, scaler, SVM.")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def load_patch_threshold(metrics_csv):
    try:
        df = pd.read_csv(metrics_csv)
        th = float(df.loc[0, "CV Optimal Threshold"] if "Optimal Threshold" in df.columns else 0.5)
        print(f"Loaded patch threshold: {th:.4f}")
        return th
    except:
        print("Using default patch threshold 0.5")
        return 0.5

def load_patient_threshold(threshold_csv):
    df = pd.read_csv(threshold_csv)
    thr = float(df.loc[0, "threshold_mean"])
    print(f"Loaded patient-level threshold: {thr:.4f}")
    return thr

transform_t = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

def collect_patch_paths_for_patient(pid):
    paths = []
    for folder_name in os.listdir(CROPPED_PATCHES_DIR):
        if folder_name.startswith(f"{pid}_"):
            folder_path = os.path.join(CROPPED_PATCHES_DIR, folder_name)
            for f in os.listdir(folder_path):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    paths.append(os.path.join(folder_path, f))
    return paths


@torch.no_grad()
def compute_patch_probabilities(patch_paths, patch_thr):
    probs = []

    for i in range(0, len(patch_paths), IMG_BATCH):
        batch = patch_paths[i:i+IMG_BATCH]

        imgs = [transform_t(Image.open(p).convert("RGB")) for p in batch]
        batch_tensor = torch.stack(imgs).to(DEVICE)

        latent, _ = ae_with_latent(batch_tensor)
        latent_flat = latent.view(latent.size(0), -1).cpu().numpy()

        latent_pca = pca.transform(latent_flat)
        
                # Embedder
        batch_tensor_pca = torch.tensor(latent_pca, dtype=torch.float32).to(DEVICE)
        embedding = embedder(batch_tensor_pca)
        emb_np = embedding.cpu().numpy()
        
        latent_scaled = scaler.transform(emb_np)
        prob = svm.predict_proba(latent_scaled)[:, 1]

        probs.extend(prob.tolist())

        del batch_tensor, latent, latent_flat, latent_pca, latent_scaled, prob
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return np.array(probs)


# --------------------------------------------------
# MAIN TEST FUNCTION
# --------------------------------------------------
def main():
    df_patients = pd.read_csv(PAT_SPLIT_CSV)
    df_patients["CODI"] = df_patients["CODI"].astype(str)

    # Load thresholds
    PATCH_THRESHOLD = load_patch_threshold(SVM_METRICS_CSV)
    PATIENT_THRESHOLD = load_patient_threshold(PATIENT_CV_THRESHOLD_CSV)

    patient_ids = []
    true_labels = []
    predicted_labels = []
    positive_percentage = []
    patch_counts = []

    print("\nRunning FINAL TEST evaluation...\n")
    for _, row in tqdm(df_patients.iterrows(), total=len(df_patients)):
        pid = row["CODI"]
        label = int(row["DENSITAT"])

        patch_list = collect_patch_paths_for_patient(pid)
        if len(patch_list) == 0:
            continue

        probs = compute_patch_probabilities(patch_list, PATCH_THRESHOLD)
        if len(probs) == 0:
            continue

        pos_pct = np.mean(probs >= PATCH_THRESHOLD)
        pred = int(pos_pct >= PATIENT_THRESHOLD)

        patient_ids.append(pid)
        true_labels.append(label)
        predicted_labels.append(pred)
        positive_percentage.append(pos_pct)
        patch_counts.append(len(probs))

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels)
    rec = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, positive_percentage)
    cm = confusion_matrix(true_labels, predicted_labels)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"AUC:        {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    out_df = pd.DataFrame({
        "patient_id": patient_ids,
        "true_label": true_labels,
        "pred_label": predicted_labels,
        "positive_percentage": positive_percentage,
        "n_patches": patch_counts
    })

    out_df.to_csv(os.path.join(OUT_DIR, "patient_test_results_cl_opt_th.csv"), index=False)
    print(f"\nSaved test results ? {OUT_DIR}/patient_test_results_cl_opt_th.csv")


if __name__ == "__main__":
    main()
