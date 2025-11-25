import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import mean_squared_error
from sklearn.model_selection import KFold
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import rgb_to_hsv
from sklearn.metrics import roc_curve, auc
import pandas as pd

from config import *

TOP_N = 20
SAVE_DIR = "../data/roc_curves"
os.makedirs(SAVE_DIR, exist_ok=True)

# loads an image and converts to RGB numpy array
def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


# loads all original-reconstruction pairs
# def load_pairs():
#     """
#     Returns:
#         originals, recons : list of HxWx3 float32
#         filenames         : list of "PatID/filename.png"
#         labels            : np.array of 0/1 (Presence)
#     """
#     originals, recons, filenames = [], [], []

#     # Walk recursively across all patient subfolders
#     for root, _, files in os.walk(ANNOTATED_PATCHES_DIR):

#         for fname in files:
#             if not fname.lower().endswith(".png"):
#                 continue

#             orig_path = os.path.join(root, fname)

#             # build relative path:
#             rel_path = os.path.relpath(os.path.join(root[:-2], fname), ANNOTATED_PATCHES_DIR)

#             # reconstruction must be in same relative location
#             recon_path = os.path.join(RECON_DIR, "Config1", rel_path)

#             if not os.path.exists(recon_path):
#                 print(f"[Warning] Missing reconstruction: {rel_path}")
#                 continue

#             originals.append(load_image_rgb(orig_path))
#             recons.append(load_image_rgb(recon_path))
#             filenames.append(rel_path)  # store patient/id/imgname

#     return originals, recons, filenames

def load_pairs():
    """
    Returns:
        originals, recons : list of HxWx3 float32 arrays
        filenames         : list of "PatID/filename.png"
        labels            : np.array of 0/1 (Presence)
    """

    # detect delimiter automatically
    df_labels = pd.read_excel(ANNOTATED_METADATA_FILE)

    # needed columns
    # required columns
    required_cols = {"Pat_ID", "Window_ID", "Presence"}
    if not required_cols.issubset(df_labels.columns):
        raise ValueError(f"Label file missing required columns: {required_cols}")

    # remove rows with missing labels
    missing = df_labels["Presence"].isna().sum()
    if missing > 0:
        print(f"[Warning] {missing} rows have missing Presence → skipping them.")
    df_labels = df_labels.dropna(subset=["Presence"])

    # convert Presence to integer safely
    df_labels["Presence"] = df_labels["Presence"].astype(int)

    # Build dictionary (Pat_ID, Window_ID) → Presence
    label_dict = {
        (str(row.Pat_ID), str(row.Window_ID)): row.Presence
        for _, row in df_labels.iterrows()
    }
    # ---------------------------------------------
    # Walk and load image + reconstruction pairs
    # ---------------------------------------------
    originals, recons, filenames, labels = [], [], [], []

    for root, _, files in os.walk(ANNOTATED_PATCHES_DIR):

        # Extract patient folder name
        pat_id = os.path.basename(root)

        for fname in files:
            if not fname.lower().endswith(".png"):
                continue

            orig_path = os.path.join(root, fname)

            # Derive Window_ID by stripping leading zeros
            window_id = os.path.splitext(fname)[0].lstrip("0")
            if window_id == "":
                window_id = "0"

            # Retrieve label from dictionary
            key = (pat_id[:-2], window_id)
            if key not in label_dict:
                print(f"[Warning] No label found for {key}, skipping.")
                continue

            label = label_dict[key]

            # build relative path e.g. PatID/filename.png
            rel_path = os.path.relpath(os.path.join(root[:-2], fname), ANNOTATED_PATCHES_DIR)

            # reconstruction path
            recon_path = os.path.join(RECON_DIR, "Config1", rel_path)

            if not os.path.exists(recon_path):
                print(f"[Warning] Missing reconstruction: {rel_path}")
                continue

            originals.append(load_image_rgb(orig_path))
            recons.append(load_image_rgb(recon_path))
            filenames.append(rel_path)
            labels.append(label)

    return originals, recons, filenames, np.array(labels, dtype=np.int32)

# Metric functions
def emr_dissimilarity(o, r):
    # o, r: (H,W,3) float32
    # SSIM ∈ [-1,1], so 1-SSIM is a dissimilarity: 0 = perfect
    score = ssim(o, r, channel_axis=-1, data_range=255.0)
    return 1.0 - score


def mse_rgb(o, r):
    return np.mean((o - r) ** 2)


def max_percentile(o, r, p=99):
    sq_err = (o - r) ** 2
    return np.percentile(sq_err, p)


def mse_red(o, r):
    # channel 0 = R
    return np.mean((o[..., 0] - r[..., 0]) ** 2)


def mse_hsv_value(o, r):
    o01 = o / 255.0
    r01 = r / 255.0
    hsv_o = rgb_to_hsv(o01)
    hsv_r = rgb_to_hsv(r01)
    return np.mean((hsv_o[..., 2] - hsv_r[..., 2]) ** 2)  # V channel

def mse_hsv_hue(o, r):
    o01 = o / 255.0
    r01 = r / 255.0
    hsv_o = rgb_to_hsv(o01)
    hsv_r = rgb_to_hsv(r01)
    return np.mean((hsv_o[..., 0] - hsv_r[..., 0]) ** 2)  # H channel


def compute_metrics(originals, recons, filenames):
    rows = []
    for o, r, name in zip(originals, recons, filenames):
        row = {
            "filename":        name,
            "emr_dissim":      emr_dissimilarity(o, r),
            "mse_rgb":         mse_rgb(o, r),
            "max_p99":         max_percentile(o, r, p=99),
            "mse_red":         mse_red(o, r),
            "mse_hsv_V":       mse_hsv_value(o, r),
            "mse_hsv_H":       mse_hsv_hue(o, r)
        }
        rows.append(row)
    return pd.DataFrame(rows)


def show_top(originals, recons, filenames, scores, title, top_n=20):
    idxs = np.argsort(scores)[-top_n:]
    for i in reversed(idxs):
        print(f"{filenames[i]} — {title} = {scores[i]:.6f}")

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(originals[i].astype(np.uint8))
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(recons[i].astype(np.uint8))
        axs[1].set_title(f"Reconstruction\n{title} = {scores[i]:.5f}")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()
    plt.close()


def plot_10fold_roc(df, metric_name, labels_column="Presence"):
    """
    df: dataframe with metrics + a binary labels column
    metric_name: column name of the metric for ROC curve
    labels_column: column name containing the 0/1 ground truth
    """

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

        # interpolate TPR values so all curves share identical FPR axis
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, alpha=0.2, lw=1, label=None)

    # ---- Mean ROC curve ----
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="blue",
        lw=2.5,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
    )

    # ---- Std deviation band ----
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color="blue", alpha=0.2)

    plt.plot([0, 1], [0, 1], "k--", lw=1)

    plt.title(f"10-fold ROC Curve — {metric_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"ROC_{metric_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()





if __name__ == "__main__":

    originals, recons, filenames, labels = load_pairs()
    print("Loaded pairs:", len(originals))
    df = compute_metrics(originals, recons, filenames)
    df["presence"] = labels

    # Save metrics per patch so you can analyse them later
    df.to_csv("../data/reconstruction_metrics.csv", index=False)

    # Example: visualize top errors for each metric
    show_top(originals, recons, filenames, df["mse_hsv_V"].values, "Value MSE HSV", TOP_N)
    show_top(originals, recons, filenames, df["mse_rgb"].values,    "MSE RGB", TOP_N)
    show_top(originals, recons, filenames, df["emr_dissim"].values, "EMR dissimilarity", TOP_N)
