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

from config2 import *

TOP_N = 20
Config = "1"


# loads an image and converts to RGB numpy array
def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


# loads all original-reconstruction pairs
def load_pairs():
    """
    Returns:
        originals, recons : list of HxWx3 float32 arrays
        filenames         : list of "PatID_suffix/filename.png"
        labels            : np.array of 0/1 (Presence)
    """

    # Load metadata
    df_labels = pd.read_excel(ANNOTATED_METADATA_FILE)
    df_labels = df_labels.dropna(subset=["Presence"])
    df_labels["Presence"] = df_labels["Presence"].astype(int)

    # Build label dictionary
    label_dict = {
        (str(row.Pat_ID), str(row.Window_ID)): row.Presence
        for _, row in df_labels.iterrows()
    }

    originals, recons, filenames, labels = [], [], [], []

    # -------------------------------
    # Load reconstructions depending on CONFIG
    # -------------------------------
    recon_root = os.path.join(RECON_DIR, f"VAE_Config{Config}")

    # Build mapping: Pat_ID → multiple annotated folders (001_0, 001_1…)
    annotated_subfolders = os.listdir(ANNOTATED_PATCHES_DIR)
    pat_to_annotated = {}
    for folder in annotated_subfolders:
        base = folder.split("_")[0]
        pat_to_annotated.setdefault(base, []).append(folder)

    # Walk reconstructions
    for root, _, files in os.walk(recon_root):

        pat_id = os.path.basename(root)

        for fname in files:
            if not fname.lower().endswith(".png"):
                continue

            recon_path = os.path.join(root, fname)

            # Derive window ID
            window_id = os.path.splitext(fname)[0].lstrip("0")
            if window_id == "":
                window_id = "0"

            key = (pat_id, window_id)
            if key not in label_dict:
                print(f"[Warning] Missing label for {key}")
                continue
            label = label_dict[key]

            # ------ Find matching annotated (original) patch ------
            if pat_id not in pat_to_annotated:
                print(f"[Warning] No annotated folder for patient {pat_id}")
                continue

            found_original = False
            for ann_folder in pat_to_annotated[pat_id]:
                orig_path = os.path.join(ANNOTATED_PATCHES_DIR, ann_folder, fname)
                if os.path.exists(orig_path):
                    found_original = True
                    break

            if not found_original:
                print(f"[Warning] No original file for {pat_id}/{fname}")
                continue

            originals.append(load_image_rgb(orig_path))
            recons.append(load_image_rgb(recon_path))
            filenames.append(f"{ann_folder}/{fname}")
            labels.append(label)

    print(f"Loaded {len(originals)} pairs.")

    return originals, recons, filenames, np.array(labels, dtype=np.int32)


# Metric functions
def emr_dissimilarity(o, r):
    # o, r: (H,W,3) float32
    # SSIM ∈ [-1,1], so 1-SSIM is a dissimilarity: 0 = perfect
    score = ssim(o, r, channel_axis=-1, data_range=1.0)
    return 1.0 - score


def mse_rgb(o, r):
    return np.mean((o - r) ** 2)


def max_percentile(o, r, p=99):
    sq_err = (o - r) ** 2
    return np.percentile(sq_err, p)


def mse_red(o, r):
    # channel 0 = R
    return np.mean((o[..., 0] - r[..., 0]) ** 2)

def mse_red_masked(o, r, red_thresh=0.4):
    # o, r expected in [0, 1]
    red = o[..., 0]
    green = o[..., 1]
    blue = o[..., 2]

    # Simple heuristic: red dominant over G/B and sufficiently strong
    red_mask = (red > red_thresh) & (red > green + 0.05) & (red > blue + 0.05)

    if not np.any(red_mask):
        # no red pixels in original – define metric as 0 or np.nan
        return 0.0

    diff = o[..., 0] - r[..., 0]
    return np.mean(diff[red_mask] ** 2)


def mse_hsv_value(o, r):
    hsv_o = rgb_to_hsv(o)
    hsv_r = rgb_to_hsv(r)
    return np.mean((hsv_o[..., 2] - hsv_r[..., 2]) ** 2)  # V channel

def mse_hsv_hue(o, r):
    hsv_o = rgb_to_hsv(o)
    hsv_r = rgb_to_hsv(r)
    return np.mean((hsv_o[..., 0] - hsv_r[..., 0]) ** 2)  # H channel

def mse_hsv_hue_circular(o, r):
    hsv_o = rgb_to_hsv(o)
    hsv_r = rgb_to_hsv(r)

    h1 = hsv_o[..., 0]
    h2 = hsv_r[..., 0]

    # circular difference on unit circle
    dh = np.abs(h1 - h2)
    dh = np.minimum(dh, 1.0 - dh)  # because 0 and 1 are the same hue
    return np.mean(dh ** 2)


def compute_metrics(originals, recons, filenames):
    rows = []
    for o, r, name in zip(originals, recons, filenames):
        o = o / 255.0
        r = r / 255.0
        row = {
            "filename":        name,
            "emr_dissim":      emr_dissimilarity(o, r),
            "mse_rgb":         mse_rgb(o, r),
            "max_p99":         max_percentile(o, r, p=99),
            "mse_red":         mse_red_masked(o, r),
            "mse_hsv_V":       mse_hsv_value(o, r),
            "mse_hsv_H":       mse_hsv_hue_circular(o, r)
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




if __name__ == "__main__":
    originals, recons, filenames, labels = load_pairs()
    print("Loaded pairs:", len(originals))

    df = compute_metrics(originals, recons, filenames)
    df["Presence"] = labels
    print(df.head())

    Path('reconstructions').mkdir(exist_ok=True)

    # Save metrics per patch so you can analyse them later
    df.to_csv(f"reconstructions/reconstruction_metrics{Config}.csv", index=False)
    print("saved reconstruction_metrics.csv")
     # Example: visualize top errors for each metric
    show_top(originals, recons, filenames, df["mse_red"].values, "MSE Red Channel (masked)", TOP_N)

    # show_top(originals, recons, filenames, df["mse_hsv_V"].values, "Value MSE HSV", TOP_N)
    # show_top(originals, recons, filenames, df["mse_rgb"].values,    "MSE RGB", TOP_N)
    # show_top(originals, recons, filenames, df["emr_dissim"].values, "EMR dissimilarity", TOP_N)
    # get the df created with the metrics and do 10-fold ROC curves for each metric
