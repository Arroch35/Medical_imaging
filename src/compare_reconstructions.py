import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import rgb_to_hsv
from sklearn.metrics import roc_curve, auc
import pandas as pd

from config2 import *

TOP_N = 20


# loads an image and converts to RGB numpy array
def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


# loads all original-reconstruction pairs
def load_pairs():
    """
    Returns:
        originals, recons : list of HxWx3 float32
        filenames         : list of "PatID/filename.png"
        labels            : np.array of 0/1 (Presence)
    """
    originals, recons, filenames = [], [], []

    # Walk recursively across all patient subfolders
    for root, _, files in os.walk(ANNOTATED_PATCHES_DIR):

        for fname in files:
            if not fname.lower().endswith(".png"):
                continue

            orig_path = os.path.join(root, fname)

            # build relative path:
            rel_path = os.path.relpath(os.path.join(root[:-2], fname), ANNOTATED_PATCHES_DIR)

            # reconstruction must be in same relative location
            recon_path = os.path.join(RECON_DIR, "Config1", rel_path)

            if not os.path.exists(recon_path):
                print(f"[Warning] Missing reconstruction: {rel_path}")
                continue

            originals.append(load_image_rgb(orig_path))
            recons.append(load_image_rgb(recon_path))
            filenames.append(rel_path)  # store patient/id/imgname

    return originals, recons, filenames


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
            "mse_hsv_H":       mse_hsv_hue(o, r),
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
    originals, recons, filenames = load_pairs()
    print("Loaded pairs:", len(originals))

    df = compute_metrics(originals, recons, filenames)
    print(df.head())

    # Save metrics per patch so you can analyse them later
    df.to_csv("reconstruction_metrics.csv", index=False)

    # Example: visualize top errors for each metric
    show_top(originals, recons, filenames, df["mse_hsv_V"].values, "Value MSE HSV", TOP_N)
    show_top(originals, recons, filenames, df["mse_rgb"].values,    "MSE RGB", TOP_N)
    show_top(originals, recons, filenames, df["emr_dissim"].values, "EMR dissimilarity", TOP_N)
