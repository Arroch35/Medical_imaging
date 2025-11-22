import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import mean_squared_error

from config import ANNOTATED_PATCHES_DIR, RECON_DIR

TOP_N = 20


def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def load_pairs():
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


def compute_mse_list(originals, recons):
    return np.array([
        mean_squared_error(o, r)
        for o, r in zip(originals, recons)
    ])


def show_top_errors(originals, recons, filenames, mse, top_n=20):
    idxs = np.argsort(mse)[-top_n:]

    for i in reversed(idxs):
        print(f"{filenames[i]} — MSE = {mse[i]:.6f}")

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(originals[i].astype(np.uint8))
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(recons[i].astype(np.uint8))
        axs[1].set_title(f"Reconstruction\nMSE = {mse[i]:.5f}")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()


# MAIN
originals, recons, filenames = load_pairs()
print(len(originals))
mse = compute_mse_list(originals, recons)
show_top_errors(originals, recons, filenames, mse, top_n=TOP_N)
