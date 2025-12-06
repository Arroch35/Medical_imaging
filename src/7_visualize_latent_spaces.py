import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR

PLOT_DIR = DATA_DIR / "plots_latent_space"
os.makedirs(PLOT_DIR, exist_ok=True)

def visualize_latent_spaces(old_latents, new_latents, labels, title_suffix=""):
    """
    Visualize original and triplet-learned latent spaces using t-SNE
    and save the output plots to ../data/plots_latent_space.
    """

    # Normalize
    scaler_old = StandardScaler()
    scaler_new = StandardScaler()

    old_latents_norm = scaler_old.fit_transform(old_latents)
    new_latents_norm = scaler_new.fit_transform(new_latents)

    perplexity_old = min(30, len(old_latents_norm) - 1)
    perplexity_new = min(30, len(new_latents_norm) - 1)

    # t-SNE
    print(f"Running t-SNE on original latent space {title_suffix}...")
    tsne_old = TSNE(n_components=2, perplexity=perplexity_old, random_state=42)
    old_2d = tsne_old.fit_transform(old_latents_norm)

    print(f"Running t-SNE on triplet-learned latent space {title_suffix}...")
    tsne_new = TSNE(n_components=2, perplexity=perplexity_new, random_state=42)
    new_2d = tsne_new.fit_transform(new_latents_norm)

    # Plot
    plt.figure(figsize=(14,6))

    # Original latent space
    plt.subplot(1,2,1)
    plt.scatter(old_2d[:,0], old_2d[:,1], c=labels, cmap="coolwarm", alpha=0.7, s=12)
    plt.title(f"Original Latent Space {title_suffix}")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True)

    # Triplet latent space
    plt.subplot(1,2,2)
    plt.scatter(new_2d[:,0], new_2d[:,1], c=labels, cmap="coolwarm", alpha=0.7, s=12)
    plt.title(f"Triplet-Learned Space {title_suffix}")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True)

    plt.tight_layout()

    # ---------- SAVE PLOT ----------
    suffix_clean = title_suffix.replace("(", "").replace(")", "").replace(" ", "_")
    filename = f"latent_space_{suffix_clean}.png"
    save_path = PLOT_DIR / filename
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot → {save_path}")

    plt.close()


# -------------------------------------------------
# Load train and test latent vectors + labels
# -------------------------------------------------
latent_dir = DATA_DIR / "latent_outputs"

# Train split
old_lat_train = np.load(latent_dir / "train_old_latents.npy")
new_lat_train = np.load(latent_dir / "train_new_latents.npy")
y_train = np.load(latent_dir / "train_old_labels.npy")  # same for new

# Test split
old_lat_test = np.load(latent_dir / "test_old_latents.npy")
new_lat_test = np.load(latent_dir / "test_new_latents.npy")
y_test = np.load(latent_dir / "test_old_labels.npy")

# -------------------------------------------------
# Visualize
# -------------------------------------------------
print("Visualizing TRAIN split...")
visualize_latent_spaces(old_lat_train, new_lat_train, y_train, title_suffix="Train")

print("Visualizing TEST split...")
visualize_latent_spaces(old_lat_test, new_lat_test, y_test, title_suffix="Test")

# todo: cambiar los string paths paths del config file
# poner a entrenar el cl model, que ha dado error en el cluster
# Cambiar las cosas que pone de cpu a gpu
# Probar los AE entrenados nuevos