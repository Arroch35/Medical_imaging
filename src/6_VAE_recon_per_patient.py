from torch.utils.data import DataLoader
import gc
import torch
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid

import torch.nn as nn
import torch.optim as optim
import tqdm


# Own Functions
from Models.AEmodels import VAECNN
from Models.datasets import Standard_Dataset
from utils import *
from configVAE import *


def mse_rgb(o, r):
    return np.mean((o - r) ** 2)


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

torch.backends.cudnn.benchmark = True

Config = '3'

def VAEConfigs(Config):
    inputmodule_paramsEnc = {'dim_input': 256, 'num_input_channels': 3}
    inputmodule_paramsDec = {'dim_input': 256}
    dim_in = inputmodule_paramsEnc['dim_input']

    net_paramsEnc = {}
    net_paramsDec = {}
    net_paramsRep = {}

    if Config == '1':
        net_paramsEnc['block_configs']=[[32, 32], [64, 64], [64, 64]]
        net_paramsEnc['stride']=[[1, 2], [1, 2], [1, 2]]
        net_paramsDec['block_configs']=[[64, 64], [64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsDec['block_configs'][0][0]

        net_paramsRep['h_dim']=65536
        net_paramsRep['z_dim']=256

    elif Config == '2':
        net_paramsEnc['block_configs']=[[32], [64], [128], [256]]
        net_paramsEnc['stride']=[[2], [2], [2], [2]]
        net_paramsDec['block_configs']=[[256], [128], [64], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsDec['block_configs'][0][0]

        net_paramsRep['h_dim']=65536
        net_paramsRep['z_dim']=512

    elif Config == '3':
        net_paramsEnc['block_configs']=[[32], [64], [64]]
        net_paramsEnc['stride']=[[1], [2], [2]]
        net_paramsDec['block_configs']=[[64], [32], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsDec['block_configs'][0][0]

        net_paramsRep['h_dim']=262144
        net_paramsRep['z_dim']=256

    return net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec, net_paramsRep
######################### 0. EXPERIMENT PARAMETERS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

crossval_cropped_folders = get_all_subfolders(CROPPED_PATCHES_DIR)
annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)


print(f"Found {len(crossval_cropped_folders)} cropped folders and {len(annotated_folders)} annotated folders.")
# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3

# 0.1 NETWORK TRAINING PARAMS
VAE_params = {
    'epochs': 50,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'img_size': (256, 256),
    'beta_start': 0.0,
    'beta_max': 1.0,
    'beta_warmup_epochs': 40,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# 0.2 FOLDERS

#### 1. LOAD DATA: Implement

# 1.1 Patient Diagnosis
cropped_folders = get_all_subfolders(CROPPED_PATCHES_DIR)

net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec, net_paramsRep = VAEConfigs(Config)

# Build VAE model (do not modify VAECNN implementation)
model = VAECNN(
    inputmodule_paramsEnc,
    net_paramsEnc,
    inputmodule_paramsDec,
    net_paramsDec,
    net_paramsRep
)


# Load checkpoint --> trained weights
ckpt_path = f"checkpoints/VAE_Config{Config}.pth"
ckpt = torch.load(ckpt_path, map_location=device)

state_dict = ckpt["state_dict"]

model.load_state_dict(state_dict)       # because your .pth is just a state_dict
model.to(VAE_params['device'])
model.eval()


if not model:
    raise RuntimeError("No models loaded. Checkpoint files missing.")
print("Model rebuilt and weights loaded.")


# print head of the state dict
for key, value in state_dict.items():
    print(f"state dict= {key}: {value.shape}")
    break  # print only the first item


list_ims_meta = LoadCropped_byPatient(cropped_folders, n_folders=155,
                                      resize=VAE_params['img_size'], verbose=False)
print(f"Loaded {len(list_ims_meta)} patients' cropped images and metadata.")

def _to_dataset(ims, meta, with_labels=False):
    X = np.stack([im.transpose(2, 0, 1) for im in ims], axis=0).astype(np.float32) / 255.0
    if with_labels:
        y = meta['Presence'].to_numpy(dtype=np.int64)
        return Standard_Dataset(X, y)
    else:
        return Standard_Dataset(X)

# 6. LOOP: RECONSTRUCT EACH PATIENT SEPARATELY
for patient_idx, patient_data in enumerate(list_ims_meta, start=1):
    patient_ims = patient_data[0]
    patient_meta = patient_data[1]

    patid = str(patient_meta["PatID"].iloc[0]) if len(patient_meta) > 0 else f"Unknown_{patient_idx}"

    print(f"\n=== Patient {patient_idx}: PatID {patid} ===")
    print(f"  -> ims shape:  {patient_ims.shape}")
    print(f"  -> meta shape: {patient_meta.shape}")
    print(patient_meta.head())

    # ---- Build Dataset & Loader ----
    vae_patient_ds = _to_dataset(patient_ims, patient_meta, with_labels=False)
    vae_patient_loader = DataLoader(
        vae_patient_ds,
        batch_size=VAE_params['batch_size'],
        shuffle=False
    )

    # ---- Reconstruction Saving ----
    total_images = len(patient_ims)
    pbar = tqdm.tqdm(total=total_images, desc=f"Saving recon for PatID {patid}")

    rows = []
    global_idx = 0

    with torch.no_grad():
        for batch in vae_patient_loader:

            if isinstance(batch, (list, tuple)):
                x_batch = batch[0]
            else:
                x_batch = batch

            x_batch = x_batch.to(device, dtype=torch.float32)

            # run VAE
            recon_batch, mu, logvar = model(x_batch)
            recon_batch = recon_batch.detach().cpu().clamp(0.0, 1.0)

            batch_size = recon_batch.shape[0] # preguntar
            # save each reconstruction
            for i in range(batch_size):

                if global_idx >= len(patient_meta):
                    break

                row = patient_meta.iloc[global_idx]
                filename = str(row["imfilename"])

                # images to arrays
                orig_np = patient_ims[global_idx].astype(np.float32) / 255.0   # (H, W, 3)
                recon_np = recon_batch[i].numpy().transpose(1, 2, 0)          # (H, W, 3), already [0,1]

                row_metrics = {
                    "PatID":       patid,
                    "imfilename":  filename,
                    "mse_rgb":     mse_rgb(orig_np, recon_np),
                    "mse_red":     mse_red_masked(orig_np, recon_np),
                }
                rows.append(row_metrics)
                global_idx += 1
                pbar.update(1)

    pbar.close()
    df_patient = pd.DataFrame(rows)

    # Folder for metrics (separate from images; we don't save PNGs anymore)
    metrics_dir = os.path.join(RECON_DIR, f"VAE_pat_metrics{Config}")
    os.makedirs(metrics_dir, exist_ok=True)

    csv_path = os.path.join(metrics_dir, f"{patid}.csv")
    df_patient.to_csv(csv_path, index=False)

    print(f"Finished saving reconstructions for PatID {patid}.")

print("\nAll patient reconstructions saved successfully!")



