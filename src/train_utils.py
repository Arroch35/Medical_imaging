import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

def _read_metadata_excel(excel_path):
    """
    Read an excel metadata file and return a DataFrame.
    If excel_path is None or doesn't exist -> returns empty DataFrame.
    """
    if excel_path is None:
        return pd.DataFrame()
    if not os.path.exists(excel_path):
        print(f"[Warning] Metadata file not found: {excel_path}")
        return pd.DataFrame()
    try:
        df = pd.read_excel(excel_path)
    except Exception:
        # Try CSV fallback
        try:
            df = pd.read_csv(excel_path)
        except Exception:
            print(f"[Warning] Could not read metadata file: {excel_path}")
            return pd.DataFrame()
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df


def _window_id_from_filename(fname):
    """
    Normalize a filename to a Window_ID style used in excel:
    strips extension and returns base name.
    """
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    return name

def LoadCropped(list_folders, n_images_per_folder=None, excelFile=None, resize=None, verbose=False):
    """
    Load cropped patches (non-annotated), optionally resizing images.
    If a PatientDiagnosis.csv file (with columns CODI, DENSITAT) is provided,
    only patches from healthy patients (DENSITAT == 'NEGATIVA') are loaded.

    Parameters
    ----------
    list_folders : list of str
        Paths containing .png patches (e.g. PatID_Section# folders)
    n_images_per_folder : int or None
        Limit number of patches per folder (first N if provided)
    excelFile : str or None
        Path to PatientDiagnosis.csv (columns: CODI, DENSITAT)
    resize : tuple(int, int) or None
        Target image size (H, W). If None, keep original 256x256.
    verbose : bool
        Print progress messages if True.

    Returns
    -------
    Ims : np.ndarray
        Array of images with shape (N, H, W, 3), dtype=uint8
    metadata : pd.DataFrame
        DataFrame with columns ['PatID', 'imfilename']
    """
    
    healthy_pats = None
    if excelFile is not None:
        if not os.path.exists(excelFile):
            raise FileNotFoundError(f"[LoadCropped] File not found: {excelFile}")

        try:
            df_diag = pd.read_csv(excelFile)
        except Exception:
            df_diag = pd.read_excel(excelFile)

        # Normalize column names
        df_diag.columns = [c.strip().upper() for c in df_diag.columns]
        if not {"CODI", "DENSITAT"}.issubset(df_diag.columns):
            raise ValueError("Diagnosis file must contain columns: 'CODI' and 'DENSITAT'")

        # Select healthy patients: DENSITAT == 'NEGATIVA'
        healthy_mask = df_diag["DENSITAT"].astype(str).str.upper().str.strip() == "NEGATIVA"
        healthy_pats = set(df_diag.loc[healthy_mask, "CODI"].astype(str))
        if verbose:
            print(f"[LoadCropped] Found {len(healthy_pats)} healthy patients in {excelFile}")


    records, images = [], []

    for folder in list_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        patid = folder_name.split("_")[0]  

        # Skip if not a healthy patient (if CSV provided)
        if healthy_pats is not None and patid not in healthy_pats:
            if verbose:
                print(f"[LoadCropped] Skipping non-healthy patient: {patid}")
            continue

        if not os.path.isdir(folder):
            if verbose:
                print(f"[LoadCropped] Folder not found, skipping: {folder}")
            continue

        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if n_images_per_folder is not None:
            files = files[:n_images_per_folder]

        for fpath in files:
            try:
                im = Image.open(fpath).convert("RGB")
                if resize is not None:
                    im = im.resize(resize, Image.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)
            except Exception as e:
                if verbose:
                    print(f"[LoadCropped] Failed to read {fpath}: {e}")
                continue

            filename = os.path.basename(fpath)
            record = {"PatID": patid, "imfilename": filename}
            records.append(record)
            images.append(arr)


    if len(images) == 0:
        H, W = resize if resize is not None else (256, 256)
        Ims = np.zeros((0, H, W, 3), dtype=np.uint8)
    else:
        Ims = np.stack(images, axis=0).astype(np.uint8)

    metadata = pd.DataFrame.from_records(records)
    if verbose:
        print(f"[LoadCropped] Loaded {len(images)} images from {len(metadata['PatID'].unique())} healthy patients.")
    return Ims, metadata


def LoadAnnotated(list_folders, n_images_per_folder=None, excelFile=None, resize=None, verbose=False):
    """
    Load annotated patches (presence of H. pylori known), optionally resizing images.

    Parameters
    ----------
    list_folders : list
        Paths containing annotated .png patches
    n_images_per_folder : int or None
        Limit number of patches per folder (first N if provided)
    excelFile : str or None
        Metadata Excel with 'Pat_ID', 'Window_ID', 'Presence' columns
    resize : tuple(int,int) or None
        Target image size (H, W). If None, keep original 256x256.
    verbose : bool
        Print progress messages if True

    Returns
    -------
    Ims : np.ndarray
        Images as (N, H, W, 3) array, dtype=uint8
    metadata : pd.DataFrame
        DataFrame with columns ['PatID', 'imfilename', 'Presence']
    """
    df = _read_metadata_excel(excelFile)
    records = []
    images = []

    for folder in list_folders:
        if not os.path.isdir(folder):
            if verbose:
                print(f"[LoadAnnotated] folder not found, skipping: {folder}")
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if n_images_per_folder is not None:
            files = files[:n_images_per_folder]

        for fpath in files:
            try:
                im = Image.open(fpath).convert("RGB")
                if resize is not None:
                    im = im.resize(resize, Image.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)
            except Exception as e:
                if verbose:
                    print(f"[LoadAnnotated] failed to read {fpath}: {e}")
                continue

            filename = os.path.basename(fpath)
            window_id = _window_id_from_filename(filename)
            folder_name = os.path.basename(os.path.normpath(folder))
            if "_" in folder_name:
                patid, section = folder_name.split("_", 1)
            else:
                patid = folder_name
                section = ""

            presence_val = None
            if not df.empty:
                if 'Aug' in window_id:
                    parts = window_id.split('_')
                    window_id_clean = f"{int(parts[0])}_{parts[1]}"
                else:
                    try:
                        window_id_clean = str(int(window_id))
                    except ValueError:
                        window_id_clean = window_id

                m = df[df['Window_ID'].astype(str).str.strip() == window_id_clean]
                if m.empty and 'Pat_ID' in df.columns:
                    m = df[(df['Window_ID'].astype(str).str.strip() == window_id_clean)
                           & (df['Pat_ID'].astype(str).str.strip() == patid)]
                if m.empty:
                    m = df[df['Window_ID'].astype(str).str.strip() == filename]
                if not m.empty and 'Presence' in m.columns:
                    presence_val = m.iloc[0]['Presence']

                if presence_val == 0 or presence_val == None:
                    continue

            record = {'PatID': patid, 'imfilename': filename, 'Presence': presence_val}
            images.append(arr)
            records.append(record)

    if len(images) == 0:
        H, W = resize if resize is not None else (256, 256)
        Ims = np.zeros((0, H, W, 3), dtype=np.uint8)
    else:
        Ims = np.stack(images, axis=0).astype(np.uint8)

    metadata = pd.DataFrame.from_records(records)
    return Ims, metadata

def get_all_subfolders(root_dir):
    """
    Return a sorted list of all subfolders in root_dir (recursively).
    Each folder typically corresponds to one patient/section.
    """
    subfolders = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            subfolders.append(os.path.join(root, d))
    subfolders = sorted(subfolders)
    return subfolders