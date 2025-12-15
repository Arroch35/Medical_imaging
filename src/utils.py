import numpy as np
import pandas as pd
import glob
from PIL import Image
import os

from Models.datasets import Standard_Dataset


def AEConfigs(Config, inputmodule_paramsEnc):
    print(Config)
    net_paramsEnc={}
    net_paramsDec={}
    inputmodule_paramsDec={}
    if Config=='1':
        # CONFIG1
        net_paramsEnc['block_configs']=[[32,32],[64,64]]
        net_paramsEnc['stride']=[[1,2],[1,2]]
        net_paramsDec['block_configs']=[[64,32],[32,inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
     

        
    elif Config=='2':
        # CONFIG 2
        net_paramsEnc['block_configs']=[[32],[64],[128],[256]]
        net_paramsEnc['stride']=[[2],[2],[2],[2]]
        net_paramsDec['block_configs']=[[128],[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
   
        
    elif Config=='3':  
        # CONFIG3
        net_paramsEnc['block_configs']=[[32],[64],[64]]
        net_paramsEnc['stride']=[[1],[2],[2]]
        net_paramsDec['block_configs']=[[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
    return net_paramsEnc,net_paramsDec,inputmodule_paramsDec

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
    all_patids = set()
    skipped_patids = set()

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
        all_patids.add(patid)

        # Skip if not a healthy patient (if CSV provided)
        if healthy_pats is not None and patid not in healthy_pats:
            skipped_patids.add(patid)
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


    # Calculate percentage of non-healthy patients
    total_patients = len(all_patids)
    non_healthy_count = len(skipped_patids)
    if total_patients > 0:
        perc_non_healthy = 100 * non_healthy_count / total_patients
    else:
        perc_non_healthy = 0.0

    if verbose:
        print(f"[LoadCropped] Total patients found: {total_patients}")
        print(f"[LoadCropped] Non-healthy patients skipped: {non_healthy_count} ({perc_non_healthy:.2f}%)")
        print(f"[LoadCropped] Loaded {len(images)} images from {len(records)} healthy patients.")


    if len(images) == 0:
        H, W = resize if resize is not None else (256, 256)
        Ims = np.zeros((0, H, W, 3), dtype=np.uint8)
    else:
        Ims = np.stack(images, axis=0).astype(np.uint8)

    metadata = pd.DataFrame.from_records(records)
    if verbose:
        print(f"[LoadCropped] Loaded {len(images)} images from {len(metadata['PatID'].unique())} healthy patients.")
    return Ims, metadata

def LoadCropped_byPatient(list_folders, n_folders, resize=None, verbose=False):
    if not isinstance(list_folders, (list, tuple)):
        raise ValueError("[LoadCroppedByFolders] 'list_folders' must be a list of folder paths.")

    if n_folders <= 0:
        raise ValueError("[LoadCroppedByFolders] 'n_folders' must be a positive integer.")

    # Clip n_folders to the available number of folders
    n_folders = min(n_folders, len(list_folders))
    selected_folders = list_folders[:n_folders]

    if verbose:
        print(f"[LoadCroppedByFolders] Using {n_folders} folder(s) out of {len(list_folders)} provided.")

    per_folder_data = []

    for idx, folder in enumerate(selected_folders):
        if verbose:
            print(f"[LoadCroppedByFolders] ({idx+1}/{n_folders}) Loading folder: {folder}")

        # Reuse existing LoadCropped logic for a single folder
        Ims, metadata = LoadCropped(
            list_folders=[folder],
            n_images_per_folder=None,  # load all images in this folder
            excelFile=None,            # no diagnosis filtering here
            resize=resize,
            verbose=verbose
        )

        per_folder_data.append((Ims, metadata))

        if verbose:
            print(f"[LoadCroppedByFolders]   -> Loaded {len(metadata)} images for PatID(s): "
                  f"{metadata['PatID'].unique().tolist() if not metadata.empty else '[]'}")

    return per_folder_data

def GetImagePaths(list_folders, n_images_per_folder=None, excelFile=None, verbose=False):
    """
    Returns image file paths + metadata without loading images.
    """

    healthy_pats = None
    all_patids = set()
    skipped_patids = set()

    if excelFile is not None:
        df_diag = pd.read_csv(excelFile) if os.path.isfile(excelFile) else None
        df_diag.columns = [c.strip().upper() for c in df_diag.columns]

        healthy_mask = df_diag["DENSITAT"].astype(str).str.upper().str.strip() == "NEGATIVA"
        healthy_pats = set(df_diag.loc[healthy_mask, "CODI"].astype(str))
        if verbose:
            print(f"[GetImagePaths] Healthy patients: {len(healthy_pats)}")

    records = []

    for folder in list_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        patid = folder_name.split("_")[0]
        all_patids.add(patid)
        
        # Exclude known contaminated patients
        exclude_list = [
        'B22-108', 'B22-112', 'B22-142', 'B22-143', 'B22-149', 'B22-150',
        'B22-151', 'B22-152', 'B22-172', 'B22-270', 'B22-94', 'B22-93',
        'B22-61', 'B22-53', 'B22-34', 'B22-284', 'B22-165', 'B22-187',
        'B22-210', 'B22-90'
        ]       
        if patid in exclude_list:
            skipped_patids.add(patid)
            if verbose:
                print(f"[GetImagePaths] Excluding flagged contaminated patient: {patid}")
            continue

        if healthy_pats is not None and patid not in healthy_pats:
            skipped_patids.add(patid)
            if verbose:
                print(f"[GetImagePaths] Skipping non-healthy: {patid}")
            continue

        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if n_images_per_folder is not None:
            files = files[:n_images_per_folder]

        for fpath in files:
            records.append({"PatID": patid, "path": fpath})

    metadata = pd.DataFrame.from_records(records)

    if verbose:
        print(f"[GetImagePaths] Total images: {len(metadata)}")
        print(f"[GetImagePaths] Unique patients: {len(metadata['PatID'].unique())}")

    return metadata


def LoadAnnotated(list_folders, patient_excel, n_images_per_folder=None, excelFile=None, resize=None, verbose=False):
    """
    Load annotated patches for patients with **non-negative diagnosis**, and compute the percentage.

    Parameters
    ----------
    list_folders : list
        Paths containing annotated .png patches
    patient_excel : str
        Excel/CSV file with patient diagnoses (columns: 'CODI', 'DENSITAT')
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
    # Load patient diagnosis file
    if not os.path.exists(patient_excel):
        raise FileNotFoundError(f"[LoadAnnotatedPositivePatients] Patient file not found: {patient_excel}")
    
    try:
        df_patients = pd.read_csv(patient_excel)
    except Exception:
        df_patients = pd.read_excel(patient_excel)

    df_patients.columns = [c.strip().upper() for c in df_patients.columns]
    if not {"CODI", "DENSITAT"}.issubset(df_patients.columns):
        raise ValueError("Patient file must contain columns: 'CODI' and 'DENSITAT'")

    # Keep only non-negative patients
    non_negative_pats = set(df_patients.loc[df_patients['DENSITAT'].str.upper().str.strip() != "NEGATIVA", "CODI"].astype(str))

    # Track all patients in the folders
    all_patients = set()
    for folder in list_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        patid = folder_name.split("_")[0] if "_" in folder_name else folder_name
        all_patients.add(patid)

    # Compute percentage
    total_patients = len(all_patients)
    non_negative_count = len([p for p in all_patients if p in non_negative_pats])
    perc_non_negative = 100 * non_negative_count / total_patients if total_patients > 0 else 0.0
    if verbose:
        print(f"[LoadAnnotated] Total patients found: {total_patients}")
        print(f"[LoadAnnotated] Non-negative patients: {non_negative_count} ({perc_non_negative:.2f}%)")

    # Load annotation metadata
    df = _read_metadata_excel(excelFile)
    records = []
    images = []

    for folder in list_folders:
        if not os.path.isdir(folder):
            if verbose:
                print(f"[LoadAnnotated] folder not found, skipping: {folder}")
            continue

        folder_name = os.path.basename(os.path.normpath(folder))
        patid = folder_name.split("_")[0] if "_" in folder_name else folder_name

        # # Skip if patient is negative
        # if patid not in non_negative_pats:
        #     continue

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

            # Get presence if excelFile is provided
            presence_val = None
            if df is not None and not df.empty:
                if 'Aug' in window_id:
                    parts = window_id.split('_')
                    window_id_clean = f"{int(parts[0])}_{parts[1]}"
                else:
                    try:
                        window_id_clean = str(int(window_id))
                    except ValueError:
                        window_id_clean = window_id

                
                m = df[(df['Window_ID'].astype(str).str.strip() == window_id_clean) &
                       (df['Pat_ID'].astype(str).str.strip() == patid)]
                if not m.empty and 'Presence' in m.columns:
                    presence_val = m.iloc[0]['Presence']
                else:
                    continue

                if presence_val not in [-1, 1]:
                    continue
                if presence_val == -1: presence_val = 0 #ROC curve expects 0/1 values

            record = {'PatID': patid, 'imfilename': filename, 'Presence': presence_val}
            images.append(arr)
            records.append(record)

    if len(images) == 0:
        H, W = resize if resize is not None else (256, 256)
        Ims = np.zeros((0, H, W, 3), dtype=np.uint8)
    else:
        Ims = np.stack(images, axis=0).astype(np.uint8)

    metadata = pd.DataFrame.from_records(records)
    if verbose:
        print(f"[LoadAnnotated] Loaded {len(images)} images from {len(metadata['PatID'].unique())} non-negative patients.")
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

def to_dataset(ims, meta, with_labels=False):
    X = np.stack([im.transpose(2,0,1) for im in ims], axis=0) / 255.0 
    if with_labels:
        y = meta['Presence'].to_numpy(dtype=np.int64) 
        return Standard_Dataset(X, y)
    else:
        return Standard_Dataset(X)