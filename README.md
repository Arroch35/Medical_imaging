# Medical_imaging
Challenge 2 of the Vision &amp; Learning subject

## Table of Contents

- [Introduction](#1-introduction)
- [Project Structure](#2-project-structure)
- [Setup](#3-setup)
- [Usage and Workflow](#4-usage-and-workflow)

## 1. Introduction
This is the repository for the challenge 2 of the Vision & Learning subject. Represents 2 
systems that help diagnose whether a patient has Helicobacter pylori.

## 2. Project structure
```bash
MEDICAL_IMAGING/
├── src/
│   ├── FeatExtraction/         
│   ├── Models/                
│   ├── TripletLoss/           
│   │
│   ├── config.py               # Configuration file (paths)
│   ├── config.py               # Configuration file (paths)
│   │
│   ├── S1_1_VAE_training.py          # System 1: VAE model training
│   ├── S1_2_save_reconstructions.py  # System 1: Save VAE reconstruction images
│   ├── S1_3_compare_reconstructions.py # System 1: Compare VAE reconstructions
│   ├── S1_4_roc_curves.py            # System 1: Compute and plot ROC curves
│   ├── S1_5_patch_optimal_threshold.py # System 1: Find optimal threshold for patch-level anomaly
│   ├── S1_6_VAE_recon_per_patient.py # System 1: VAE reconstruction error per patient
│   ├── S1_7_VAE_patient_patch_classification.py # System 1: Patient patch-level classification
│   ├── S1_8_diagnosis_optimal_threshold.py # System 1: Optimal threshold for final diagnosis
│   ├── S1_9_final_diagnosis_patient.py # System 1: Compute final patient diagnosis metrics
│   │
│   ├── S2_1_training_AE.py           # System 2: Autoencoder (AE) model training
│   ├── S2_2_save_reconstructions.py  # System 2: Save AE reconstruction images
│   ├── S2_3_metrics.py               # System 2: Compute various evaluation metrics
│   ├── S2_4_compute_roc_curves.py    # System 2: Compute and plot ROC curves
│   ├── S2_5_0_extract_latent_vectors.py # System 2: Extract latent vectors from AE
│   ├── S2_5_1_train_pca.py           # System 2: Train PCA on latent vectors
│   ├── S2_5_2_contrastive_learning_training.py # System 2: Contrastive Learning training
│   ├── S2_6_save_latents.py          # System 2: Save latent representations
│   ├── S2_7_visualize_latent_spaces.py # System 2: Visualize the learned latent spaces
│   ├── S2_8_svm_training.py          # System 2: Train SVM on latent features
│   ├── S2_9_divide_crppod_for_patient_diagnosis.py # System 2: Divide data for patient diagnosis
│   ├── S2_10_1_find_threshold_patients_with_cl.py # System 2: Find threshold (with Contrastive Learning)
│   ├── S2_10_2_no_cl.py              # System 2: Find threshold (without Contrastive Learning)
│   ├── S2_11_1_validation.py         # System 2: Final validation (with Contrastive Learning)
│   ├── S2_11_2_no_cl.py              # System 2: Final validation (without Contrastive Learning)
│   │
│   └── utils.py                # Utility functions

``` 
## 3. Setup

To set up the project environment, run the following commands in the root directory:

```bash
# 1. Create a virtual environment (named 'venv')
python -m venv venv

# 2. Activate the environment (Linux/macOS)
source venv/bin/activate

# 2. Activate the environment (Windows)
# venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

The **`config.py`** file is used to configure all project and data path variables, ensuring consistency across the codebase. Change the value of the variable `YOUR_PATH_TO_THE_PROJECT` according to your path.

---

## 4. Usage and Workflow

All files are named by the order they must be run. SX correspond to the system the file belongs to, and the 
next number "SX_**X**" is the order in which the file must be executed.