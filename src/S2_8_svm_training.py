from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
import pandas as pd
import os

# ------------------------------------------------------------
# Load TRAIN + TEST latents
# ------------------------------------------------------------
X_train = np.load("../data/latent_outputs/train_new_latents.npy")
X_test  = np.load("../data/latent_outputs/test_new_latents.npy")

y_train = np.load("../data/latent_outputs/train_new_labels.npy")
y_test  = np.load("../data/latent_outputs/test_new_labels.npy")

#X_train = train["latents"]
#y_train = train["labels"]

#X_test  = test["latents"]
#y_test  = test["labels"]

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
KERNELS = ["linear", "poly", "rbf", "sigmoid"]

best_kernel = None
best_cv_auc = -1
best_test_auc = -1
best_std_auc = np.inf

# Scale only on train, apply to test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

out_dir = "../data/svm"
os.makedirs(out_dir, exist_ok=True)

# Store metrics for CSV
all_metrics = []

# ------------------------------------------------------------
# Evaluate each kernel
# ------------------------------------------------------------
for kernel in KERNELS:
    aucs = []
    accs = []
    precs = []
    recalls = []
    f1s = []
    cms = []

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    all_y_val = []
    all_y_prob = []

    # ------------------------- CV loop -------------------------
    for train_idx, val_idx in kf.split(X_train_scaled, y_train):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        clf = SVC(kernel=kernel, probability=True, random_state=42)
        clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)

        # Collect for threshold
        all_y_val.extend(y_val)
        all_y_prob.extend(y_prob)

        # Metrics
        aucs.append(roc_auc_score(y_val, y_prob))
        accs.append(accuracy_score(y_val, y_pred))
        precs.append(precision_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        cms.append(confusion_matrix(y_val, y_pred, normalize="true"))

        fpr, tpr, _ = roc_curve(y_val, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0
        tprs.append(interp_tpr)

    # ------------------------- Compute CV statistics -------------------------
    fpr_all, tpr_all, thr_all = roc_curve(all_y_val, all_y_prob)
    J = tpr_all - fpr_all
    optimal_threshold = thr_all[np.argmax(J)]

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    mean_acc = np.mean(accs)
    mean_prec = np.mean(precs)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    mean_cm = np.mean(cms, axis=0)

    # --------------------- Test set evaluation ---------------------
    clf_test = SVC(kernel=kernel, probability=True, random_state=42)
    clf_test.fit(X_train_scaled, y_train)

    y_test_prob = clf_test.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = clf_test.predict(X_test_scaled)

    test_auc = roc_auc_score(y_test, y_test_prob)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_cm = confusion_matrix(y_test, y_test_pred, normalize="true")

    # --------------------- Save metrics ---------------------
    all_metrics.append([
        kernel,
        mean_auc, std_auc,
        mean_acc, mean_prec, mean_recall, mean_f1, optimal_threshold,
        test_auc, test_acc, test_prec, test_recall, test_f1
    ])

    # --------------------- Update best kernel decision ---------------------
    if (mean_auc > best_cv_auc) or \
       (np.isclose(mean_auc, best_cv_auc) and test_auc > best_test_auc) or \
       (np.isclose(mean_auc, best_cv_auc) and np.isclose(test_auc, best_test_auc) and std_auc < best_std_auc):

        best_kernel = kernel
        best_cv_auc = mean_auc
        best_test_auc = test_auc
        best_std_auc = std_auc

    # --------------------- CV ROC plot ---------------------
    plt.figure()
    mean_tpr = np.mean(tprs, axis=0)
    roc_auc_val = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"AUC={roc_auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC Curve (CV) - {kernel}")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_cv_{kernel}_no_cl.png")
    plt.close()

    # --------------------- Test ROC plot ---------------------
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    plt.figure()
    plt.plot(fpr_test, tpr_test, lw=2, label=f"AUC={test_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC Curve (Test) - {kernel}")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_test_{kernel}.png")
    plt.close()

    # --------------------- Test Confusion Matrix ---------------------
    plt.figure(figsize=(4,3))
    sns.heatmap(test_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["0","1"], yticklabels=["0","1"])
    plt.title(f"Test Confusion Matrix - {kernel}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cm_test_{kernel}.png")
    plt.close()

# ------------------------------------------------------------
# Save metrics CSV
# ------------------------------------------------------------
df = pd.DataFrame(all_metrics, columns=[
    "Kernel",
    "CV Mean AUC", "CV AUC Std",
    "CV Accuracy", "CV Precision", "CV Recall", "CV F1",
    "CV Optimal Threshold",
    "Test AUC", "Test Accuracy", "Test Precision", "Test Recall", "Test F1"
])

df.to_csv(os.path.join(out_dir, "svm_kernel_metrics.csv"), index=False)
print(df)
print(f"\nSelected kernel: {best_kernel}")
print(f"CV Mean AUC={best_cv_auc:.4f}, Test AUC={best_test_auc:.4f}, CV Std={best_std_auc:.4f}")

# ------------------------------------------------------------
# Train final SVM using best kernel
# ------------------------------------------------------------
final_clf = SVC(kernel=best_kernel, probability=True, random_state=42)
final_clf.fit(X_train_scaled, y_train)

joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
joblib.dump(final_clf, os.path.join(out_dir, "best_svm.pkl"))

print("Saved best SVM and scaler.")
