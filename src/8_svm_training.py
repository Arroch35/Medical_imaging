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

KERNELS = ["linear", "poly", "rbf", "sigmoid"]
X = np.load("../data/latent_outputs/train_new_latents.npy")
y = np.load("../data/latent_outputs/train_new_labels.npy")

best_kernel = None
best_mean_auc = 0
best_std_auc = np.inf

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Store metrics for CSV
all_metrics = []

for kernel in KERNELS:
    aucs = []
    accs = []
    precs = []
    recalls = []
    f1s = []
    cms = []

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    # To collect all probabilities and true labels across CV for threshold
    all_y_val = []
    all_y_prob = []

    for train_idx, val_idx in kf.split(X_scaled, y):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = SVC(kernel=kernel, probability=True, random_state=42)
        clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)

        # Collect for threshold computation
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
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # Compute threshold using all CV predictions
    fpr_all, tpr_all, thresholds_all = roc_curve(all_y_val, all_y_prob)
    J = tpr_all - fpr_all
    idx = np.argmax(J)
    optimal_threshold = thresholds_all[idx]

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    mean_acc = np.mean(accs)
    mean_prec = np.mean(precs)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    mean_cm = np.mean(cms, axis=0)

    all_metrics.append([
        kernel, mean_auc, std_auc,
        mean_acc, mean_prec, mean_recall, mean_f1,
        optimal_threshold
    ])

    # Update best kernel
    if mean_auc > best_mean_auc or (np.isclose(mean_auc, best_mean_auc) and std_auc < best_std_auc):
        best_mean_auc = mean_auc
        best_std_auc = std_auc
        best_kernel = kernel

    # -----------------------------------------------------------
    # Plot ROC Curve
    # -----------------------------------------------------------
    plt.figure()
    mean_tpr = np.mean(tprs, axis=0)
    roc_auc_val = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, lw=2,
             label=f"AUC = {roc_auc_val:.3f} ± {std_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Kernel: {kernel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"roc_kernel_{kernel}.png")
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(mean_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.title(f"Mean Confusion Matrix - Kernel: {kernel}")
    plt.tight_layout()
    plt.savefig(f"cm_kernel_{kernel}.png")
    plt.close()

# Save metrics summary
df = pd.DataFrame(all_metrics, columns=[
    "Kernel", "Mean AUC", "AUC STD",
    "Accuracy", "Precision", "Recall", "F1-score",
    "Optimal Threshold"
])
out_dir = "../data/svm"
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, "svm_kernel_metrics.csv"), index=False)

print(df)
print(f"\nSelected kernel: {best_kernel} (Mean AUC={best_mean_auc:.4f}, Std={best_std_auc:.4f})")

# Train final SVM on full data
final_clf = SVC(kernel=best_kernel, probability=True, random_state=42)
final_clf.fit(X_scaled, y)

# Save scaler + SVM
joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
joblib.dump(final_clf, os.path.join(out_dir, "best_svm_auc.pkl"))

print(f"Optimal threshold for final kernel ({best_kernel}) can be retrieved from the CSV file.")
