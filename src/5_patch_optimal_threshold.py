from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


def get_optimal_thresholds(df, metric_name, labels_column="Presence"):
    """
    Compute optimal (Youden-J) thresholds for each fold of a 10-fold split,
    without modifying the ROC plotting function.
    """

    y = df[labels_column].values
    scores = df[metric_name].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    results = []

    for fold_idx, (_, test_idx) in enumerate(kf.split(scores), start=1):
        y_true = y[test_idx]
        y_score = scores[test_idx]

        # ROC curve for this fold
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # Youden’s J = TPR - FPR
        J = tpr - fpr
        best_idx = np.argmax(J)

        results.append({
            "fold": fold_idx,
            "best_threshold": thresholds[best_idx],
            "best_tpr": tpr[best_idx],
            "best_fpr": fpr[best_idx],
            "youden_J": J[best_idx]
        })

    return pd.DataFrame(results)


#get reconstruction csv
Config = '3'
reconstruction_df = pd.read_csv(f"reconstruction_metrics{Config}.csv")
thresholds_mse_red = get_optimal_thresholds(reconstruction_df, "mse_red")
print(thresholds_mse_red)

thresholds_mse_red.to_csv(f"optimal_thresholds_mse_red_Config{Config}.csv", index=False)
