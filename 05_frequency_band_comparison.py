import numpy as np
import mne

from mne.decoding import CSP
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================
# 1. Load data
# =========================

X_eeg = np.load("X_eeg.npy")
y = np.load("y.npy")
groups = np.load("groups.npy")

print("X_eeg shape:", X_eeg.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)


# =========================
# 2. Settings
# =========================

sfreq = 160

frequency_bands = {
    "8-30 Hz": (8, 30),
    "8-13 Hz (mu/alpha)": (8, 13),
    "13-30 Hz (beta)": (13, 30)
}

n_components = 4

# Fixed SVM parameters from previous experiments
C = 10
gamma = 0.001

group_kfold = GroupKFold(n_splits=5)


# =========================
# 3. Compare frequency bands
# =========================

for band_name, (low_freq, high_freq) in frequency_bands.items():

    print("\n" + "=" * 50)
    print(f"Frequency band: {band_name}")
    print("=" * 50)

    # Filter the raw EEG data
    X_filtered = mne.filter.filter_data(
        X_eeg,
        sfreq=sfreq,
        l_freq=low_freq,
        h_freq=high_freq,
        verbose=False
    )

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    # Subject-wise cross-validation
    for fold, (train_idx, test_idx) in enumerate(
        group_kfold.split(X_filtered, y, groups), start=1
    ):

        X_train = X_filtered[train_idx]
        X_test = X_filtered[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # -------------------------
        # CSP
        # -------------------------

        csp = CSP(
            n_components=n_components,
            reg=None,
            log=True,
            norm_trace=False
        )

        X_train_csp = csp.fit_transform(X_train, y_train)
        X_test_csp = csp.transform(X_test)

        # -------------------------
        # SVM
        # -------------------------

        model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma
        )

        model.fit(X_train_csp, y_train)

        y_pred = model.predict(X_test_csp)

        # -------------------------
        # Evaluation
        # -------------------------

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=3,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label=3,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label=3,
            zero_division=0
        )

        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.3f}, "
            f"Precision={precision:.3f}, "
            f"Recall={recall:.3f}, "
            f"F1={f1:.3f}"
        )

    # =========================
    # Mean results
    # =========================

    print("\nMean results:")
    print(f"Accuracy : {np.mean(fold_accuracies):.3f}")
    print(f"Precision: {np.mean(fold_precisions):.3f}")
    print(f"Recall   : {np.mean(fold_recalls):.3f}")
    print(f"F1-score : {np.mean(fold_f1s):.3f}")