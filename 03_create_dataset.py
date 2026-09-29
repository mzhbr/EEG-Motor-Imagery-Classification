import mne
import numpy as np

data_path = r"C:\Users\Asus\mne_data"

subjects = range(1, 11)
runs = [4, 8, 12]

all_epochs = []
all_labels = []
all_subjects = []

for subject in subjects:

    print(f"Processing Subject {subject}...")

    for run in runs:

        file_path = mne.datasets.eegbci.load_data(
            subject=subject,
            runs=[run],
            path=data_path,
            update_path=False
        )[0]

        raw = mne.io.read_raw_edf(
            file_path,
            preload=True,
            verbose=False
        )

        # Extract event markers
        events, event_id = mne.events_from_annotations(
            raw,
            verbose=False
        )

        # Keep only left- and right-hand motor imagery
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"left": 2, "right": 3},
            tmin=0,
            tmax=4,
            baseline=None,
            preload=True,
            verbose=False
        )

        # Store the EEG trials
        all_epochs.append(epochs.get_data())

        # Store labels
        labels = epochs.events[:, -1]

        all_labels.extend(labels)

        # Store subject ID
        all_subjects.extend(
            [subject] * len(labels)
        )

        print(
            f"  Run {run}: "
            f"{len(epochs)} trials"
        )

print("\nDone!")

print("Number of EEG groups:", len(all_epochs))
print("Total trials:", len(all_labels))
print("Labels:", np.unique(all_labels))
print("Subjects:", len(np.unique(all_subjects)))

# Combine all EEG trials into one array
X_eeg = np.concatenate(all_epochs, axis=0)

# Convert labels and subject IDs to arrays
y = np.array(all_labels)
groups = np.array(all_subjects)

print("\nFinal dataset:")
print("X shape:", X_eeg.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)

from mne.filter import filter_data

sfreq = 160

X_filtered = filter_data(
    X_eeg,
    sfreq=sfreq,
    l_freq=8,
    h_freq=30,
    verbose=False
)

print("\nAfter filtering:")
print("X_filtered shape:", X_filtered.shape)


from scipy.signal import welch
import numpy as np

bands = {
    "alpha": (8, 13),
    "beta": (13, 30)
}

features = []

for trial in X_filtered:

    trial_features = []

    for channel in trial:

        freqs, psd = welch(
            channel,
            fs=sfreq,
            nperseg=256
        )

        for band_name, (low, high) in bands.items():

            mask = (freqs >= low) & (freqs <= high)

            band_power = np.trapz(
                psd[mask],
                freqs[mask]
            )

            trial_features.append(band_power)

    features.append(trial_features)

X_features = np.array(features)

print("\nFeature extraction complete!")
print("X_features shape:", X_features.shape)

print("\nFinal ML dataset:")
print("X:", X_features.shape)
print("y:", y.shape)
print("groups:", groups.shape)

print("\nClass distribution:")
print("Left:", np.sum(y == 2))
print("Right:", np.sum(y == 3))

print("\nSubjects:")
print(np.unique(groups))

from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(cv.split(X_features, y, groups), 1):
    print(f"\nFold {fold}")
    print("Train subjects:", np.unique(groups[train_idx]))
    print("Test subjects:", np.unique(groups[test_idx]))
    print("Train samples:", len(train_idx))
    print("Test samples:", len(test_idx))


from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

cv = GroupKFold(n_splits=5)

accuracies = []
precisions = []
recalls = []
f1_scores = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_features, y, groups), 1
):

    # Split data
    X_train = X_features[train_idx]
    X_test = X_features[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Scale features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    model = SVC(kernel="rbf")

    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=3)
    recall = recall_score(y_test, y_pred, pos_label=3)
    f1 = f1_score(y_test, y_pred, pos_label=3)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(f"\nFold {fold}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\n===== Mean Results =====")
print(f"Accuracy:  {np.mean(accuracies):.3f}")
print(f"Precision: {np.mean(precisions):.3f}")
print(f"Recall:    {np.mean(recalls):.3f}")
print(f"F1-score:  {np.mean(f1_scores):.3f}")

# Save data for CSP analysis

np.save("X_eeg.npy", X_eeg)
np.save("X_filtered.npy", X_filtered)
np.save("y.npy", y)
np.save("groups.npy", groups)

print("\nData saved for CSP.")