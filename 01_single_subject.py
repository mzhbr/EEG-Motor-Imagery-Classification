import mne
import matplotlib.pyplot as plt

file_path = r"C:\Users\Asus\mne_data\MNE-eegbci-data\files\eegmmidb\1.0.0\S001\S001R04.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

events, event_id = mne.events_from_annotations(raw)

epochs = mne.Epochs(
    raw,
    events,
    event_id={"left": 2, "right": 3},
    tmin=0,
    tmax=4,
    baseline=None,
    preload=True
)

# Filtering
epochs_filtered = epochs.copy().filter(
    l_freq=8,
    h_freq=30
)

print(epochs_filtered)

# Compare original and filtered C3 signal

c3_idx = epochs.ch_names.index("C3..")

original = epochs["left"].get_data()[0][c3_idx]
filtered = epochs_filtered["left"].get_data()[0][c3_idx]

times = epochs.times

plt.figure(figsize=(12, 5))

plt.plot(times, original, label="Original")
plt.plot(times, filtered, label="Filtered")

plt.xlabel("Time (s)")
plt.ylabel("EEG amplitude")
plt.title("C3 - Original vs Filtered EEG")
plt.legend()

plt.tight_layout()
plt.show(block=True)

import numpy as np
from scipy.signal import welch

# Get EEG data
data = epochs_filtered.get_data()

print("Data shape:", data.shape)

# Frequency bands
bands = {
    "alpha": (8, 13),
    "beta": (13, 30)
}

# Sampling frequency
sfreq = epochs_filtered.info["sfreq"]

features = []

for epoch in data:

    epoch_features = []

    for channel in epoch:

        freqs, psd = welch(
            channel,
            fs=sfreq,
            nperseg=256
        )

        for band_name, (low, high) in bands.items():

            band_power = np.mean(
                psd[(freqs >= low) & (freqs <= high)]
            )

            epoch_features.append(band_power)

    features.append(epoch_features)

features = np.array(features)

print("Feature shape:", features.shape)

# Machine learning input

X = features

# Labels: left = 0, right = 1
y = np.array([
    0 if event == 2 else 1
    for event in epochs.events[:, -1]
])

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", y)