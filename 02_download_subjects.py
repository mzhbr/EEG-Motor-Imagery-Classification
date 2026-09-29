import mne
from pathlib import Path

data_path = Path(r"C:\Users\Asus\mne_data")

subjects = range(1, 11)
runs = [4, 8, 12]

files = []

for subject in subjects:
    for run in runs:
        subject_files = mne.datasets.eegbci.load_data(
            subject=subject,
            runs=[run],
            path=data_path,
            update_path=False
        )

        files.extend(subject_files)

print("Total files:", len(files))
print(files[:5])