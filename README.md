# EEG Motor Imagery Classification Using Machine Learning

## Overview

This project investigates whether machine learning can distinguish between **left-hand and right-hand motor imagery** from electroencephalography (EEG) signals.

The project uses EEG recordings from the **PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)** and explores signal preprocessing, frequency-domain feature extraction, Common Spatial Pattern (CSP) analysis, and machine learning classification.

The main research question is:

> **Can machine learning distinguish left- and right-hand motor imagery from EEG signals?**

A key focus of this project is **subject-independent evaluation**, where data from the same participant are kept entirely within either the training or testing set.

---

## Dataset

This project uses the [PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)](https://physionet.org/content/eegmmidb/1.0.0/).

For this analysis, recordings from **10 subjects (subjects 1–10)** were used, with runs 4, 8, and 12 selected for the left/right motor-imagery task.

| Property           |                  Value |
| ------------------ | ---------------------: |
| Subjects           |                     10 |
| Runs per subject   |                      3 |
| Runs               |               4, 8, 12 |
| EEG channels       |                     64 |
| Sampling frequency |                 160 Hz |
| Total trials       |                    450 |
| Epoch duration     |              4 seconds |
| Classes            | Left-hand / Right-hand |

The final dataset contains:

* 230 left-hand trials
* 220 right-hand trials

---

## Analysis Pipeline

The overall pipeline is:

**Raw EEG → Epoching → Filtering → Feature Extraction / CSP → Machine Learning → Subject-wise Evaluation**

Two main approaches were investigated:

1. **Band-power features + machine learning**
2. **Common Spatial Pattern (CSP) + SVM**

Three frequency ranges were also compared:

* 8–30 Hz
* 8–13 Hz (mu/alpha range)
* 13–30 Hz (beta range)

---

## Preprocessing

EEG recordings were divided into **4-second epochs** corresponding to the motor-imagery events.

For the main frequency-based analysis, EEG signals were filtered between **8 and 30 Hz**.

Additional experiments investigated narrower frequency ranges:

* **8–13 Hz** — mu/alpha range
* **13–30 Hz** — beta range

---

## Feature Extraction

### Band-Power Features

Power spectral density was estimated using **Welch's method**.

For the main baseline, power features were extracted from the 8–30 Hz range.

Two frequency-band features were calculated for each of the 64 EEG channels, resulting in:

**128 features per trial**

These features were evaluated using Logistic Regression and SVM classifiers.

### Common Spatial Pattern

Common Spatial Pattern (CSP) was used to extract spatial features that emphasize differences between the two motor-imagery classes.

CSP was fitted separately within each training fold to avoid information leakage from the test subjects.

---

## Machine Learning

The following classifiers were evaluated:

* Logistic Regression
* Support Vector Machine (SVM) with an RBF kernel

For the main CSP experiments:

```text
CSP components: 4
SVM kernel: RBF
C: 10
Gamma: 0.001
```

Feature scaling for Logistic Regression was performed using `StandardScaler`, fitted only on the training data within each cross-validation fold.

---

## Cross-Validation

A **5-fold GroupKFold** strategy was used for subject-independent evaluation.

The participant ID was used as the grouping variable, ensuring that data from the same participant did not appear in both the training and test sets within a fold.

Each fold contains approximately:

* 8 subjects for training
* 2 subjects for testing
* 360 training trials
* 90 testing trials

This setup provides an evaluation of how the models perform on participants that were not included in training.

---

## Results

### Band Power + Logistic Regression

| Metric    |     Score |
| --------- | --------: |
| Accuracy  | **56.2%** |
| Precision |     60.9% |
| Recall    |     47.3% |
| F1-score  |     47.9% |

### Band Power + SVM

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 53.6% |
| Precision | 46.4% |
| Recall    | 18.5% |
| F1-score  | 25.6% |

### CSP + SVM

The number of CSP components was investigated using 4, 6, and 8 components.

| CSP Components | Accuracy | Precision | Recall | F1-score |
| -------------: | -------: | --------: | -----: | -------: |
|              4 |    53.3% |     54.7% |  66.6% |    54.7% |
|              6 |    52.0% |     45.7% |  62.1% |    49.5% |
|              8 |    50.4% |     44.7% |  48.7% |    43.5% |

### CSP + SVM with Hyperparameter Tuning

CSP with four components was evaluated using nested GroupKFold cross-validation while tuning the SVM hyperparameters.

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 55.6% |
| Precision | 47.0% |
| Recall    | 40.0% |
| F1-score  | 41.7% |

### Frequency Band Comparison

| Frequency Band     |  Accuracy | Precision |    Recall |  F1-score |
| ------------------ | --------: | --------: | --------: | --------: |
| 8–30 Hz            |     51.1% |     43.8% |     31.6% |     26.1% |
| 8–13 Hz (mu/alpha) | **53.6%** |     52.7% | **54.4%** | **51.7%** |
| 13–30 Hz (beta)    |     53.1% | **60.9%** |     46.4% |     41.0% |

---

## Discussion

The experiments demonstrate that classification performance varies depending on the feature representation, classifier, and frequency range.

The band-power features combined with Logistic Regression achieved a mean accuracy of **56.2%** under subject-wise cross-validation.

Among the CSP-based frequency-band experiments, the **8–13 Hz** range produced the highest mean F1-score.

The results also showed substantial variation between cross-validation folds. This highlights the challenge of generalizing EEG-based motor-imagery classification across different participants.

An important observation from the experiments is that a more complex feature extraction method does not necessarily result in better classification performance under subject-independent evaluation.

---

## Limitations

* Only 10 participants were included in the current analysis.
* EEG signals can vary considerably between individuals.
* Only runs 4, 8, and 12 were used.
* The observed classification performance was relatively close to chance level for binary classification.
* The analysis was performed on a single publicly available dataset and a limited subset of its recordings.

Therefore, the current results should be considered an **exploratory analysis** rather than a highly accurate general-purpose EEG classifier.

---

## Future Work

Possible extensions include:

* Increasing the number of participants
* Evaluating additional EEG frequency ranges
* Testing additional spatial-filtering methods
* Exploring temporal and spectral feature combinations
* Comparing additional machine-learning models
* Investigating subject-specific versus subject-independent models
* Evaluating additional EEG datasets
* Exploring deep-learning approaches for EEG classification

---

## Project Structure

```text
EEG-Motor-Imagery-Classification/
│
├── 01_single_subject.py
├── 02_download_subjects.py
├── 03_create_dataset.py
├── 04_csp_svm.py
├── 05_frequency_band_comparison.py
│
├── README.md
├── .gitignore
└── results.txt
```

Generated EEG data files such as `.npy` files are excluded from version control.

---

## Technologies

* Python
* MNE-Python
* NumPy
* SciPy
* scikit-learn
* EEG signal processing
* Machine learning
* Common Spatial Pattern (CSP)
* Subject-wise cross-validation

---

## How to Run

Install the required Python packages:

```bash
pip install numpy mne scikit-learn scipy
```

Download the selected EEG recordings:

```bash
python 02_download_subjects.py
```

Create the dataset:

```bash
python 03_create_dataset.py
```

Run the CSP experiments:

```bash
python 04_csp_svm.py
```

Compare frequency bands:

```bash
python 05_frequency_band_comparison.py
```

---

## Dataset Source

PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB):

https://physionet.org/content/eegmmidb/1.0.0/

The dataset is not redistributed in this repository. It should be obtained directly from PhysioNet.

---

## Project Status

**Exploratory analysis completed**

This project provides an end-to-end pipeline for subject-independent EEG motor-imagery classification, including preprocessing, feature extraction, CSP analysis, machine-learning classification, hyperparameter evaluation, and frequency-band comparison.
