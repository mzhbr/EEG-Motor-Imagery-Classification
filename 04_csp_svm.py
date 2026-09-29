import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# 1. Load saved data
# ============================================================

X_filtered = np.load("X_filtered.npy")
y = np.load("y.npy")
groups = np.load("groups.npy")

print("X_filtered shape:", X_filtered.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)


# ============================================================
# 2. Outer Cross-Validation
# ============================================================

outer_cv = GroupKFold(n_splits=5)

outer_accuracies = []
outer_precisions = []
outer_recalls = []
outer_f1s = []


# ============================================================
# 3. Hyperparameter values
# ============================================================

C_values = [0.1, 1, 10, 100]
gamma_values = ["scale", 0.001, 0.01, 0.1]

n_components = 4


# ============================================================
# 4. Outer CV
# ============================================================

for outer_fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X_filtered, y, groups), 1
):

    print("\n" + "=" * 60)
    print(f"OUTER FOLD {outer_fold}")
    print("=" * 60)

    X_train = X_filtered[train_idx]
    X_test = X_filtered[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    groups_train = groups[train_idx]

    print("Training samples:", len(train_idx))
    print("Test samples:", len(test_idx))

    print("Training subjects:", np.unique(groups_train))
    print("Test subjects:", np.unique(groups[test_idx]))


    # ========================================================
    # 5. Inner CV for hyperparameter tuning
    # ========================================================

    inner_cv = GroupKFold(n_splits=4)

    best_score = -1
    best_C = None
    best_gamma = None


    for C in C_values:

        for gamma in gamma_values:

            inner_scores = []


            # ------------------------------------------------
            # Inner folds
            # ------------------------------------------------

            for inner_train_idx, inner_val_idx in inner_cv.split(
                X_train,
                y_train,
                groups_train
            ):

                X_inner_train = X_train[inner_train_idx]
                X_inner_val = X_train[inner_val_idx]

                y_inner_train = y_train[inner_train_idx]
                y_inner_val = y_train[inner_val_idx]


                # ------------------------------------------------
                # Fit CSP ONLY on inner training data
                # ------------------------------------------------

                csp_inner = CSP(
                    n_components=n_components,
                    reg=None,
                    log=True,
                    norm_trace=False
                )

                X_inner_train_csp = csp_inner.fit_transform(
                    X_inner_train,
                    y_inner_train
                )

                X_inner_val_csp = csp_inner.transform(
                    X_inner_val
                )


                # ------------------------------------------------
                # Train SVM
                # ------------------------------------------------

                model = SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma
                )

                model.fit(
                    X_inner_train_csp,
                    y_inner_train
                )


                # ------------------------------------------------
                # Validation prediction
                # ------------------------------------------------

                y_val_pred = model.predict(
                    X_inner_val_csp
                )

                score = accuracy_score(
                    y_inner_val,
                    y_val_pred
                )

                inner_scores.append(score)


            # ------------------------------------------------
            # Mean inner CV score
            # ------------------------------------------------

            mean_inner_score = np.mean(inner_scores)

            print(
                f"C={C}, gamma={gamma} "
                f"-> Inner Accuracy={mean_inner_score:.3f}"
            )


            # ------------------------------------------------
            # Keep best hyperparameters
            # ------------------------------------------------

            if mean_inner_score > best_score:

                best_score = mean_inner_score
                best_C = C
                best_gamma = gamma


    # ========================================================
    # 6. Show selected hyperparameters
    # ========================================================

    print("\nBest hyperparameters:")
    print("C:", best_C)
    print("gamma:", best_gamma)
    print("Inner CV Accuracy:", round(best_score, 3))


    # ========================================================
    # 7. Fit CSP on the FULL outer training set
    # ========================================================

    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False
    )

    X_train_csp = csp.fit_transform(
        X_train,
        y_train
    )

    X_test_csp = csp.transform(
        X_test
    )


    print("\nCSP train shape:", X_train_csp.shape)
    print("CSP test shape:", X_test_csp.shape)


    # ========================================================
    # 8. Train final SVM using selected parameters
    # ========================================================

    model = SVC(
        kernel="rbf",
        C=best_C,
        gamma=best_gamma
    )

    model.fit(
        X_train_csp,
        y_train
    )


    # ========================================================
    # 9. Test on completely unseen subjects
    # ========================================================

    y_pred = model.predict(
        X_test_csp
    )


    # ========================================================
    # 10. Calculate metrics
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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


    # ========================================================
    # 11. Store results
    # ========================================================

    outer_accuracies.append(accuracy)
    outer_precisions.append(precision)
    outer_recalls.append(recall)
    outer_f1s.append(f1)


    # ========================================================
    # 12. Print fold results
    # ========================================================

    print("\nOuter Fold Results:")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")


# ============================================================
# 13. Final Results
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Mean Accuracy : {np.mean(outer_accuracies):.3f}"
)

print(
    f"Mean Precision: {np.mean(outer_precisions):.3f}"
)

print(
    f"Mean Recall   : {np.mean(outer_recalls):.3f}"
)

print(
    f"Mean F1-score : {np.mean(outer_f1s):.3f}"
)


# ============================================================
# 14. Fold-by-fold results
# ============================================================

print("\nFold-by-fold Accuracy:")

for i, score in enumerate(outer_accuracies, 1):
    print(f"Fold {i}: {score:.3f}")