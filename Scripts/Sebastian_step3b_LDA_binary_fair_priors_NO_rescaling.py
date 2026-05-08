#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sebastian Project — Stage 3 (Binary) — "Fair" LDA with class priors (NO re-standardisation)
-----------------------------------------------------------------------------------------
Purpose
  Build a more "fair" binary classifier (pure vs adulterated) by controlling the class priors
  in Linear Discriminant Analysis (LDA), while respecting the dataset documentation that the
  spectra are already z-score normalised (mean=0, SD=1).

Key point (important!)
  The dataset authors already applied z-score normalisation to each spectrum. Therefore, we do
  NOT apply StandardScaler again in this script (to avoid redundant preprocessing and to stay
  fully consistent with the dataset description).

Task definition
  Binary target (y):
      0 = pure honey (Concentration_Class == 0)
      1 = adulterated honey (Concentration_Class > 0)

Validation strategy (must be preserved)
  Train: Acquisition in {1,2,3,4,5}
  Test : Acquisition == 6   (external independent test set)

Outputs
  - Excel report with CV and test metrics (openpyxl)
  - Confusion matrix plot (PNG)
  - Text log with key results and a quick check of feature scaling
"""

import sys
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = r"C:\Sebastian\Pythonsebastian\adulteration_dataset_26_08_2021.xlsx"

# Output folder (change if needed)
BASE_DIR = Path(r"C:\Sebastian\Stage3_fair_binary_LDA_NO_rescaling")
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"

RANDOM_STATE = 42
N_SPLITS = 5

# Prior strategy:
#   "empirical" : use class frequencies from training data (sklearn default)
#   "equal"     : force priors to [0.5, 0.5]  -> usually improves recall_pure
#   "custom"    : manually set priors in CUSTOM_PRIORS below
PRIOR_STRATEGY = "equal"
CUSTOM_PRIORS = (0.5, 0.5)  # (prior_pure, prior_adulterated) if PRIOR_STRATEGY="custom"

META_COLS = ["Brand", "Class", "Acquisition", "Concentration", "Concentration_Class"]


# -----------------------------
# Helper functions
# -----------------------------
def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def create_binary_target(concentration_class: pd.Series) -> np.ndarray:
    # 0 = pure (0%), 1 = adulterated (>0%)
    y = pd.to_numeric(concentration_class, errors="coerce").fillna(0.0)
    return (y > 0).astype(int).values


def prepare_train_test(df: pd.DataFrame):
    df = df.copy()
    df["Acquisition"] = df["Acquisition"].astype(int)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values
    y = create_binary_target(df["Concentration_Class"])

    train_mask = df["Acquisition"].isin([1, 2, 3, 4, 5])
    test_mask = df["Acquisition"] == 6

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    return X_train, X_test, y_train, y_test, feature_cols


def resolve_priors():
    # LDA expects priors in class order [0, 1] = [pure, adulterated]
    if PRIOR_STRATEGY.lower() == "empirical":
        return None  # sklearn will infer from training data
    if PRIOR_STRATEGY.lower() == "equal":
        return np.array([0.5, 0.5], dtype=float)
    if PRIOR_STRATEGY.lower() == "custom":
        p0, p1 = CUSTOM_PRIORS
        priors = np.array([p0, p1], dtype=float)
        if not np.isclose(priors.sum(), 1.0):
            raise ValueError("CUSTOM_PRIORS must sum to 1.0 (e.g., 0.4, 0.6).")
        return priors
    raise ValueError("PRIOR_STRATEGY must be one of: empirical, equal, custom")


def compute_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Positive class = 1 (adulterated)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_adulterated": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_adulterated": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1_adulterated": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_pure": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_pure": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1_pure": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def quick_scaling_check(X: np.ndarray, n_bands_to_check: int = 10) -> dict:
    """
    Quick diagnostic: check mean and std for a subset of spectral variables.
    For z-score normalised spectra, we expect mean ~ 0 and std ~ 1 (approximately).
    """
    rng = np.random.default_rng(42)
    n_features = X.shape[1]
    idx = rng.choice(n_features, size=min(n_bands_to_check, n_features), replace=False)

    means = np.mean(X[:, idx], axis=0)
    stds = np.std(X[:, idx], axis=0, ddof=0)

    return {
        "checked_feature_indices": idx.tolist(),
        "mean_min": float(np.min(means)),
        "mean_max": float(np.max(means)),
        "std_min": float(np.min(stds)),
        "std_max": float(np.max(stds)),
        "mean_avg_abs": float(np.mean(np.abs(means))),
        "std_avg": float(np.mean(stds)),
    }


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=15)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(int(cm[i, j]), "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()

    log_lines = []
    log_lines.append(f"[INFO] Run timestamp: {datetime.now().isoformat(timespec='seconds')}")
    log_lines.append(f"[INFO] Python: {sys.version.split()[0]} | Platform: {platform.platform()}")
    log_lines.append(f"[INFO] DATA_PATH: {DATA_PATH}")
    log_lines.append(f"[INFO] PRIOR_STRATEGY: {PRIOR_STRATEGY} | CUSTOM_PRIORS: {CUSTOM_PRIORS}")
    log_lines.append("[INFO] NOTE: No StandardScaler is used (dataset is already z-score normalised).")

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(df)

    log_lines.append(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    log_lines.append(f"[INFO] Class counts in train: pure={int((y_train==0).sum())}, adulterated={int((y_train==1).sum())}")
    log_lines.append(f"[INFO] Class counts in test : pure={int((y_test==0).sum())}, adulterated={int((y_test==1).sum())}")

    scaling_diag = quick_scaling_check(X_train, n_bands_to_check=12)
    log_lines.append(f"[CHECK] Z-score diagnostic (train subset): {scaling_diag}")

    priors = resolve_priors()
    log_lines.append(f"[INFO] LDA priors used: {priors if priors is not None else 'empirical (from training distribution)'}")

    model = LinearDiscriminantAnalysis(priors=priors)

    # Cross-validation on training data only
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    cv_rows = []
    fold_id = 1
    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)

        m = compute_metrics_binary(y_val, y_val_pred)
        m["fold"] = fold_id
        cv_rows.append(m)

        log_lines.append(
            f"[CV] Fold {fold_id}: acc={m['accuracy']:.4f}, bal_acc={m['balanced_accuracy']:.4f}, "
            f"recall_pure={m['recall_pure']:.4f}, recall_adulterated={m['recall_adulterated']:.4f}"
        )
        fold_id += 1

    cv_df = pd.DataFrame(cv_rows)
    cv_summary = cv_df.drop(columns=["fold"]).agg(["mean", "std"]).T.reset_index()
    cv_summary.columns = ["metric", "mean", "std"]

    # Fit on full training and evaluate on external test (Acquisition 6)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    test_metrics = compute_metrics_binary(y_test, y_test_pred)
    test_df = pd.DataFrame({"metric": list(test_metrics.keys()), "value": list(test_metrics.values())})

    labels = [0, 1]
    label_names = ["pure", "adulterated"]
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{n}" for n in label_names],
        columns=[f"pred_{n}" for n in label_names]
    )

    report_dict = classification_report(
        y_test, y_test_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # Save outputs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = OUT_DIR / f"results_LDA_binary_fair_NO_rescaling_{PRIOR_STRATEGY}_{stamp}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        cv_df.to_excel(writer, sheet_name="CV_folds", index=False)
        cv_summary.to_excel(writer, sheet_name="CV_summary", index=False)
        test_df.to_excel(writer, sheet_name="Test_summary", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_matrix")
        report_df.to_excel(writer, sheet_name="Classification_report")

    fig_path = FIG_DIR / f"confusion_matrix_LDA_fair_NO_rescaling_{PRIOR_STRATEGY}_{stamp}.png"
    plot_confusion_matrix(
        cm,
        labels=label_names,
        title=f"LDA (priors={PRIOR_STRATEGY}, no rescaling) — Test set (Acq 6)",
        out_path=fig_path
    )

    log_path = LOG_DIR / f"run_log_LDA_fair_NO_rescaling_{PRIOR_STRATEGY}_{stamp}.txt"
    log_lines.append("\n[TEST] Summary metrics:")
    for k, v in test_metrics.items():
        log_lines.append(f"  - {k}: {v:.6f}")

    log_lines.append(f"\n[OUTPUT] Excel : {xlsx_path}")
    log_lines.append(f"[OUTPUT] Figure: {fig_path}")
    log_lines.append(f"[OUTPUT] Log   : {log_path}")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print("[DONE] Fair LDA (no rescaling) completed.")
    print(f"Excel saved to: {xlsx_path}")
    print(f"Confusion matrix plot saved to: {fig_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
