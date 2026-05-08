#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 (Sebastian Project) — SVM (RBF) for Binary Classification
Task: Predict binary authenticity:
    authentic (pure honey, 0% sugar) vs adulterated (>0% sugar)

Key methodological choices (aligned with our project decisions):
- Acquisition-based split:
    Train = acquisitions 1–5
    Test  = acquisition 6 (independent external test set)
- No additional scaling:
    Spectral features in this dataset are already z-score normalised by the dataset authors
    (mean ≈ 0, standard deviation ≈ 1). Any scaling checks are diagnostic only.
- Class imbalance handling:
    SVC uses class_weight="balanced" to reduce bias toward the majority class.

Outputs (reproducible thesis-ready package):
- Excel workbook (openpyxl) with:
    * class counts (train/test)
    * CV tuning results
    * best hyperparameters
    * external test metrics
    * confusion matrix (numeric)
    * classification report table
- Figures (PNG):
    * confusion matrix
    * ROC curve (optional; requires probability estimates)
    * Precision–Recall curve (optional; requires probability estimates)
- Text log file

IMPORTANT: Update DATA_PATH to match your local file location.
"""

from __future__ import annotations

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# =========================
# Configuration
# =========================

# TODO: Update this path on your machine.
DATA_PATH = r"C:\Sebastian\Pythonsebastian\adulteration_dataset_26_08_2021.xlsx"

RUN_FOLDER = Path("Step3_SVM_Binary")

RANDOM_STATE = 42
N_SPLITS_CV = 5

# If True, SVC will compute probability estimates (slower but enables ROC/PR curves).
ENABLE_PROBABILITIES = True


# =========================
# Helpers
# =========================

def setup_folders(run_folder: Path) -> Dict[str, Path]:
    outputs = run_folder / "outputs"
    figures = run_folder / "figures"
    logs = run_folder / "logs"
    for p in (run_folder, outputs, figures, logs):
        p.mkdir(parents=True, exist_ok=True)
    return {"run": run_folder, "outputs": outputs, "figures": figures, "logs": logs}


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("stage3_svm_binary")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, List[str]]:
    """
    Detect key columns robustly across possible naming conventions.
    Returns: (acquisition_col, concentration_class_col, feature_cols)
    """
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    # Acquisition
    acquisition_col = None
    for cand in ["acquisition", "acq", "acq_id"]:
        if cand in cols_lower:
            acquisition_col = cols_lower[cand]
            break
    if acquisition_col is None:
        raise ValueError("Could not find Acquisition column (expected 'Acquisition' or similar).")

    # concentration_class
    concentration_class_col = None
    for cand in ["concentration_class", "concentration class", "concentrationclass"]:
        if cand in cols_lower:
            concentration_class_col = cols_lower[cand]
            break
    if concentration_class_col is None:
        for c in cols:
            if "concentration" in c.lower() and "class" in c.lower():
                concentration_class_col = c
                break
    if concentration_class_col is None:
        raise ValueError("Could not find concentration_class column.")

    # Features = numeric columns excluding metadata
    exclude_lower = {
        acquisition_col.lower(),
        concentration_class_col.lower(),
        "brand",
        "class",
        "concentration",
        "id",
        "sample",
    }

    feature_cols = [c for c in cols if c.lower() not in exclude_lower]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    if len(feature_cols) < 50:
        raise ValueError(
            f"Too few numeric feature columns detected ({len(feature_cols)}). "
            f"Please verify dataset columns."
        )

    return acquisition_col, concentration_class_col, feature_cols


def make_binary_target(conc_class_series: pd.Series) -> pd.Series:
    """
    Binary target:
    - authentic = 0 (pure honey, 0% sugar)
    - adulterated = 1 (>0% sugar)
    """
    def to_float_safe(x):
        try:
            return float(x)
        except Exception:
            s = str(x).replace("%", "").strip()
            return float(s)

    vals = conc_class_series.apply(to_float_safe)
    return (vals > 0).astype(int)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=30, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f"{val}", ha="center", va="center",
                     color="white" if val > thresh else "black", fontsize=10)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC Curve (AUC = {roc_auc:.3f}) — External Test")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return float(roc_auc)


def plot_pr(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(recall, precision)
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (AP = {ap:.3f}) — External Test")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return float(ap)


def classification_report_table(y_true, y_pred, target_names: List[str]) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class"})


# =========================
# Main
# =========================

def main() -> None:
    folders = setup_folders(RUN_FOLDER)
    logger = setup_logger(folders["logs"] / "run_log.txt")

    logger.info("=== Stage 3: SVM (RBF) — BINARY classification (authentic vs adulterated) ===")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Run folder: {folders['run'].resolve()}")
    logger.info(f"ENABLE_PROBABILITIES = {ENABLE_PROBABILITIES}")

    t0 = time.time()

    df = pd.read_excel(DATA_PATH)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    acquisition_col, concentration_class_col, feature_cols = detect_columns(df)
    logger.info(f"Detected acquisition column: {acquisition_col}")
    logger.info(f"Detected concentration_class column: {concentration_class_col}")
    logger.info(f"Number of spectral features used: {len(feature_cols)}")

    # Acquisition-based split
    train_mask = df[acquisition_col].isin([1, 2, 3, 4, 5])
    test_mask = df[acquisition_col] == 6

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    if df_train.empty or df_test.empty:
        raise ValueError("Train or test split is empty. Verify acquisition values and split logic.")

    # Binary targets
    y_train = make_binary_target(df_train[concentration_class_col])
    y_test = make_binary_target(df_test[concentration_class_col])

    # Features
    X_train = df_train[feature_cols].astype(float).values
    X_test = df_test[feature_cols].astype(float).values

    label_names = ["Authentic (0%)", "Adulterated (>0%)"]

    # Class distribution
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()
    dist_df = pd.DataFrame({"train_count": train_counts, "test_count": test_counts}).fillna(0).astype(int)
    dist_df.index = label_names
    dist_df.index.name = "binary_class"

    logger.info("Train/Test class counts:\n" + dist_df.to_string())

    # CV and model
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=ENABLE_PROBABILITIES,
        random_state=RANDOM_STATE,
    )

    # Defensible grid (kept intentionally small for reproducibility and to avoid over-tuning)
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1],
    }

    search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    logger.info("Starting GridSearchCV on TRAIN ONLY (Acq 1–5).")
    search.fit(X_train, y_train.values)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV macro-F1: {search.best_score_:.6f}")

    best_model = search.best_estimator_

    # External test evaluation
    y_pred = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # recall for adulterated
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan  # recall for authentic
    precision_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

    test_metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "sensitivity_recall_adulterated": float(sensitivity),
        "specificity_authentic": float(specificity),
        "precision_adulterated": float(precision_pos),
    }

    # Optional probability-based curves
    if ENABLE_PROBABILITIES:
        y_score = best_model.predict_proba(X_test)[:, 1]
        roc_auc = plot_roc(y_test.values, y_score, folders["figures"] / "ROC_SVM_binary_external_test.png")
        ap = plot_pr(y_test.values, y_score, folders["figures"] / "PR_SVM_binary_external_test.png")
        test_metrics["roc_auc"] = float(roc_auc)
        test_metrics["average_precision"] = float(ap)

    logger.info("External test metrics:\n" + json.dumps(test_metrics, indent=2))

    # Save confusion matrix figure
    plot_confusion_matrix(
        cm,
        labels=label_names,
        title="Confusion Matrix — SVM (Binary, External Test, Acq 6)",
        out_path=folders["figures"] / "confusion_matrix_SVM_binary_external_test.png",
    )

    # Prepare tables for Excel
    cv_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    best_params_df = pd.DataFrame([search.best_params_])
    test_metrics_df = pd.DataFrame([test_metrics])

    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.index.name = "true"
    cm_df.columns.name = "pred"

    class_report_df = classification_report_table(y_test.values, y_pred, label_names)

    # Save outputs to Excel
    out_xlsx = folders["outputs"] / "Stage3_SVM_Binary_Results.xlsx"
    logger.info(f"Saving Excel outputs to: {out_xlsx.resolve()}")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        dist_df.to_excel(writer, sheet_name="train_test_class_counts")
        cv_df.to_excel(writer, sheet_name="GridSearchCV_results", index=False)
        best_params_df.to_excel(writer, sheet_name="best_params", index=False)
        test_metrics_df.to_excel(writer, sheet_name="external_test_metrics", index=False)
        cm_df.to_excel(writer, sheet_name="confusion_matrix")
        class_report_df.to_excel(writer, sheet_name="classification_report", index=False)

    # Save quick JSON summary
    summary = {
        "data_path": DATA_PATH,
        "task": "binary_authentic_vs_adulterated",
        "train_split": "Acquisition in {1,2,3,4,5}",
        "test_split": "Acquisition == 6",
        "n_train": int(df_train.shape[0]),
        "n_test": int(df_test.shape[0]),
        "n_features": int(len(feature_cols)),
        "svm": {
            "kernel": "rbf",
            "class_weight": "balanced",
            "probability": bool(ENABLE_PROBABILITIES),
            "best_params": search.best_params_,
            "best_cv_macro_f1": float(search.best_score_),
            "external_test_metrics": test_metrics,
        },
        "outputs": {
            "excel": str(out_xlsx),
            "figures_folder": str(folders["figures"]),
            "log_file": str(folders["logs"] / "run_log.txt"),
        },
    }

    out_json = folders["outputs"] / "Stage3_SVM_Binary_Summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"Done. Total runtime: {elapsed:.1f} seconds")
    logger.info(f"Summary JSON: {out_json.resolve()}")


if __name__ == "__main__":
    main()
