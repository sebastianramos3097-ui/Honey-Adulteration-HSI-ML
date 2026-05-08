#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 (Sebastian Project) — Boosting Models for Multiclass Classification
Task: Predict concentration_class (multiclass) from 128 hyperspectral bands.

Models:
1) HistGradientBoostingClassifier (scikit-learn)
2) XGBoost (xgboost.XGBClassifier) — optional, runs only if xgboost is installed

Validation:
- Acquisition-based split (train: acquisitions 1–5; test: acquisition 6)
- Hyperparameter tuning via RandomizedSearchCV on TRAIN ONLY
- Scoring: Macro F1-score (robust to class imbalance)

Important methodological note:
- The dataset authors already z-score normalised spectra (mean≈0, SD≈1).
- Therefore, we DO NOT apply any additional scaling or transformation.
- Any "scaling checks" (if computed) are diagnostic only.

Outputs:
- Excel workbook with CV results, best params, test metrics, and classification reports
- Confusion matrix plots (PNG)
- Full text log

Author: (generated for Sebastian MSc thesis pipeline)
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder


# =========================
# Configuration
# =========================

# TODO: Update this path on your machine.
DATA_PATH = r"C:\Sebastian\Pythonsebastian\adulteration_dataset_26_08_2021.xlsx"

# Output folder (created next to this script's working directory by default).
RUN_FOLDER = Path("Step3_Boosting_Multiclass")

RANDOM_STATE = 42
N_SPLITS_CV = 5

# RandomizedSearchCV iterations (increase if you have a faster workstation)
HGB_N_ITER = 30
XGB_N_ITER = 40

# Set to True to also save per-class metrics as a separate table
SAVE_CLASS_REPORT_TABLE = True


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
    logger = logging.getLogger("stage3_boosting_multiclass")
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
    Returns: (acquisition_col, target_col, feature_cols)
    """
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    # Acquisition column
    acq_candidates = ["acquisition", "acq", "acq_id"]
    acquisition_col = None
    for cand in acq_candidates:
        if cand in cols_lower:
            acquisition_col = cols_lower[cand]
            break
    if acquisition_col is None:
        raise ValueError("Could not find the Acquisition column. Expected a column like 'Acquisition'.")

    # Target column (multiclass)
    target_candidates = ["concentration_class", "concentration class", "concentrationclass", "target"]
    target_col = None
    for cand in target_candidates:
        if cand in cols_lower:
            target_col = cols_lower[cand]
            break
    if target_col is None:
        # Try fuzzy find
        for c in cols:
            if "concentration" in c.lower() and "class" in c.lower():
                target_col = c
                break
    if target_col is None:
        raise ValueError("Could not find the target column concentration_class.")

    # Exclude common metadata from features
    exclude_lower = set([
        acquisition_col.lower(),
        target_col.lower(),
        "brand",
        "class",
        "concentration",
        "concentration (%)",
        "concentration_percent",
        "concentrationpercentage",
        "sample",
        "id",
    ])

    feature_cols = [c for c in cols if c.lower() not in exclude_lower]

    # If feature columns are more than 128, keep only numeric columns
    if len(feature_cols) > 140:
        numeric_cols = []
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(c)
        feature_cols = numeric_cols

    if len(feature_cols) < 50:
        raise ValueError(
            f"Too few feature columns detected ({len(feature_cols)}). "
            f"Please verify the dataset structure and update 'exclude_lower' if needed."
        )

    return acquisition_col, target_col, feature_cols


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    # Annotate counts
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f"{val}", ha="center", va="center",
                     color="white" if val > thresh else "black", fontsize=9)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def classification_report_table(y_true, y_pred, labels: List[str]) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    df_rep = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class"})
    return df_rep


def safe_int(x):
    try:
        if pd.isna(x):
            return x
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)) and float(x).is_integer():
            return int(x)
        return x
    except Exception:
        return x


# =========================
# Main
# =========================

def main() -> None:
    folders = setup_folders(RUN_FOLDER)
    logger = setup_logger(folders["logs"] / "run_log.txt")

    logger.info("=== Stage 3: Boosting models for multiclass classification ===")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Run folder: {folders['run'].resolve()}")

    # Load data
    t0 = time.time()
    df = pd.read_excel(DATA_PATH)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)[:12]} ... ({len(df.columns)} total)")

    acquisition_col, target_col, feature_cols = detect_columns(df)
    logger.info(f"Detected acquisition column: {acquisition_col}")
    logger.info(f"Detected target column: {target_col}")
    logger.info(f"Detected number of feature columns: {len(feature_cols)}")

    # Split by acquisition
    train_mask = df[acquisition_col].isin([1, 2, 3, 4, 5])
    test_mask = df[acquisition_col] == 6

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    if df_train.empty or df_test.empty:
        raise ValueError("Train or test split is empty. Please verify acquisition values and split logic.")

    # Prepare X and y
    X_train = df_train[feature_cols].astype(float).values
    X_test = df_test[feature_cols].astype(float).values

    y_train_raw = df_train[target_col].apply(safe_int).values
    y_test_raw = df_test[target_col].apply(safe_int).values

    # Encode labels if needed (keeps a stable order for plots/reports)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    class_labels = [str(c) for c in le.classes_]

    logger.info(f"Classes detected (ordered): {class_labels}")

    # Class distribution tables
    train_counts = pd.Series(y_train_raw).value_counts().sort_index()
    test_counts = pd.Series(y_test_raw).value_counts().sort_index()
    dist_df = pd.DataFrame({"train_count": train_counts, "test_count": test_counts}).fillna(0).astype(int)
    dist_df.index.name = "concentration_class"

    # Sample weights to mitigate moderate imbalance (used for both models)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # CV scheme (training only)
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    # =========================
    # Model 1: HistGradientBoostingClassifier
    # =========================
    logger.info("---- Training: HistGradientBoostingClassifier (HGB) ----")
    hgb = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )

    hgb_param_distributions = {
        "learning_rate": np.linspace(0.03, 0.2, 8),
        "max_iter": [200, 300, 500, 800],
        "max_depth": [None, 3, 5, 7, 9],
        "min_samples_leaf": [10, 20, 30, 50, 80],
        "l2_regularization": np.logspace(-6, -1, 6),
        "max_bins": [128, 196, 255],
    }

    hgb_search = RandomizedSearchCV(
        estimator=hgb,
        param_distributions=hgb_param_distributions,
        n_iter=HGB_N_ITER,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        random_state=RANDOM_STATE,
        verbose=1,
        return_train_score=True,
    )

    hgb_search.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info(f"HGB best params: {hgb_search.best_params_}")
    logger.info(f"HGB best CV macro-F1: {hgb_search.best_score_:.6f}")

    # Evaluate on external test set
    hgb_best = hgb_search.best_estimator_
    y_pred_hgb = hgb_best.predict(X_test)

    hgb_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_hgb),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_hgb),
        "macro_f1": f1_score(y_test, y_pred_hgb, average="macro"),
    }
    logger.info(f"HGB external test metrics: {json.dumps(hgb_metrics, indent=2)}")

    cm_hgb = confusion_matrix(y_test, y_pred_hgb)
    fig_hgb = folders["figures"] / "confusion_matrix_HGB_external_test.png"
    plot_confusion_matrix(cm_hgb, class_labels, "Confusion Matrix — HGB (External Test, Acquisition 6)", fig_hgb)

    hgb_report_df = classification_report_table(y_test, y_pred_hgb, class_labels)

    # =========================
    # Model 2: XGBoost (optional)
    # =========================
    xgb_available = False
    xgb_results = {}
    xgb_cv_df = None
    xgb_report_df = None
    cm_xgb = None
    fig_xgb = None

    try:
        import xgboost as xgb  # type: ignore
        xgb_available = True
    except Exception as e:
        logger.warning("XGBoost is not installed in this environment. Skipping XGBClassifier.")
        logger.warning(f"Import error: {repr(e)}")

    if xgb_available:
        logger.info("---- Training: XGBClassifier (XGBoost) ----")
        num_class = len(class_labels)

        # Conservative, thesis-friendly defaults (avoid extreme complexity)
        xgb_clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_class,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        xgb_param_distributions = {
            "n_estimators": [300, 500, 800, 1200],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 2, 5, 10],
            "gamma": [0, 0.5, 1, 2],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        }

        xgb_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=xgb_param_distributions,
            n_iter=XGB_N_ITER,
            scoring="f1_macro",
            n_jobs=-1,
            cv=cv,
            random_state=RANDOM_STATE,
            verbose=1,
            return_train_score=True,
        )

        xgb_search.fit(X_train, y_train, sample_weight=sample_weight)
        logger.info(f"XGB best params: {xgb_search.best_params_}")
        logger.info(f"XGB best CV macro-F1: {xgb_search.best_score_:.6f}")

        xgb_best = xgb_search.best_estimator_
        y_pred_xgb = xgb_best.predict(X_test)

        xgb_results = {
            "accuracy": accuracy_score(y_test, y_pred_xgb),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_xgb),
            "macro_f1": f1_score(y_test, y_pred_xgb, average="macro"),
        }
        logger.info(f"XGB external test metrics: {json.dumps(xgb_results, indent=2)}")

        cm_xgb = confusion_matrix(y_test, y_pred_xgb)
        fig_xgb = folders["figures"] / "confusion_matrix_XGB_external_test.png"
        plot_confusion_matrix(cm_xgb, class_labels, "Confusion Matrix — XGB (External Test, Acquisition 6)", fig_xgb)

        xgb_report_df = classification_report_table(y_test, y_pred_xgb, class_labels)

        # Save CV results
        xgb_cv_df = pd.DataFrame(xgb_search.cv_results_)

    # =========================
    # Save outputs to Excel
    # =========================
    out_xlsx = folders["outputs"] / "Stage3_Boosting_Multiclass_Results.xlsx"
    logger.info(f"Saving Excel outputs to: {out_xlsx.resolve()}")

    hgb_cv_df = pd.DataFrame(hgb_search.cv_results_)
    hgb_best_params_df = pd.DataFrame([hgb_search.best_params_])
    hgb_test_metrics_df = pd.DataFrame([hgb_metrics])

    # Confusion matrices as tables
    cm_hgb_df = pd.DataFrame(cm_hgb, index=class_labels, columns=class_labels)
    cm_hgb_df.index.name = "true"
    cm_hgb_df.columns.name = "pred"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        dist_df.to_excel(writer, sheet_name="train_test_class_counts")

        hgb_cv_df.to_excel(writer, sheet_name="HGB_cv_results", index=False)
        hgb_best_params_df.to_excel(writer, sheet_name="HGB_best_params", index=False)
        hgb_test_metrics_df.to_excel(writer, sheet_name="HGB_test_metrics", index=False)
        cm_hgb_df.to_excel(writer, sheet_name="HGB_confusion_matrix")
        if SAVE_CLASS_REPORT_TABLE:
            hgb_report_df.to_excel(writer, sheet_name="HGB_class_report", index=False)

        if xgb_available and xgb_cv_df is not None:
            xgb_cv_df.to_excel(writer, sheet_name="XGB_cv_results", index=False)
            pd.DataFrame([xgb_search.best_params_]).to_excel(writer, sheet_name="XGB_best_params", index=False)
            pd.DataFrame([xgb_results]).to_excel(writer, sheet_name="XGB_test_metrics", index=False)

            if cm_xgb is not None:
                cm_xgb_df = pd.DataFrame(cm_xgb, index=class_labels, columns=class_labels)
                cm_xgb_df.index.name = "true"
                cm_xgb_df.columns.name = "pred"
                cm_xgb_df.to_excel(writer, sheet_name="XGB_confusion_matrix")

            if SAVE_CLASS_REPORT_TABLE and xgb_report_df is not None:
                xgb_report_df.to_excel(writer, sheet_name="XGB_class_report", index=False)

    # Save a small JSON summary for quick reading
    summary = {
        "data_path": DATA_PATH,
        "train_split": "Acquisition in {1,2,3,4,5}",
        "test_split": "Acquisition == 6",
        "n_train": int(df_train.shape[0]),
        "n_test": int(df_test.shape[0]),
        "n_features": int(len(feature_cols)),
        "classes": class_labels,
        "HGB": {
            "best_params": hgb_search.best_params_,
            "best_cv_macro_f1": float(hgb_search.best_score_),
            "external_test_metrics": hgb_metrics,
            "confusion_matrix_figure": str(fig_hgb),
        },
        "XGB": {
            "available": bool(xgb_available),
            "external_test_metrics": xgb_results if xgb_available else None,
            "confusion_matrix_figure": str(fig_xgb) if fig_xgb else None,
        },
        "excel_output": str(out_xlsx),
    }

    out_json = folders["outputs"] / "Stage3_Boosting_Multiclass_Summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"Done. Total runtime: {elapsed:.1f} seconds")
    logger.info("Key outputs:")
    logger.info(f"- Excel: {out_xlsx.resolve()}")
    logger.info(f"- Figures: {folders['figures'].resolve()}")
    logger.info(f"- Log: {folders['logs'].resolve() / 'run_log.txt'}")
    logger.info(f"- Summary JSON: {out_json.resolve()}")


if __name__ == "__main__":
    main()
