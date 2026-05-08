#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 (Sebastian Project) — XGBoost Regressor for Regression
Task: Predict concentration (continuous % sugar) from 128 hyperspectral bands.

Key methodological choices (aligned with our project decisions):
- Acquisition-based split:
    Train = acquisitions 1–5
    Test  = acquisition 6 (independent external test set)
- No additional scaling:
    Spectral features are already z-score normalised by the dataset authors (mean≈0, SD≈1).
- Leakage prevention:
    All tuning and CV are performed on the TRAIN set only.
- Hyperparameter tuning:
    RandomizedSearchCV on TRAIN ONLY, optimising negative RMSE.
- Fast training:
    XGBoost is configured with tree_method="hist" for speed.

Outputs (reproducible thesis-ready package):
- Excel workbook (openpyxl) with:
    * target stats (train/test)
    * CV tuning results
    * best hyperparameters
    * external test metrics (RMSE, MAE, R²)
    * residual summary
    * external test predictions (observed, predicted, residual)
    * feature importance table (gain)
- Figures (PNG):
    * Predicted vs Observed (External Test)
    * Residual plot (External Test)
    * Feature importance bar plot (top 25; gain)
- Text log file
- JSON summary file

IMPORTANT:
- This script requires 'xgboost' to be installed: pip install xgboost
- Update DATA_PATH to match your local file location.
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

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost
import xgboost as xgb


# =========================
# Configuration
# =========================

# TODO: Update this path on your machine.
DATA_PATH = r"C:\Sebastian\Pythonsebastian\adulteration_dataset_26_08_2021.xlsx"

RUN_FOLDER = Path("Step3_XGBoost_Regression")

RANDOM_STATE = 42
N_SPLITS_CV = 5

# RandomizedSearchCV iterations (laptop-friendly default)
N_ITER = 40


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
    logger = logging.getLogger("stage3_xgb_regression")
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
    Returns: (acquisition_col, concentration_col, feature_cols)
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

    # Concentration (continuous)
    concentration_col = None
    for cand in ["concentration", "concentration (%)", "concentration_percent"]:
        if cand in cols_lower:
            concentration_col = cols_lower[cand]
            break
    if concentration_col is None:
        # Fallback: search for 'concentration' but not 'class'
        for c in cols:
            cl = c.lower()
            if "concentration" in cl and "class" not in cl:
                concentration_col = c
                break
    if concentration_col is None:
        raise ValueError("Could not find concentration column (continuous target).")

    # Features = numeric columns excluding metadata
    exclude_lower = {
        acquisition_col.lower(),
        concentration_col.lower(),
        "brand",
        "class",
        "concentration_class",
        "concentration class",
        "concentrationclass",
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

    return acquisition_col, concentration_col, feature_cols


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def plot_predicted_vs_observed(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(y_true, y_pred)
    min_v = float(min(y_true.min(), y_pred.min()))
    max_v = float(max(y_true.max(), y_pred.max()))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Observed concentration (%)")
    plt.ylabel("Predicted concentration (%)")
    plt.title("XGBoost Regression — Predicted vs Observed (External Test, Acq 6)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted concentration (%)")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.title("XGBoost Regression — Residuals (External Test, Acq 6)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance_gain(booster: xgb.Booster, feature_names: List[str], out_path: Path, top_n: int = 25) -> pd.DataFrame:
    """
    Plot XGBoost feature importance by 'gain' and return a DataFrame.
    """
    score = booster.get_score(importance_type="gain")
    imp = []
    for k, v in score.items():
        idx = int(k.replace("f", ""))
        name = feature_names[idx] if idx < len(feature_names) else k
        imp.append((name, v))
    df_imp = pd.DataFrame(imp, columns=["feature", "gain"]).sort_values("gain", ascending=False)

    df_top = df_imp.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(df_top["feature"], df_top["gain"])
    plt.xlabel("Gain importance")
    plt.ylabel("Spectral band")
    plt.title(f"XGBoost Feature Importance (Top {top_n}) — Gain")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return df_imp


# =========================
# Main
# =========================

def main() -> None:
    folders = setup_folders(RUN_FOLDER)
    logger = setup_logger(folders["logs"] / "run_log.txt")

    logger.info("=== Stage 3: XGBoost — Regression (concentration) ===")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Run folder: {folders['run'].resolve()}")
    logger.info(f"N_ITER (RandomizedSearchCV) = {N_ITER}")

    t0 = time.time()

    df = pd.read_excel(DATA_PATH)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    acquisition_col, concentration_col, feature_cols = detect_columns(df)
    logger.info(f"Detected acquisition column: {acquisition_col}")
    logger.info(f"Detected concentration column: {concentration_col}")
    logger.info(f"Number of spectral features used: {len(feature_cols)}")

    # Acquisition-based split
    train_mask = df[acquisition_col].isin([1, 2, 3, 4, 5])
    test_mask = df[acquisition_col] == 6

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    if df_train.empty or df_test.empty:
        raise ValueError("Train or test split is empty. Verify acquisition values and split logic.")

    X_train = df_train[feature_cols].astype(float).values
    X_test = df_test[feature_cols].astype(float).values

    y_train = df_train[concentration_col].astype(float).values
    y_test = df_test[concentration_col].astype(float).values

    # Target stats (useful to report / sanity check)
    target_stats = pd.DataFrame({
        "split": ["train", "test"],
        "n": [len(y_train), len(y_test)],
        "mean": [float(np.mean(y_train)), float(np.mean(y_test))],
        "sd": [float(np.std(y_train, ddof=1)), float(np.std(y_test, ddof=1))],
        "min": [float(np.min(y_train)), float(np.min(y_test))],
        "max": [float(np.max(y_train)), float(np.max(y_test))],
    })
    logger.info("Target stats (concentration %):\n" + target_stats.to_string(index=False))

    # CV scheme (training only)
    cv = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    xgb_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Defensible hyperparameter space (avoid extremes; laptop-friendly)
    param_distributions = {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 5, 10],
        "gamma": [0, 0.5, 1, 2],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    }

    search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )

    logger.info("Starting RandomizedSearchCV on TRAIN ONLY (Acq 1–5).")
    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV neg-RMSE: {search.best_score_:.6f} (higher is better; negative RMSE)")

    best_model = search.best_estimator_

    # External test evaluation
    y_pred_test = best_model.predict(X_test)

    test_metrics = {
        "RMSE": rmse(y_test, y_pred_test),
        "MAE": float(mean_absolute_error(y_test, y_pred_test)),
        "R2": float(r2_score(y_test, y_pred_test)),
    }

    logger.info("External test metrics:\n" + json.dumps(test_metrics, indent=2))

    residuals = y_test - y_pred_test
    residual_summary = pd.DataFrame({
        "metric": ["mean_residual", "sd_residual", "min_residual", "max_residual"],
        "value": [
            float(np.mean(residuals)),
            float(np.std(residuals, ddof=1)),
            float(np.min(residuals)),
            float(np.max(residuals)),
        ],
    })

    # Figures
    plot_predicted_vs_observed(
        y_true=y_test,
        y_pred=y_pred_test,
        out_path=folders["figures"] / "predicted_vs_observed_XGB_external_test.png",
    )
    plot_residuals(
        y_true=y_test,
        y_pred=y_pred_test,
        out_path=folders["figures"] / "residuals_XGB_external_test.png",
    )

    # Feature importance (gain)
    booster = best_model.get_booster()
    df_imp = plot_feature_importance_gain(
        booster=booster,
        feature_names=feature_cols,
        out_path=folders["figures"] / "feature_importance_XGB_gain_top25.png",
        top_n=25,
    )

    # Save to Excel
    out_xlsx = folders["outputs"] / "Stage3_XGBoost_Regression_Results.xlsx"
    logger.info(f"Saving Excel outputs to: {out_xlsx.resolve()}")

    cv_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    best_params_df = pd.DataFrame([search.best_params_])
    test_metrics_df = pd.DataFrame([test_metrics])

    preds_df = pd.DataFrame({
        "observed_concentration": y_test,
        "predicted_concentration": y_pred_test,
        "residual": residuals,
    })

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        target_stats.to_excel(writer, sheet_name="target_stats", index=False)
        cv_df.to_excel(writer, sheet_name="RandomizedSearchCV_results", index=False)
        best_params_df.to_excel(writer, sheet_name="best_params", index=False)
        test_metrics_df.to_excel(writer, sheet_name="external_test_metrics", index=False)
        residual_summary.to_excel(writer, sheet_name="residual_summary", index=False)
        preds_df.to_excel(writer, sheet_name="external_test_predictions", index=False)
        df_imp.to_excel(writer, sheet_name="feature_importance_gain", index=False)

    # Save JSON summary
    summary = {
        "data_path": DATA_PATH,
        "task": "regression_concentration",
        "train_split": "Acquisition in {1,2,3,4,5}",
        "test_split": "Acquisition == 6",
        "n_train": int(df_train.shape[0]),
        "n_test": int(df_test.shape[0]),
        "n_features": int(len(feature_cols)),
        "xgboost": {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "best_params": search.best_params_,
            "best_cv_neg_rmse": float(search.best_score_),
            "external_test_metrics": test_metrics,
        },
        "outputs": {
            "excel": str(out_xlsx),
            "figures_folder": str(folders["figures"]),
            "log_file": str(folders["logs"] / "run_log.txt"),
        },
    }

    out_json = folders["outputs"] / "Stage3_XGBoost_Regression_Summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"Done. Total runtime: {elapsed:.1f} seconds")
    logger.info(f"Summary JSON: {out_json.resolve()}")


if __name__ == "__main__":
    main()
