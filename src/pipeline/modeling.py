"""Age-prediction modeling with leakage-safe cross-validation.

Provides utilities for building sklearn pipelines, running GroupKFold CV,
computing metrics, and serializing model artifacts.  All preprocessing
happens inside the CV loop to prevent data leakage.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Dummy baseline ─────────────────────────────────────────────────────

class MeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the training-set mean age for every sample."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


# ── Column selection ───────────────────────────────────────────────────

def select_feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Return numeric columns that are actual features (not metadata).

    Args:
        df: Features DataFrame.
        cfg: The ``preprocessing`` config section.

    Returns:
        Sorted list of feature column names.
    """
    exclude = set(cfg.get("exclude_columns", []))
    feat_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in ("object", "bool", "category"):
            continue
        feat_cols.append(col)
    return sorted(feat_cols)


# ── Pipeline builders ──────────────────────────────────────────────────

def build_preprocess_pipeline(cfg: dict) -> Pipeline:
    """Build an imputer + scaler pipeline (fit inside CV).

    Args:
        cfg: The ``preprocessing`` config section.

    Returns:
        sklearn Pipeline.
    """
    steps = []
    strategy = cfg.get("numeric_imputer", "median")
    steps.append(("imputer", SimpleImputer(strategy=strategy)))
    if cfg.get("scale_numeric", True):
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def build_models(cfg: dict) -> dict[str, Any]:
    """Instantiate all enabled models from config.

    Args:
        cfg: The ``models`` config section.

    Returns:
        Dict of model_name -> estimator.
    """
    models: dict[str, Any] = {}

    if cfg.get("baseline_mean", {}).get("enabled", True):
        models["baseline_mean"] = MeanRegressor()

    ridge_cfg = cfg.get("ridge", {})
    if ridge_cfg.get("enabled", False):
        alphas = ridge_cfg.get("alphas", [1.0])
        models["ridge"] = RidgeCV(alphas=alphas)

    en_cfg = cfg.get("elasticnet", {})
    if en_cfg.get("enabled", False):
        models["elasticnet"] = ElasticNetCV(
            alphas=en_cfg.get("alphas", [0.1, 1.0]),
            l1_ratio=en_cfg.get("l1_ratio", [0.5]),
            max_iter=5000,
        )

    hgb_cfg = cfg.get("hgb", {})
    if hgb_cfg.get("enabled", False):
        params = hgb_cfg.get("params", {})
        models["hgb"] = HistGradientBoostingRegressor(
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            max_iter=params.get("max_iter", 500),
            l2_regularization=params.get("l2_regularization", 1.0),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            random_state=1337,
        )

    return models


# ── Cross-validation ───────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE and R² from arrays."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_group_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    preprocess_cfg: dict,
    models_cfg: dict,
    cv_cfg: dict,
    logger=None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run GroupKFold CV for all enabled models.

    All preprocessing is fit on training folds only (no leakage).

    Args:
        df: Features DataFrame.
        feature_cols: List of feature column names.
        target_col: Target column name.
        group_col: Group column for participant-level splits.
        preprocess_cfg: Preprocessing config section.
        models_cfg: Models config section.
        cv_cfg: CV config section.
        logger: Optional logger.

    Returns:
        (results_dict, cv_predictions_df)
    """
    n_splits = cv_cfg.get("n_splits", 5)
    n_groups = df[group_col].nunique()

    # Auto-reduce splits if too few groups
    if n_groups < n_splits:
        old = n_splits
        n_splits = max(3, n_groups)
        if logger:
            logger.warning("Reduced n_splits from %d to %d (only %d groups)",
                           old, n_splits, n_groups)

    gkf = GroupKFold(n_splits=n_splits)
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df[group_col].values

    models = build_models(models_cfg)
    all_preds: list[dict] = []
    fold_metrics: dict[str, list[dict]] = {name: [] for name in models}

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit preprocessor on training fold only
        pp = build_preprocess_pipeline(preprocess_cfg)
        X_train_pp = pp.fit_transform(X_train)
        X_test_pp = pp.transform(X_test)

        for model_name, estimator in models.items():
            try:
                from sklearn.base import clone
                est = clone(estimator)

                # HGB handles NaN natively, so give it unscaled data
                if model_name == "hgb":
                    imp = SimpleImputer(strategy="median")
                    X_tr = imp.fit_transform(X_train)
                    X_te = imp.transform(X_test)
                    est.fit(X_tr, y_train)
                    preds = est.predict(X_te)
                else:
                    est.fit(X_train_pp, y_train)
                    preds = est.predict(X_test_pp)

                metrics = compute_metrics(y_test, preds)
                fold_metrics[model_name].append({
                    "fold": fold_i, **metrics,
                    "n_train": len(train_idx), "n_test": len(test_idx),
                })

                for j, idx in enumerate(test_idx):
                    row = df.iloc[idx]
                    all_preds.append({
                        "model": model_name,
                        "fold": fold_i,
                        "sub": row.get(group_col, ""),
                        "condition": row.get("condition", ""),
                        "ses": row.get("ses", ""),
                        "task": row.get("task", ""),
                        "y_true": float(y_test[j]),
                        "y_pred": float(preds[j]),
                        "residual": float(preds[j] - y_test[j]),
                    })

                if logger:
                    logger.info("  Fold %d | %-15s | MAE=%.2f  R²=%.3f",
                                fold_i, model_name, metrics["mae"], metrics["r2"])

            except Exception as e:
                if logger:
                    logger.warning("  Fold %d | %-15s | FAILED: %s",
                                   fold_i, model_name, e)
                fold_metrics[model_name].append({
                    "fold": fold_i, "mae": float("nan"), "r2": float("nan"),
                    "error": str(e)[:200],
                })

    # Aggregate
    results: dict[str, Any] = {}
    for model_name, folds in fold_metrics.items():
        maes = [f["mae"] for f in folds if not np.isnan(f.get("mae", float("nan")))]
        r2s = [f["r2"] for f in folds if not np.isnan(f.get("r2", float("nan")))]
        results[model_name] = {
            "folds": folds,
            "mae_mean": float(np.mean(maes)) if maes else float("nan"),
            "mae_std": float(np.std(maes)) if maes else float("nan"),
            "r2_mean": float(np.mean(r2s)) if r2s else float("nan"),
            "r2_std": float(np.std(r2s)) if r2s else float("nan"),
            "n_folds_ok": len(maes),
        }

    cv_pred_df = pd.DataFrame(all_preds)
    return results, cv_pred_df


# ── Final model fitting ───────────────────────────────────────────────

def fit_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    best_model_name: str,
    preprocess_cfg: dict,
    models_cfg: dict,
) -> tuple[Pipeline, Any]:
    """Fit the best model on all data for deployment/inspection.

    Returns:
        (full_pipeline, fitted_estimator)
    """
    X = df[feature_cols].values
    y = df[target_col].values

    pp = build_preprocess_pipeline(preprocess_cfg)
    models = build_models(models_cfg)
    est = models[best_model_name]

    if best_model_name == "hgb":
        imp = SimpleImputer(strategy="median")
        X_pp = imp.fit_transform(X)
        est.fit(X_pp, y)
        full_pipe = Pipeline([("imputer", imp), ("model", est)])
    else:
        X_pp = pp.fit_transform(X)
        est.fit(X_pp, y)
        full_pipe = Pipeline(pp.steps + [("model", est)])

    return full_pipe, est


# ── Results JSON ───────────────────────────────────────────────────────

def build_model_results_json(
    cv_results: dict,
    cfg: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    best_model: str,
) -> dict[str, Any]:
    """Build the model_results.json payload."""
    # Git hash (best effort)
    git_hash = ""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            git_hash = r.stdout.strip()
    except Exception:
        pass

    # Config hash
    config_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "config_hash": config_hash,
        "n_rows": len(df),
        "n_participants": int(df[cfg.get("groups", {}).get("column", "sub")].nunique()),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "best_model": best_model,
        "models": cv_results,
    }


def save_model_artifacts(
    pipeline: Pipeline,
    out_dir: str | Path,
    model_name: str,
) -> str:
    """Save fitted pipeline to disk via joblib."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_pipeline.joblib"
    joblib.dump(pipeline, path)
    return str(path)
