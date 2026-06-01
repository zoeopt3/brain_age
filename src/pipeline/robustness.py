"""Robustness checks: EO vs EC, feature family sensitivity, bootstrap.

All checks use GroupKFold by participant to prevent leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import select_feature_columns


def _build_pipeline(cfg: dict) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy=cfg.get("numeric_imputer", "median")))]
    if cfg.get("scale_numeric", True):
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _build_model(model_name: str, seed: int = 1337):
    if model_name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=500,
            l2_regularization=1.0, min_samples_leaf=5, random_state=seed,
        )
    elif model_name == "ridge":
        from sklearn.linear_model import RidgeCV
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    else:
        from sklearn.linear_model import RidgeCV
        return RidgeCV(alphas=[1.0])


def _cv_evaluate(X, y, groups, model_name, preprocess_cfg, cv_cfg):
    """Run GroupKFold CV and return MAE/R² summary."""
    seed = cv_cfg.get("random_seed", 1337)
    n_splits = min(cv_cfg.get("n_splits", 5), len(np.unique(groups)))
    n_splits = max(3, n_splits)
    gkf = GroupKFold(n_splits=n_splits)

    maes, r2s = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        pp = _build_pipeline(preprocess_cfg)
        est = _build_model(model_name, seed)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_name == "hgb":
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_train)
            X_te = imp.transform(X_test)
        else:
            X_tr = pp.fit_transform(X_train)
            X_te = pp.transform(X_test)

        est.fit(X_tr, y_train)
        preds = est.predict(X_te)
        maes.append(mean_absolute_error(y_test, preds))
        r2s.append(r2_score(y_test, preds))

    return {
        "mae_mean": round(float(np.mean(maes)), 3),
        "mae_std": round(float(np.std(maes)), 3),
        "r2_mean": round(float(np.mean(r2s)), 3),
        "r2_std": round(float(np.std(r2s)), 3),
        "n_folds": len(maes),
    }


def load_best_model_type(model_results_path: str) -> str:
    """Read model_results.json and return best non-baseline model name."""
    try:
        with open(model_results_path) as f:
            data = json.load(f)
        models = data.get("models", {})
        best_name, best_mae = "ridge", float("inf")
        for name, res in models.items():
            if name == "baseline_mean":
                continue
            mae = res.get("mae_mean", float("inf"))
            if mae < best_mae:
                best_mae = mae
                best_name = name
        return best_name
    except Exception:
        return "ridge"


# ── EO vs EC ───────────────────────────────────────────────────────────

def run_eo_vs_ec(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    preprocess_cfg: dict,
    cv_cfg: dict,
    target_col: str = "age",
    group_col: str = "sub",
    logger=None,
) -> dict[str, Any]:
    """Train separate models on EO and EC recordings and compare."""
    results: dict[str, Any] = {}

    if "condition" not in df.columns:
        return {"error": "no condition column"}

    for cond in ["EO", "EC"]:
        sub = df[df["condition"] == cond].copy()
        n_groups = sub[group_col].nunique()

        if len(sub) < 10 or n_groups < 3:
            results[cond] = {"error": f"too few rows ({len(sub)}) or groups ({n_groups})"}
            if logger:
                logger.warning("  %s: too few data (%d rows, %d groups)", cond, len(sub), n_groups)
            continue

        X = sub[feature_cols].values
        y = sub[target_col].values
        groups = sub[group_col].values

        res = _cv_evaluate(X, y, groups, model_name, preprocess_cfg, cv_cfg)
        res["n_rows"] = len(sub)
        res["n_participants"] = int(n_groups)
        results[cond] = res

        if logger:
            logger.info("  %s: MAE=%.2f ± %.2f  R²=%.3f  (%d rows, %d participants)",
                        cond, res["mae_mean"], res["mae_std"], res["r2_mean"],
                        len(sub), n_groups)

    # Combined (for reference)
    X_all = df[feature_cols].values
    y_all = df[target_col].values
    g_all = df[group_col].values
    res_all = _cv_evaluate(X_all, y_all, g_all, model_name, preprocess_cfg, cv_cfg)
    res_all["n_rows"] = len(df)
    res_all["n_participants"] = int(df[group_col].nunique())
    results["combined"] = res_all

    if logger:
        logger.info("  Combined: MAE=%.2f ± %.2f  R²=%.3f  (%d rows)",
                    res_all["mae_mean"], res_all["mae_std"], res_all["r2_mean"], len(df))

    return results


# ── Feature family sensitivity ─────────────────────────────────────────

def run_feature_family_sensitivity(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    family_cfg: dict,
    preprocess_cfg: dict,
    cv_cfg: dict,
    target_col: str = "age",
    group_col: str = "sub",
    logger=None,
) -> dict[str, Any]:
    """Compare model performance across feature subsets."""
    periodic_kw = family_cfg.get("periodic_keywords", ["_bp_", "paf", "iaf"])
    aperiodic_kw = family_cfg.get("aperiodic_keywords", ["ap_exponent", "ap_offset"])

    periodic_cols = [c for c in feature_cols
                     if any(k in c.lower() for k in periodic_kw)]
    aperiodic_cols = [c for c in feature_cols
                      if any(k in c.lower() for k in aperiodic_kw)]
    other_cols = [c for c in feature_cols
                  if c not in periodic_cols and c not in aperiodic_cols]

    experiments = {
        "all_features": feature_cols,
        "periodic_only": periodic_cols,
        "aperiodic_only": aperiodic_cols,
        "periodic_plus_aperiodic": periodic_cols + aperiodic_cols,
        "other_only": other_cols,
    }

    y = df[target_col].values
    groups = df[group_col].values
    results: dict[str, Any] = {}

    for exp_name, cols in experiments.items():
        if not cols:
            results[exp_name] = {"error": "no_features", "n_features": 0}
            if logger:
                logger.warning("  %s: no features — skipping", exp_name)
            continue

        X = df[cols].values
        res = _cv_evaluate(X, y, groups, model_name, preprocess_cfg, cv_cfg)
        res["n_features"] = len(cols)
        res["features"] = cols
        results[exp_name] = res

        if logger:
            logger.info("  %-25s  MAE=%.2f ± %.2f  (%d features)",
                        exp_name, res["mae_mean"], res["mae_std"], len(cols))

    return results
