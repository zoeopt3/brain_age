"""Explainability: permutation importance, SHAP, and ablation experiments.

Computes feature importance on held-out CV folds to avoid leakage,
runs targeted ablation experiments (remove PAF, remove aperiodic, etc.),
and optionally computes SHAP values for tree-based models.
"""

from __future__ import annotations

import json
import warnings
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


# ── Model selection ────────────────────────────────────────────────────

def load_best_model_name(model_results_path: str, cfg: dict) -> str:
    """Determine the best model name from model_results.json.

    Args:
        model_results_path: Path to model_results.json.
        cfg: The ``model_selection`` config section.

    Returns:
        Model name string (e.g. "ridge", "hgb").
    """
    fallback = cfg.get("fallback_model_name", "ridge")
    exclude = set(cfg.get("exclude_from_best", ["baseline_mean"]))

    try:
        with open(model_results_path) as f:
            results = json.load(f)
        models = results.get("models", {})
        best_name = None
        best_mae = float("inf")
        for name, res in models.items():
            if name in exclude:
                continue
            mae = res.get("mae_mean", float("inf"))
            if mae < best_mae:
                best_mae = mae
                best_name = name
        return best_name or fallback
    except Exception:
        return fallback


# ── Feature group detection ────────────────────────────────────────────

def build_feature_groups(
    feature_cols: list[str],
    cfg: dict,
) -> dict[str, list[str]]:
    """Classify feature columns into named groups using keyword matching.

    Args:
        feature_cols: All feature column names.
        cfg: The ``feature_groups`` config section.

    Returns:
        Dict of group_name -> list of matching column names.
    """
    groups: dict[str, list[str]] = {}

    for group_key, keywords in cfg.items():
        group_name = group_key.replace("_keywords", "")
        matched = []
        for col in feature_cols:
            col_lower = col.lower()
            if any(kw.lower() in col_lower for kw in keywords):
                matched.append(col)
        groups[group_name] = matched

    # "other" = everything not in any named group
    all_grouped = set()
    for cols in groups.values():
        all_grouped.update(cols)
    groups["other"] = [c for c in feature_cols if c not in all_grouped]

    return groups


def columns_after_dropping(
    feature_cols: list[str],
    groups: dict[str, list[str]],
    drop_group: str | None = None,
    drop_groups: list[str] | None = None,
) -> list[str]:
    """Return feature columns after removing one or more groups."""
    to_drop: set[str] = set()
    if drop_group and drop_group in groups:
        to_drop.update(groups[drop_group])
    if drop_groups:
        for g in drop_groups:
            if g in groups:
                to_drop.update(groups[g])
    return [c for c in feature_cols if c not in to_drop]


# ── Pipeline builder (reuse Stage 4 logic) ─────────────────────────────

def _build_pipeline(cfg: dict) -> Pipeline:
    steps = []
    steps.append(("imputer", SimpleImputer(strategy=cfg.get("numeric_imputer", "median"))))
    if cfg.get("scale_numeric", True):
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _build_model(model_name: str, seed: int = 1337):
    """Rebuild a model estimator by name."""
    if model_name == "ridge":
        from sklearn.linear_model import RidgeCV
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    elif model_name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=500,
            l2_regularization=1.0, min_samples_leaf=5, random_state=seed,
        )
    elif model_name == "elasticnet":
        from sklearn.linear_model import ElasticNetCV
        return ElasticNetCV(alphas=[0.01, 0.1, 1.0, 10.0], l1_ratio=[0.1, 0.5, 0.9], max_iter=5000)
    else:
        from sklearn.linear_model import RidgeCV
        return RidgeCV(alphas=[1.0])


# ── Permutation importance (foldwise) ──────────────────────────────────

def compute_foldwise_permutation_importance(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    preprocess_cfg: dict,
    cv_cfg: dict,
) -> pd.DataFrame:
    """Compute permutation importance on held-out folds.

    For each fold: fit model on train, compute baseline MAE on test,
    permute each feature in test and measure MAE increase.

    Returns:
        DataFrame with columns: feature, importance_mean, importance_std, fold_count.
    """
    n_splits = cv_cfg.get("n_splits", 5)
    seed = cv_cfg.get("random_seed", 1337)
    n_repeats = 10

    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        n_splits = max(3, n_groups)

    gkf = GroupKFold(n_splits=n_splits)
    rng = np.random.RandomState(seed)

    # Collect per-fold importances
    all_importances = []  # list of (n_features,) arrays

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pp = _build_pipeline(preprocess_cfg)
        est = _build_model(model_name, seed)

        if model_name == "hgb":
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_train)
            X_te = imp.transform(X_test)
        else:
            X_tr = pp.fit_transform(X_train)
            X_te = pp.transform(X_test)

        est.fit(X_tr, y_train)
        baseline_mae = mean_absolute_error(y_test, est.predict(X_te))

        fold_imp = np.zeros(X.shape[1])
        for feat_i in range(X.shape[1]):
            deltas = []
            for _ in range(n_repeats):
                X_te_perm = X_te.copy()
                X_te_perm[:, feat_i] = rng.permutation(X_te_perm[:, feat_i])
                perm_mae = mean_absolute_error(y_test, est.predict(X_te_perm))
                deltas.append(perm_mae - baseline_mae)
            fold_imp[feat_i] = np.mean(deltas)
        all_importances.append(fold_imp)

    imp_array = np.array(all_importances)  # (n_folds, n_features)
    imp_mean = imp_array.mean(axis=0)
    imp_std = imp_array.std(axis=0)

    result = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": imp_mean,
        "importance_std": imp_std,
        "fold_count": len(all_importances),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return result


# ── SHAP (optional) ────────────────────────────────────────────────────

def compute_shap_if_available(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    preprocess_cfg: dict,
    cfg: dict,
) -> pd.DataFrame | None:
    """Compute SHAP values if the shap package is installed.

    Returns:
        DataFrame of SHAP values (n_samples × n_features), or None.
    """
    if not cfg.get("enabled", True):
        return None

    try:
        import shap
    except ImportError:
        return None

    max_samples = cfg.get("max_samples", 500)
    bg_samples = cfg.get("background_samples", 100)
    seed = 1337

    pp = _build_pipeline(preprocess_cfg)
    est = _build_model(model_name, seed)

    if model_name == "hgb":
        imp = SimpleImputer(strategy="median")
        X_pp = imp.fit_transform(X)
    else:
        X_pp = pp.fit_transform(X)

    est.fit(X_pp, y)

    n = min(max_samples, len(X_pp))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_pp), n, replace=False)
    X_sample = X_pp[idx]

    try:
        if model_name == "hgb":
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X_sample)
        else:
            bg_n = min(bg_samples, len(X_pp))
            bg_idx = rng.choice(len(X_pp), bg_n, replace=False)
            explainer = shap.KernelExplainer(est.predict, X_pp[bg_idx])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(X_sample, nsamples=100)

        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        return shap_df

    except Exception:
        return None


# ── Ablation experiments ───────────────────────────────────────────────

def run_ablation_experiments(
    model_name: str,
    df: pd.DataFrame,
    all_feature_cols: list[str],
    target_col: str,
    group_col: str,
    feature_groups: dict[str, list[str]],
    experiments: list[dict],
    preprocess_cfg: dict,
    cv_cfg: dict,
    logger=None,
) -> dict[str, Any]:
    """Run ablation experiments: retrain model with feature subsets.

    Each experiment either drops a feature group or keeps only specific groups.

    Returns:
        Dict of experiment_name -> {mae_mean, mae_std, r2_mean, r2_std, n_features, dropped}.
    """
    seed = cv_cfg.get("random_seed", 1337)
    n_splits = cv_cfg.get("n_splits", 5)

    y = df[target_col].values
    groups = df[group_col].values
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        n_splits = max(3, n_groups)

    gkf = GroupKFold(n_splits=n_splits)
    results: dict[str, Any] = {}

    for exp in experiments:
        name = exp["name"]
        drop_group = exp.get("drop_group")
        drop_groups = exp.get("drop_groups")
        keep_groups = exp.get("keep_groups")

        # Determine columns for this experiment
        if keep_groups:
            cols = []
            for g in keep_groups:
                cols.extend(feature_groups.get(g, []))
            # Also keep "other" features that aren't in any named group
            cols.extend(feature_groups.get("other", []))
            cols = sorted(set(cols))
        elif drop_group or drop_groups:
            cols = columns_after_dropping(all_feature_cols, feature_groups, drop_group, drop_groups)
        else:
            cols = list(all_feature_cols)

        if not cols:
            if logger:
                logger.warning("  Ablation '%s': no features remaining — skipping", name)
            results[name] = {"mae_mean": float("nan"), "error": "no_features"}
            continue

        X = df[cols].values
        dropped = sorted(set(all_feature_cols) - set(cols))

        maes, r2s = [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pp = _build_pipeline(preprocess_cfg)
            est = _build_model(model_name, seed)

            try:
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
            except Exception as e:
                if logger:
                    logger.warning("  Ablation '%s' fold failed: %s", name, e)

        results[name] = {
            "mae_mean": float(np.mean(maes)) if maes else float("nan"),
            "mae_std": float(np.std(maes)) if maes else float("nan"),
            "r2_mean": float(np.mean(r2s)) if r2s else float("nan"),
            "r2_std": float(np.std(r2s)) if r2s else float("nan"),
            "n_features": len(cols),
            "n_dropped": len(dropped),
            "dropped_features": dropped,
            "n_folds_ok": len(maes),
        }

        if logger:
            logger.info("  %-30s  MAE=%.2f ± %.2f  (%d features, %d dropped)",
                        name, results[name]["mae_mean"], results[name]["mae_std"],
                        len(cols), len(dropped))

    return results


# ── Narrative notes generator ──────────────────────────────────────────

def generate_explainability_notes(
    importance_df: pd.DataFrame,
    ablation_results: dict,
    feature_groups: dict[str, list[str]],
    model_name: str,
) -> str:
    """Generate a short Markdown narrative about what the model uses.

    Relates top features to EEG developmental biology.
    """
    lines = [
        "# Explainability Notes",
        "",
        f"**Model:** {model_name}",
        "",
        "## Key Findings",
        "",
    ]

    # Top features
    top = importance_df.head(10)
    lines.append("### Top 10 Features by Permutation Importance")
    lines.append("")
    lines.append("| Rank | Feature | Importance (MAE increase) |")
    lines.append("|------|---------|--------------------------|")
    for i, row in top.iterrows():
        lines.append(f"| {i+1} | `{row['feature']}` | {row['importance_mean']:.4f} ± {row['importance_std']:.4f} |")
    lines.append("")

    # Feature group presence in top features
    top_names = set(top["feature"].tolist())
    lines.append("### Feature Group Representation in Top 10")
    lines.append("")
    for gname, gcols in feature_groups.items():
        if gname == "other":
            continue
        in_top = top_names & set(gcols)
        lines.append(f"- **{gname}**: {len(in_top)} of {len(gcols)} features in top 10"
                      + (f" ({', '.join(sorted(in_top))})" if in_top else ""))
    lines.append("")

    # Ablation summary
    if ablation_results:
        lines.append("### Ablation Results")
        lines.append("")
        lines.append("| Experiment | MAE (years) | Features Used | Features Dropped |")
        lines.append("|-----------|------------|---------------|-----------------|")
        for name, res in ablation_results.items():
            mae = f"{res['mae_mean']:.2f} ± {res['mae_std']:.2f}" if not np.isnan(res.get("mae_mean", float("nan"))) else "N/A"
            lines.append(f"| {name} | {mae} | {res.get('n_features', '?')} | {res.get('n_dropped', '?')} |")
        lines.append("")

    # Interpretive notes
    lines.extend([
        "## Interpretation",
        "",
        "- **Aperiodic (1/f) features** capture the spectral slope, which flattens during ",
        "  brain development (childhood → adolescence). A steeper slope (higher exponent) is ",
        "  associated with younger ages, reflecting greater neural noise and immature cortical circuits.",
        "",
        "- **Peak alpha frequency (PAF)** increases with age during development, reflecting ",
        "  thalamocortical circuit maturation. PAF is one of the most robust developmental EEG markers.",
        "",
        "- **Individualized alpha power (IAF)** captures alpha-band energy at each participant's ",
        "  own peak frequency rather than a fixed 8-12 Hz window, reducing inter-individual ",
        "  variability that is unrelated to age.",
        "",
        "- **Regional band powers** may show different developmental trajectories: frontal theta ",
        "  typically decreases with age, while posterior alpha increases. These regional patterns ",
        "  can improve age prediction beyond global summaries.",
        "",
        "## Caveats",
        "",
        "- This analysis is for **research and education only**; it is not a diagnostic tool.",
        "- The current dataset has a narrow age range, limiting the ability to detect developmental trends.",
        "- Feature importance values are sensitive to the specific model, CV split, and dataset.",
        "- Permutation importance can underestimate the importance of correlated features.",
        "- SHAP values (if computed) reflect the fitted model, not causal relationships.",
        "",
        "---",
        "*Generated automatically. Review before drawing conclusions.*",
    ])

    return "\n".join(lines)
