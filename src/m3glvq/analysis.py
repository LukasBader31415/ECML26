# functions/m3glvq_analysis.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pathlib import Path
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)
from sklearn.cluster import AgglomerativeClustering


# ============================================================================
# Basic helpers
# ============================================================================

def _slice_distance_like(D, tr_idx, va_idx):
    """
    Slice a distance structure into train-train and val-train parts.
    Supports:
      - single ndarray (n x n)
      - list of ndarrays [(n x n), ...]
    """
    if isinstance(D, list):
        D_tr = [M[np.ix_(tr_idx, tr_idx)] for M in D]
        D_va = [M[np.ix_(va_idx, tr_idx)] for M in D]
        return D_tr, D_va

    D_arr = np.asarray(D)
    D_tr = D_arr[np.ix_(tr_idx, tr_idx)]
    D_va = D_arr[np.ix_(va_idx, tr_idx)]
    return D_tr, D_va


def _get_vweights(model) -> Optional[np.ndarray]:
    """
    Read final matrix weights from model.
    """
    if hasattr(model, "_vWeights") and getattr(model, "_vWeights") is not None:
        return np.asarray(model._vWeights, dtype=float).ravel()

    if hasattr(model, "get_vweight_path"):
        try:
            vpath = model.get_vweight_path()
        except Exception:
            vpath = None

        if vpath is not None:
            vpath = np.asarray(vpath, dtype=float)
            if vpath.ndim == 2 and vpath.shape[0] > 0:
                return vpath[-1].ravel()
            if vpath.ndim == 1 and vpath.size > 0:
                return vpath.ravel()

    return None


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector onto probability simplex: w >= 0, sum(w)=1
    """
    v = np.asarray(v, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("Cannot project empty vector onto simplex.")

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def _extract_weight_columns(
    fold_df: pd.DataFrame,
    weight_prefix: str = "vweight_",
) -> List[str]:
    cols = [c for c in fold_df.columns if c.startswith(weight_prefix)]
    if not cols:
        raise ValueError(
            f"No weight columns found with prefix '{weight_prefix}'. "
            "Expected columns like 'vweight_0', 'vweight_1', ..."
        )
    return sorted(cols, key=lambda x: int(x.replace(weight_prefix, "")))


def _pairwise_l1_distances(W: np.ndarray) -> np.ndarray:
    """
    Pairwise L1 distance matrix for rows in W.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i, :] = np.abs(W[i] - W).sum(axis=1)
    return D


def _pairwise_l2_distances(W: np.ndarray) -> np.ndarray:
    """
    Pairwise L2 distance matrix for rows in W.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        diff = W[i] - W
        D[i, :] = np.sqrt(np.sum(diff * diff, axis=1))
    return D


def _medoid_index(W: np.ndarray, metric: str = "l1") -> int:
    """
    Return index of medoid row in W.
    """
    if W.shape[0] == 1:
        return 0

    if metric == "l1":
        D = _pairwise_l1_distances(W)
    elif metric == "l2":
        D = _pairwise_l2_distances(W)
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'l1' or 'l2'.")

    return int(np.argmin(D.sum(axis=1)))


def _normalize_weight_vector(v: np.ndarray) -> np.ndarray:
    """
    Safe normalize to simplex.
    """
    return _project_simplex(np.asarray(v, dtype=float).ravel())


def _dominant_weight_label(w: np.ndarray) -> str:
    """
    Human-readable dominant pattern label.
    """
    w = np.asarray(w, dtype=float).ravel()
    j = int(np.argmax(w))
    return f"matrix_{j}"


# ============================================================================
# Result container
# ============================================================================

@dataclass
class M3GLVQAnalysisResult:
    run_name: str
    fold_df: pd.DataFrame
    oof_df: pd.DataFrame
    summary: Dict[str, float]
    tracked_paths: Optional[Dict[int, np.ndarray]] = None


# ============================================================================
# OOF summary
# ============================================================================

def summarize_oof_predictions(oof_df: pd.DataFrame) -> Dict[str, float]:
    y_true = np.asarray(oof_df["y_true"]).astype(int)
    y_pred = np.asarray(oof_df["y_pred"]).astype(int)

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


# ============================================================================
# CV analysis
# ============================================================================

def run_m3glvq_cv_analysis(
    D,
    y,
    model_cls,
    model_params,
    n_splits: int = 5,
    random_state: int = 42,
    meta_df: Optional[pd.DataFrame] = None,
    run_name: str = "m3glvq_run",
    store_paths: bool = True,
    verbose: bool = True,
) -> M3GLVQAnalysisResult:
    """
    Run stratified K-fold CV for M3GLVQ-like model.

    Returns:
      - fold_df : one row per fold
      - oof_df  : one row per sample
      - summary : OOF metrics
    """
    y = np.asarray(y).astype(int).ravel()
    if y.size == 0:
        raise ValueError("y must not be empty.")

    if meta_df is not None and len(meta_df) != len(y):
        raise ValueError("meta_df must have same number of rows as y.")

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_pred = np.full_like(y, fill_value=-1)
    oof_fold = np.full_like(y, fill_value=-1)

    fold_rows: List[Dict[str, Any]] = []
    tracked_paths: Dict[int, np.ndarray] = {}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(y)), y), start=0):
        if verbose:
            print(f"[Fold {fold_idx + 1}/{n_splits}]")

        D_tr, D_va = _slice_distance_like(D, tr_idx, va_idx)
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        model = model_cls(**model_params)
        model.fit(D_tr, y_tr)
        y_pred = model.predict(D_va)

        oof_pred[va_idx] = y_pred
        oof_fold[va_idx] = fold_idx

        row: Dict[str, Any] = {
            "run_name": run_name,
            "fold": fold_idx,
            "train_size": int(len(tr_idx)),
            "val_size": int(len(va_idx)),
            "balanced_accuracy": float(balanced_accuracy_score(y_va, y_pred)),
            "recall": float(recall_score(y_va, y_pred, zero_division=0)),
            "precision": float(precision_score(y_va, y_pred, zero_division=0)),
            "f1": float(f1_score(y_va, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_va, y_pred)),
            "random_state": int(random_state),
        }

        # add model params to fold row
        for k, v in model_params.items():
            row[k] = v

        # final weights
        vweights = _get_vweights(model)
        if vweights is not None:
            for j, wj in enumerate(vweights):
                row[f"vweight_{j}"] = float(wj)

        # prototypes
        if hasattr(model, "_w") and getattr(model, "_w") is not None:
            row["prototype_indices"] = list(np.asarray(model._w, dtype=int).ravel())

        if hasattr(model, "_y") and getattr(model, "_y") is not None:
            proto_labels = np.asarray(model._y, dtype=int).ravel()
            proto_idx = np.asarray(model._w, dtype=int).ravel() if hasattr(model, "_w") else None
            if proto_idx is not None and proto_idx.shape[0] == proto_labels.shape[0]:
                row["prototypes_label_0"] = proto_idx[proto_labels == 0].tolist()
                row["prototypes_label_1"] = proto_idx[proto_labels == 1].tolist()

        # paths
        if store_paths and hasattr(model, "get_vweight_path"):
            try:
                vpath = model.get_vweight_path()
            except Exception:
                vpath = None
            if vpath is not None:
                tracked_paths[fold_idx] = np.asarray(vpath, dtype=float)

        fold_rows.append(row)

    if np.any(oof_pred < 0):
        raise RuntimeError("Some OOF predictions were not filled.")

    fold_df = pd.DataFrame(fold_rows)

    oof_df = pd.DataFrame({
        "sample_idx": np.arange(len(y)),
        "fold": oof_fold,
        "y_true": y,
        "y_pred": oof_pred,
        "correct": (oof_pred == y).astype(int),
        "run_name": run_name,
    })

    if meta_df is not None:
        oof_df = pd.concat([oof_df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    summary = summarize_oof_predictions(oof_df)

    return M3GLVQAnalysisResult(
        run_name=run_name,
        fold_df=fold_df,
        oof_df=oof_df,
        summary=summary,
        tracked_paths=tracked_paths if store_paths else None,
    )


# ============================================================================
# Selection of good runs
# ============================================================================

def select_good_runs(
    fold_df: pd.DataFrame,
    score_col: str = "balanced_accuracy",
    selection_strategy: str = "within_delta",
    delta: float = 0.02,
    top_n: Optional[int] = None,
    top_quantile: Optional[float] = None,
) -> pd.DataFrame:
    """
    Select good runs/folds according to score.

    selection_strategy:
      - "within_delta" : keep rows with score >= best_score - delta
      - "top_n"        : keep best top_n rows
      - "top_quantile" : keep rows >= quantile threshold
    """
    if score_col not in fold_df.columns:
        raise KeyError(f"score_col '{score_col}' not found in fold_df.")

    df = fold_df.copy()
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("fold_df is empty.")

    if selection_strategy == "within_delta":
        best_score = float(df[score_col].max())
        out = df[df[score_col] >= best_score - delta].copy()
        return out.reset_index(drop=True)

    if selection_strategy == "top_n":
        if top_n is None:
            raise ValueError("For selection_strategy='top_n', top_n must be set.")
        if top_n < 1:
            raise ValueError("top_n must be >= 1.")
        return df.head(int(top_n)).reset_index(drop=True)

    if selection_strategy == "top_quantile":
        if top_quantile is None:
            raise ValueError("For selection_strategy='top_quantile', top_quantile must be set.")
        if not (0.0 < top_quantile <= 1.0):
            raise ValueError("top_quantile must be in (0, 1].")
        thr = float(df[score_col].quantile(top_quantile))
        out = df[df[score_col] >= thr].copy()
        return out.reset_index(drop=True)

    raise ValueError(
        "Unsupported selection_strategy. "
        "Use 'within_delta', 'top_n', or 'top_quantile'."
    )


# ============================================================================
# Clustering of weight profiles
# ============================================================================

def cluster_weight_profiles(
    fold_df: pd.DataFrame,
    weight_prefix: str = "vweight_",
    max_clusters: int = 4,
    distance_threshold: Optional[float] = None,
    metric: str = "l1",
    linkage: str = "average",
    min_cluster_size: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster rows by final weight vectors.

    Parameters
    ----------
    max_clusters : int
        Maximum number of clusters if distance_threshold is None.
    distance_threshold : float or None
        If given, clustering will stop based on this threshold.
        Good with metric='l1' because weights live on simplex.
    metric : str
        'l1' or 'l2'
    linkage : str
        passed to AgglomerativeClustering
    min_cluster_size : int
        clusters smaller than this can later be marked as noise-like groups
        in summary, but they are still kept in labeled data.

    Returns
    -------
    labeled_runs : DataFrame
        Original selected rows + cluster labels
    cluster_summary : DataFrame
        Summary table per cluster
    """
    if len(fold_df) == 0:
        raise ValueError("fold_df is empty.")

    weight_cols = _extract_weight_columns(fold_df, weight_prefix=weight_prefix)
    W = fold_df[weight_cols].to_numpy(dtype=float)

    if W.shape[0] == 1:
        labeled_runs = fold_df.copy()
        labeled_runs["cluster"] = 0
        cluster_summary = pd.DataFrame([{
            "cluster": 0,
            "n_runs": 1,
            "cluster_valid": bool(min_cluster_size <= 1),
            "dominant_matrix": _dominant_weight_label(W[0]),
            "ref_weight_mean": _normalize_weight_vector(W[0]),
            "ref_weight_median": _normalize_weight_vector(W[0]),
            "ref_weight_medoid": _normalize_weight_vector(W[0]),
        }])
        return labeled_runs, cluster_summary

    if metric == "l1":
        metric_name = "manhattan"
    elif metric == "l2":
        metric_name = "euclidean"
    else:
        raise ValueError("metric must be 'l1' or 'l2'.")

    # sklearn compatibility:
    # use either n_clusters or distance_threshold, not both
    if distance_threshold is not None:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(distance_threshold),
            metric=metric_name,
            linkage=linkage,
        )
    else:
        n_clusters = min(int(max_clusters), W.shape[0])
        if n_clusters < 1:
            raise ValueError("max_clusters must be >= 1.")
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric_name,
            linkage=linkage,
        )

    labels = clustering.fit_predict(W)

    labeled_runs = fold_df.copy().reset_index(drop=True)
    labeled_runs["cluster"] = labels

    cluster_rows: List[Dict[str, Any]] = []

    for cid, g in labeled_runs.groupby("cluster"):
        idx = g.index.to_numpy()
        Wc = W[idx]

        mean_w = _normalize_weight_vector(Wc.mean(axis=0))
        median_w = _normalize_weight_vector(np.median(Wc, axis=0))
        medoid_local = _medoid_index(Wc, metric=metric)
        medoid_w = _normalize_weight_vector(Wc[medoid_local])

        row = {
            "cluster": int(cid),
            "n_runs": int(len(g)),
            "cluster_valid": bool(len(g) >= min_cluster_size),
            "dominant_matrix": _dominant_weight_label(mean_w),
            "ref_weight_mean": mean_w,
            "ref_weight_median": median_w,
            "ref_weight_medoid": medoid_w,
        }

        # cluster internal spread
        if len(g) > 1:
            if metric == "l1":
                D = _pairwise_l1_distances(Wc)
            else:
                D = _pairwise_l2_distances(Wc)
            row["intra_cluster_distance_mean"] = float(D[np.triu_indices_from(D, k=1)].mean())
            row["intra_cluster_distance_max"] = float(D[np.triu_indices_from(D, k=1)].max())
        else:
            row["intra_cluster_distance_mean"] = 0.0
            row["intra_cluster_distance_max"] = 0.0

        # average metrics if available
        for col in ["balanced_accuracy", "recall", "precision", "f1", "accuracy"]:
            if col in g.columns:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=0))

        cluster_rows.append(row)

    cluster_summary = pd.DataFrame(cluster_rows).sort_values(
        by="balanced_accuracy_mean" if "balanced_accuracy_mean" in pd.DataFrame(cluster_rows).columns else "cluster",
        ascending=False,
    ).reset_index(drop=True)

    return labeled_runs, cluster_summary


# ============================================================================
# High-level weight profile builder
# ============================================================================

def build_reference_weight_profiles(
    fold_df: pd.DataFrame,
    score_col: str = "balanced_accuracy",
    selection_strategy: str = "within_delta",
    delta: float = 0.02,
    top_n: Optional[int] = None,
    top_quantile: Optional[float] = None,
    weight_prefix: str = "vweight_",
    max_clusters: int = 4,
    distance_threshold: Optional[float] = None,
    metric: str = "l1",
    linkage: str = "average",
    min_cluster_size: int = 1,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Full pipeline for Phase 1 weight regime extraction:

    1. Select good runs
    2. Cluster good runs by end weights
    3. Build cluster summary with representative reference profiles
    """
    good_runs = select_good_runs(
        fold_df=fold_df,
        score_col=score_col,
        selection_strategy=selection_strategy,
        delta=delta,
        top_n=top_n,
        top_quantile=top_quantile,
    )

    if verbose:
        print(f"[Selection] kept {len(good_runs)} of {len(fold_df)} rows using '{selection_strategy}'.")

    labeled_runs, cluster_summary = cluster_weight_profiles(
        fold_df=good_runs,
        weight_prefix=weight_prefix,
        max_clusters=max_clusters,
        distance_threshold=distance_threshold,
        metric=metric,
        linkage=linkage,
        min_cluster_size=min_cluster_size,
    )

    if verbose:
        n_clusters = labeled_runs["cluster"].nunique()
        print(f"[Clustering] found {n_clusters} cluster(s).")

    return {
        "good_runs": good_runs.reset_index(drop=True),
        "labeled_runs": labeled_runs.reset_index(drop=True),
        "cluster_summary": cluster_summary.reset_index(drop=True),
    }


# ============================================================================
# Fixed-weight evaluation suite
# ============================================================================

def run_fixed_weight_profile_suite(
    D,
    y,
    model_cls,
    base_model_params: Dict[str, Any],
    weight_profiles: pd.DataFrame,
    profile_kind: str = "mean",
    n_splits: int = 5,
    random_state: int = 42,
    meta_df: Optional[pd.DataFrame] = None,
    run_name_prefix: str = "m3glvq_fixed",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate multiple fixed-weight reference profiles.

    profile_kind:
      - 'mean'   -> use ref_weight_mean
      - 'median' -> use ref_weight_median
      - 'medoid' -> use ref_weight_medoid
    """
    if profile_kind not in {"mean", "median", "medoid"}:
        raise ValueError("profile_kind must be one of: 'mean', 'median', 'medoid'.")

    results: List[M3GLVQAnalysisResult] = []

    for _, row in weight_profiles.iterrows():
        cid = int(row["cluster"])

        if profile_kind == "mean":
            w_ref = row["ref_weight_mean"]
        elif profile_kind == "median":
            w_ref = row["ref_weight_median"]
        else:
            w_ref = row["ref_weight_medoid"]

        params = dict(base_model_params)
        params["v_init"] = _normalize_weight_vector(np.asarray(w_ref, dtype=float))
        params["eta"] = 0.0  # freeze by zero step

        run_name = f"{run_name_prefix}_cluster{cid}_{profile_kind}"

        if verbose:
            print(f"[Fixed Profile] cluster={cid} profile_kind={profile_kind} weights={params['v_init']}")

        res = run_m3glvq_cv_analysis(
            D=D,
            y=y,
            model_cls=model_cls,
            model_params=params,
            n_splits=n_splits,
            random_state=random_state,
            meta_df=meta_df,
            run_name=run_name,
            store_paths=True,
            verbose=verbose,
        )
        results.append(res)

    comparison_df = compare_analysis_runs(results)

    return {
        "results": results,
        "comparison_df": comparison_df,
    }


# ============================================================================
# Comparison
# ============================================================================

def compare_analysis_runs(results: List[M3GLVQAnalysisResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        row = {"run_name": res.run_name}
        row.update(res.summary)

        if hasattr(res, "fold_df") and res.fold_df is not None and len(res.fold_df) > 0:
            for col in ["balanced_accuracy", "recall", "precision", "f1", "accuracy"]:
                if col in res.fold_df.columns:
                    row[f"{col}_fold_mean"] = float(res.fold_df[col].mean())
                    row[f"{col}_fold_std"] = float(res.fold_df[col].std(ddof=0))

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by="balanced_accuracy",
        ascending=False
    ).reset_index(drop=True)


# ============================================================================
# Save helpers
# ============================================================================

def save_analysis_result(result: M3GLVQAnalysisResult, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.fold_df.to_pickle(out_dir / f"{result.run_name}_folds.pkl")
    result.oof_df.to_pickle(out_dir / f"{result.run_name}_oof.pkl")

    with (out_dir / f"{result.run_name}_summary.pkl").open("wb") as f:
        pickle.dump(result.summary, f)

    if result.tracked_paths is not None:
        with (out_dir / f"{result.run_name}_paths.pkl").open("wb") as f:
            pickle.dump(result.tracked_paths, f)


def save_reference_profile_bundle(bundle: Dict[str, pd.DataFrame], out_dir, prefix: str = "profiles") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle["good_runs"].to_pickle(out_dir / f"{prefix}_good_runs.pkl")
    bundle["labeled_runs"].to_pickle(out_dir / f"{prefix}_labeled_runs.pkl")
    bundle["cluster_summary"].to_pickle(out_dir / f"{prefix}_cluster_summary.pkl")