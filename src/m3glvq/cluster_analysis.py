import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def select_top_runs(
    all_runs_fold_df: pd.DataFrame,
    metric_col: str = "balanced_accuracy",
    quantile: float = 0.9,
) -> tuple[pd.DataFrame, float]:
    threshold = all_runs_fold_df[metric_col].quantile(quantile)
    df_high = all_runs_fold_df[all_runs_fold_df[metric_col] >= threshold].copy()
    return df_high, threshold


def cluster_weight_profiles(
    df_high: pd.DataFrame,
    v_cols=("vweight_0", "vweight_1", "vweight_2"),
    n_clusters: int = 5,
    random_state: int = 42,
    top_frac: float = 0.20,
):
    X = df_high[list(v_cols)].values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df_high = df_high.copy()
    df_high["PC1"] = X_pca[:, 0]
    df_high["PC2"] = X_pca[:, 1]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_

    df_high["cluster"] = labels

    core_mask = np.zeros(len(df_high), dtype=bool)

    for c in range(n_clusters):
        idx_c = np.where(labels == c)[0]
        X_c = X[idx_c]
        center = cluster_centers[c]

        dists = np.linalg.norm(X_c - center, axis=1)
        n_top = max(1, int(len(idx_c) * top_frac))
        idx_sorted = np.argsort(dists)
        core_idx_cluster = idx_c[idx_sorted[:n_top]]
        core_mask[core_idx_cluster] = True

    filtered_centers = np.zeros_like(cluster_centers)

    for c in range(n_clusters):
        mask_core_c = (labels == c) & core_mask
        X_c_core = X[mask_core_c]

        if len(X_c_core) > 0:
            filtered_centers[c] = X_c_core.mean(axis=0)
        else:
            filtered_centers[c] = cluster_centers[c]

    centers_pca = pca.transform(filtered_centers)

    return {
        "df_high": df_high,
        "X": X,
        "labels": labels,
        "pca": pca,
        "X_pca": X_pca,
        "cluster_centers": cluster_centers,
        "filtered_centers": filtered_centers,
        "centers_pca": centers_pca,
        "core_mask": core_mask,
    }


def summarize_clusters(
    df_high: pd.DataFrame,
    filtered_centers,
    v_cols=("vweight_0", "vweight_1", "vweight_2"),
    k_cols=("K_0", "K_1"),
    metric_col="balanced_accuracy",
):
    cluster_info = []

    for c in sorted(df_high["cluster"].unique()):
        df_cluster = df_high[df_high["cluster"] == c]

        size = len(df_cluster)
        mean_acc = df_cluster[metric_col].mean()
        center = filtered_centers[c]

        weight_means = df_cluster[list(v_cols)].mean()
        weight_stds = df_cluster[list(v_cols)].std()

        k_means = df_cluster[list(k_cols)].mean()
        k_stds = df_cluster[list(k_cols)].std()

        cluster_info.append({
            "cluster": c,
            "size": size,
            "mean_balanced_accuracy": mean_acc,
            "center": center,
            "weight_means": weight_means,
            "weight_stds": weight_stds,
            "k_means": k_means,
            "k_stds": k_stds
        })

    return sorted(cluster_info, key=lambda x: x["size"], reverse=True)


def extract_core_k_points(
    df_high: pd.DataFrame,
    X,
    labels,
    filtered_centers,
    n_clusters: int,
    n_core: int = 20,
):
    core20_mask = np.zeros(len(df_high), dtype=bool)

    for c in range(n_clusters):
        idx_c = np.where(labels == c)[0]
        X_c = X[idx_c]
        center = filtered_centers[c]

        dists = np.linalg.norm(X_c - center, axis=1)

        n_take = min(n_core, len(idx_c))
        idx_sorted = np.argsort(dists)
        idx_core = idx_c[idx_sorted[:n_take]]

        core20_mask[idx_core] = True

    out = df_high.copy()
    out["core20"] = core20_mask
    return out


def summarize_core_points(
    df_high: pd.DataFrame,
    v_cols=("vweight_0", "vweight_1", "vweight_2"),
    k_cols=("K_0", "K_1"),
):
    cluster_core_info = []

    for c in sorted(df_high["cluster"].unique()):
        df_cluster_core = df_high[
            (df_high["cluster"] == c) & (df_high["core20"])
        ].copy()

        size_core = len(df_cluster_core)
        if size_core == 0:
            continue

        mean_acc = df_cluster_core["balanced_accuracy"].mean()
        std_acc = df_cluster_core["balanced_accuracy"].std()

        mean_recall = df_cluster_core["recall"].mean() if "recall" in df_cluster_core.columns else np.nan
        std_recall = df_cluster_core["recall"].std() if "recall" in df_cluster_core.columns else np.nan

        weight_means = df_cluster_core[list(v_cols)].mean()
        weight_stds = df_cluster_core[list(v_cols)].std()

        k_means = df_cluster_core[list(k_cols)].mean()
        k_stds = df_cluster_core[list(k_cols)].std()

        core_center = df_cluster_core[list(v_cols)].mean().values

        cluster_core_info.append({
            "cluster": c,
            "size_core": size_core,
            "mean_balanced_accuracy_core": mean_acc,
            "std_balanced_accuracy_core": std_acc,
            "mean_recall_core": mean_recall,
            "std_recall_core": std_recall,
            "core_center": core_center,
            "vweight_0_mean": weight_means["vweight_0"],
            "vweight_1_mean": weight_means["vweight_1"],
            "vweight_2_mean": weight_means["vweight_2"],
            "vweight_0_std": weight_stds["vweight_0"],
            "vweight_1_std": weight_stds["vweight_1"],
            "vweight_2_std": weight_stds["vweight_2"],
            "K_0_mean": k_means["K_0"],
            "K_1_mean": k_means["K_1"],
            "K_0_std": k_stds["K_0"],
            "K_1_std": k_stds["K_1"],
        })

    cluster_core_df = pd.DataFrame(cluster_core_info)
    if not cluster_core_df.empty:
        cluster_core_df = cluster_core_df.sort_values("size_core", ascending=False)

    return cluster_core_df
