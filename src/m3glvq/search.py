from itertools import product
from typing import Any
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def build_K_grid(K0_values, K1_values) -> list[dict]:
    return [{"K0": k0, "K1": k1} for k0, k1 in product(K0_values, K1_values)]


def run_m3glvq_search(
    D,
    y,
    model_cls,
    meta_df: pd.DataFrame,
    run_m3glvq_cv_analysis,
    etas,
    K0_values,
    K1_values,
    T: int = 150,
    v_init=None,
    n_splits: int = 5,
    random_state: int = 42,
    store_paths: bool = True,
    show_progress: bool = True,
):
    if v_init is None:
        v_init = np.ones(len(D)) / len(D)

    all_fold_dfs = []
    all_results = []

    param_grid = list(product(etas, K0_values, K1_values))
    iterator = param_grid

    if show_progress:
        iterator = tqdm(param_grid, desc="M3GLVQ Search", total=len(param_grid), leave=True)

    for eta, k0, k1 in iterator:
        params = {
            "K": {0: k0, 1: k1},
            "T": T,
            "eta": eta,
            "v_init": np.asarray(v_init, dtype=float),
            "track_path": True,
            "track_vweights": True,
            "track_metrics": True,
        }

        run_name = f"m3glvq_k0_{k0}_k1_{k1}_eta_{str(eta).replace('.', '')}"

        res = run_m3glvq_cv_analysis(
            D=D,
            y=y,
            model_cls=model_cls,
            model_params=params,
            meta_df=meta_df,
            n_splits=n_splits,
            random_state=random_state,
            run_name=run_name,
            store_paths=store_paths,
            verbose=False,
        )

        fold_df = res.fold_df.copy()
        fold_df["K_0"] = k0
        fold_df["K_1"] = k1
        fold_df["eta"] = eta
        fold_df["run_name"] = run_name

        all_fold_dfs.append(fold_df)
        all_results.append(res)

        if show_progress and hasattr(iterator, "set_postfix"):
            ba = fold_df["balanced_accuracy"].mean() if "balanced_accuracy" in fold_df.columns else None
            iterator.set_postfix({
                "K0": k0,
                "K1": k1,
                "eta": eta,
                "bal_acc": f"{ba:.4f}" if ba is not None else "na",
            })

    all_runs_fold_df = pd.concat(all_fold_dfs, ignore_index=True)
    return all_runs_fold_df, all_results
