from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def build_K_grid(K0_values, K1_values) -> list[dict]:
    return [{"K0": k0, "K1": k1} for k0 in K0_values for k1 in K1_values]


def run_m3glvq_persisted(
    D,
    y,
    model_cls,
    meta_df: pd.DataFrame,
    run_m3glvq_cv_analysis,
    etas,
    K_grid,
    out_dir="m3glvq_runs",
    T: int = 150,
    v_init=None,
    n_splits: int = 5,
    random_state: int = 42,
    resume: bool = True,
    save_full_result: bool = True,
    verbose: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if v_init is None:
        v_init = np.ones(len(D)) / len(D)

    all_rows = []
    all_results = []

    jobs = [(eta, item["K0"], item["K1"]) for eta in etas for item in K_grid]
    iterator = tqdm(jobs, desc="M3GLVQ persisted", total=len(jobs), leave=True) if verbose else jobs

    for eta, k0, k1 in iterator:
        run_name = f"m3glvq_k0_{k0}_k1_{k1}_eta_{str(eta).replace('.', '')}"
        run_path = out_dir / run_name
        fold_path = run_path / "fold_df.csv"
        result_path = run_path / "result.pkl"

        if resume and fold_path.exists():
            fold_df = pd.read_csv(fold_path)
            all_rows.append(fold_df)

            if save_full_result and result_path.exists():
                try:
                    all_results.append(_load_pickle(result_path))
                except Exception:
                    pass
            continue

        run_path.mkdir(parents=True, exist_ok=True)

        params = {
            "K": {0: k0, 1: k1},
            "T": T,
            "eta": eta,
            "v_init": np.asarray(v_init, dtype=float),
            "track_path": True,
            "track_vweights": True,
            "track_metrics": True,
        }

        res = run_m3glvq_cv_analysis(
            D=D,
            y=y,
            model_cls=model_cls,
            model_params=params,
            meta_df=meta_df,
            n_splits=n_splits,
            random_state=random_state,
            run_name=run_name,
            store_paths=True,
            verbose=False,
        )

        fold_df = res.fold_df.copy()
        fold_df["K_0"] = k0
        fold_df["K_1"] = k1
        fold_df["eta"] = eta
        fold_df["run_name"] = run_name

        fold_df.to_csv(fold_path, index=False)
        all_rows.append(fold_df)

        if save_full_result:
            _save_pickle(res, result_path)
            all_results.append(res)

    all_runs_fold_df = pd.concat(all_rows, ignore_index=True)

    all_runs_fold_df.to_csv(out_dir / "all_runs_fold_df.csv", index=False)

    return all_runs_fold_df, all_results


def run_m3glvq_fixed_weights_persisted(
    D,
    y,
    model_cls,
    meta_df: pd.DataFrame,
    run_m3glvq_cv_analysis,
    profiles,
    T: int = 150,
    n_splits: int = 10,
    random_state: int = 42,
    out_dir="m3glvq_fixed_runs",
    resume: bool = True,
    save_full_result: bool = True,
    verbose: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_results = []

    iterator = tqdm(profiles, desc="M3GLVQ fixed profiles", total=len(profiles), leave=True) if verbose else profiles

    for profile in iterator:
        profile_name = profile["profile_name"]
        weights = np.asarray(profile["weights"], dtype=float)
        K = profile["K"]

        run_name = f"{profile_name}_k0_{K[0]}_k1_{K[1]}"
        run_path = out_dir / run_name
        fold_path = run_path / "fold_df.csv"
        result_path = run_path / "result.pkl"

        if resume and fold_path.exists():
            fold_df = pd.read_csv(fold_path)
            all_rows.append(fold_df)

            if save_full_result and result_path.exists():
                try:
                    all_results.append(_load_pickle(result_path))
                except Exception:
                    pass
            continue

        run_path.mkdir(parents=True, exist_ok=True)

        weights = np.asarray(profile["weights"], dtype=float)
        weights = weights / weights.sum()
        
        params = {
            "K": K,
            "T": T,
            "eta": 0.0,
            "v_init": weights,
            "track_path": True,
            "track_vweights": True,
            "track_metrics": True,
        }

        res = run_m3glvq_cv_analysis(
            D=D,
            y=y,
            model_cls=model_cls,
            model_params=params,
            meta_df=meta_df,
            n_splits=n_splits,
            random_state=random_state,
            run_name=run_name,
            store_paths=True,
            verbose=False,
        )

        fold_df = res.fold_df.copy()
        fold_df["profile_name"] = profile_name
        fold_df["K_0"] = K[0]
        fold_df["K_1"] = K[1]
        fold_df["vweight_0"] = weights[0]
        fold_df["vweight_1"] = weights[1]
        fold_df["vweight_2"] = weights[2]
        fold_df["run_name"] = run_name

        fold_df.to_csv(fold_path, index=False)
        all_rows.append(fold_df)

        if save_full_result:
            _save_pickle(res, result_path)
            all_results.append(res)

    all_fixed_fold_df = pd.concat(all_rows, ignore_index=True)
    all_fixed_fold_df.to_csv(out_dir / "all_fixed_fold_df.csv", index=False)

    return all_fixed_fold_df, all_results
