import pandas as pd


def summarize_fixed_profiles(all_fixed_fold_df: pd.DataFrame) -> pd.DataFrame:
    summary_fixed = (
        all_fixed_fold_df
        .groupby(["profile_name", "K_0", "K_1"], as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .sort_values("balanced_accuracy_mean", ascending=False)
    )
    return summary_fixed
