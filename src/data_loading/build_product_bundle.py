from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.data_loading.load_targets import load_targets
from src.data_loading.load_matrices import load_dissimilarity_matrices


@dataclass
class ProductBundle:
    product_code: str
    y_df: pd.DataFrame
    y: pd.Series
    matrices: dict[str, pd.DataFrame]


def load_product_bundle(
    product_code: str,
    y_base_path: Path,
    d_base_path: Path,
) -> ProductBundle:
    """
    Baut einen Bundle für genau einen Produktcode:
    - y_df
    - y als Series mit CustomerCode-Index
    - matrices = {naics, hs, am}
    """

    targets = load_targets(y_base_path, [product_code])
    y_df = targets[product_code].copy()

    y = pd.Series(
        y_df["y"].to_numpy(dtype=int),
        index=y_df["CustomerCode"].astype(str),
        name="y",
    )

    all_matrices = load_dissimilarity_matrices(d_base_path)

    required = {
        "dissimilarity_naics_matrix",
        "dissimilarity_hs_matrix",
        f"dissimilarity_am_{product_code}_matrix",
    }
    missing = required - set(all_matrices.keys())
    if missing:
        raise ValueError(f"Fehlende Matrizen für {product_code}: {missing}")

    matrices = {
        "naics": all_matrices["dissimilarity_naics_matrix"],
        "hs": all_matrices["dissimilarity_hs_matrix"],
        "am": all_matrices[f"dissimilarity_am_{product_code}_matrix"],
    }

    return ProductBundle(
        product_code=product_code,
        y_df=y_df,
        y=y,
        matrices=matrices,
    )