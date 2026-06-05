from pathlib import Path
import json
import pickle
from typing import Dict

import pandas as pd


def build_customer_mapping(customer_ids, start: int = 1) -> dict[str, str]:
    ids = sorted(pd.Index(customer_ids).astype(str).unique())
    return {cid: str(i) for i, cid in enumerate(ids, start=start)}


def apply_mapping_to_y(
    y_df: pd.DataFrame,
    customer_mapping: dict[str, str],
    private_product_code: str,
    public_product_code: str,
) -> pd.DataFrame:
    df = y_df.copy()

    if private_product_code not in df.columns:
        raise ValueError(
            f"Spalte '{private_product_code}' nicht in y_df gefunden. "
            f"Vorhanden: {list(df.columns)}"
        )

    if "CustomerCode" not in df.columns:
        raise ValueError("y_df braucht die Spalte 'CustomerCode'.")

    df["CustomerCode"] = df["CustomerCode"].astype(str)

    missing = set(df["CustomerCode"]) - set(customer_mapping.keys())
    if missing:
        raise ValueError(f"CustomerCodes im y_df nicht vollständig im Mapping: {list(sorted(missing))[:5]}")

    df["CustomerCode"] = df["CustomerCode"].map(customer_mapping)
    df = df.rename(columns={private_product_code: public_product_code})

    return df


def apply_mapping_to_matrix(
    D_df: pd.DataFrame,
    customer_mapping: dict[str, str],
) -> pd.DataFrame:
    D = D_df.copy()

    D.index = D.index.astype(str)
    D.columns = D.columns.astype(str)

    missing_idx = set(D.index) - set(customer_mapping.keys())
    missing_col = set(D.columns) - set(customer_mapping.keys())

    if missing_idx:
        raise ValueError(f"Matrix-Index nicht vollständig im Mapping: {list(sorted(missing_idx))[:5]}")
    if missing_col:
        raise ValueError(f"Matrix-Spalten nicht vollständig im Mapping: {list(sorted(missing_col))[:5]}")

    D.index = D.index.map(customer_mapping)
    D.columns = D.columns.map(customer_mapping)

    return D


def collect_customer_ids(
    y_df: pd.DataFrame,
    matrices: Dict[str, pd.DataFrame],
) -> list[str]:
    ids = set(y_df["CustomerCode"].astype(str))

    for _, D in matrices.items():
        ids.update(pd.Index(D.index).astype(str))
        ids.update(pd.Index(D.columns).astype(str))

    return sorted(ids)


def pseudonymize_product_bundle(
    *,
    private_y_base: str | Path,
    private_d_base: str | Path,
    public_y_base: str | Path,
    public_d_base: str | Path,
    private_product_code: str = "0201",
    public_product_code: str = "A",
    mapping_out_path: str | Path | None = None,
    start_index: int = 1,
    overwrite: bool = True,
) -> dict:
    private_y_base = Path(private_y_base)
    private_d_base = Path(private_d_base)
    public_y_base = Path(public_y_base)
    public_d_base = Path(public_d_base)

    public_y_base.mkdir(parents=True, exist_ok=True)
    public_d_base.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load private source data
    # ----------------------------
    y_path = private_y_base / f"{private_product_code}.pkl"
    if not y_path.exists():
        raise FileNotFoundError(y_path)

    y_df = pd.read_pickle(y_path).copy()

    matrices = {
        "naics": pd.read_pickle(private_d_base / "dissimilarity_naics_matrix.pkl"),
        "hs": pd.read_pickle(private_d_base / "dissimilarity_hs_matrix.pkl"),
        "am": pd.read_pickle(private_d_base / f"dissimilarity_am_{private_product_code}_matrix.pkl"),
    }

    # ----------------------------
    # Build mapping
    # ----------------------------
    customer_ids = collect_customer_ids(y_df, matrices)
    customer_mapping = build_customer_mapping(customer_ids, start=start_index)

    # ----------------------------
    # Apply mapping
    # ----------------------------
    y_public = apply_mapping_to_y(
        y_df=y_df,
        customer_mapping=customer_mapping,
        private_product_code=private_product_code,
        public_product_code=public_product_code,
    )

    matrices_public = {
        "naics": apply_mapping_to_matrix(matrices["naics"], customer_mapping),
        "hs": apply_mapping_to_matrix(matrices["hs"], customer_mapping),
        "am": apply_mapping_to_matrix(matrices["am"], customer_mapping),
    }

    # ----------------------------
    # Save anonymized public files
    # ----------------------------
    y_public_path = public_y_base / f"{public_product_code}.pkl"
    d_naics_path = public_d_base / "dissimilarity_naics_matrix.pkl"
    d_hs_path = public_d_base / "dissimilarity_hs_matrix.pkl"
    d_am_path = public_d_base / f"dissimilarity_am_{public_product_code}_matrix.pkl"

    if (not overwrite) and any(p.exists() for p in [y_public_path, d_naics_path, d_hs_path, d_am_path]):
        raise FileExistsError("Zieldateien existieren bereits und overwrite=False.")

    y_public.to_pickle(y_public_path)
    matrices_public["naics"].to_pickle(d_naics_path)
    matrices_public["hs"].to_pickle(d_hs_path)
    matrices_public["am"].to_pickle(d_am_path)

    # ----------------------------
    # Optional: save private mapping outside repo
    # ----------------------------
    if mapping_out_path is not None:
        mapping_out_path = Path(mapping_out_path)
        mapping_out_path.parent.mkdir(parents=True, exist_ok=True)

        reverse_mapping = {v: k for k, v in customer_mapping.items()}

        payload = {
            "private_product_code": private_product_code,
            "public_product_code": public_product_code,
            "customer_mapping": customer_mapping,
            "reverse_customer_mapping": reverse_mapping,
        }

        with mapping_out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "n_customers": len(customer_mapping),
        "public_product_code": public_product_code,
        "y_public_path": str(y_public_path),
        "d_naics_path": str(d_naics_path),
        "d_hs_path": str(d_hs_path),
        "d_am_path": str(d_am_path),
    }