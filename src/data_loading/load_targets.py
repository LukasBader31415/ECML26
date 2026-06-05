from pathlib import Path
import pandas as pd


def load_targets(base_path: Path, product_codes: list[str]) -> dict[str, pd.DataFrame]:
    """
    Lädt die y-DataFrames für alle Produktcodes.
    Erwartet Dateien wie 0201.pkl, 0601.pkl, ...
    """
    out: dict[str, pd.DataFrame] = {}

    for code in product_codes:
        file_path = base_path / f"{code}.pkl"
        df = pd.read_pickle(file_path).copy()

        if code not in df.columns:
            raise ValueError(f"Spalte '{code}' nicht in {file_path} gefunden.")

        df = df.rename(columns={code: "y"})

        required_cols = {"CustomerCode", "y"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{file_path} fehlt: {missing}")

        df["CustomerCode"] = df["CustomerCode"].astype(str)
        df["y"] = df["y"].astype(int)

        out[code] = df

    return out