from pathlib import Path
import pandas as pd


def load_dissimilarity_matrices(base_path: Path) -> dict[str, pd.DataFrame]:
    """
    Lädt alle Dateien dissimilarity_*.pkl aus einem Ordner.
    """
    matrices = {
        file.stem: pd.read_pickle(file)
        for file in base_path.glob("dissimilarity_*.pkl")
    }

    if not matrices:
        raise ValueError(f"Keine dissimilarity_*.pkl Dateien gefunden in {base_path}")

    return matrices