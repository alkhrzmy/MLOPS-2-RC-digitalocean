from __future__ import annotations

from pathlib import Path
import pandas as pd


def basic_stats(metadata_path: Path) -> dict:
    df = pd.read_csv(metadata_path)
    duration_mean = df["duration"].mean() if "duration" in df else None
    product_counts = df["product"].value_counts().to_dict() if "product" in df else {}
    qty_counts = df["quantity"].value_counts().to_dict() if "quantity" in df else {}
    return {
        "rows": len(df),
        "duration_mean": duration_mean,
        "product_counts": product_counts,
        "qty_counts": qty_counts,
    }


def main(metadata_path: str = "data/processed/metadata.csv") -> None:
    stats = basic_stats(Path(metadata_path))
    print(stats)


if __name__ == "__main__":
    main()
