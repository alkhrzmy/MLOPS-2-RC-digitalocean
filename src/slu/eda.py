from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def basic_stats(df: pd.DataFrame) -> dict:
    duration_mean = df["duration"].mean() if "duration" in df else None
    product_counts = df["product"].value_counts().to_dict() if "product" in df else {}
    qty_counts = df["quantity"].value_counts().to_dict() if "quantity" in df else {}
    return {
        "rows": len(df),
        "duration_mean": duration_mean,
        "product_counts": product_counts,
        "qty_counts": qty_counts,
    }


def plot_counts(series: pd.Series, title: str, outfile: Path, max_classes: int = 30) -> None:
    counts = series.value_counts().sort_values(ascending=False).head(max_classes)
    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar", color="#5dd4ff")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_duration(df: pd.DataFrame, outfile: Path) -> None:
    if "duration" not in df:
        return
    plt.figure(figsize=(8, 5))
    df["duration"].plot(kind="hist", bins=30, color="#7b7bff", edgecolor="white")
    plt.title("Duration distribution (seconds)")
    plt.xlabel("Duration (s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def run_eda(metadata_path: Path, output_dir: Path, max_classes: int = 30) -> Tuple[dict, Path]:
    df = pd.read_csv(metadata_path)
    summary = basic_stats(df)

    product_png = output_dir / "product_counts.png"
    qty_png = output_dir / "quantity_counts.png"
    duration_png = output_dir / "duration_hist.png"
    summary_json = output_dir / "summary.json"

    if "product" in df:
        plot_counts(df["product"], "Product frequency", product_png, max_classes)
    if "quantity" in df:
        plot_counts(df["quantity"], "Quantity frequency", qty_png, max_classes)
    plot_duration(df, duration_png)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2))

    return summary, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA plots and stats for SLU metadata")
    parser.add_argument("--metadata", default="data/processed/metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--output-dir", default="artifacts/eda", help="Directory to store plots and summary")
    parser.add_argument("--max-classes", type=int, default=30, help="Max classes per bar plot")
    args = parser.parse_args()

    summary, out_dir = run_eda(Path(args.metadata), Path(args.output_dir), args.max_classes)
    print(f"Saved EDA to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
