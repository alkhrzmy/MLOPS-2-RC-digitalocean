from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_logs(log_path: Path) -> pd.DataFrame:
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def plot_hist(series: pd.Series, title: str, xlabel: str, outfile: Path, bins: int = 30):
    plt.figure(figsize=(8, 5))
    series.dropna().plot(kind="hist", bins=bins, color="#5dd4ff", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_bar_counts(series: pd.Series, title: str, outfile: Path, max_classes: int = 30):
    counts = series.value_counts().sort_values(ascending=False).head(max_classes)
    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar", color="#7b7bff")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def build_report(log_path: Path, output_dir: Path, max_classes: int = 30):
    df = load_logs(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": len(df),
        "latency_ms_mean": float(df.get("latency_ms", pd.Series(dtype=float)).mean() or 0),
        "latency_ms_p95": float(df.get("latency_ms", pd.Series(dtype=float)).quantile(0.95) or 0),
        "audio_duration_mean": float(df.get("audio_duration", pd.Series(dtype=float)).mean() or 0),
        "confidence_mean": float(df.get("confidence", pd.Series(dtype=float)).mean() or 0),
    }

    plot_hist(df.get("latency_ms"), "Latency distribution (ms)", "Latency (ms)", output_dir / "latency_hist.png")
    plot_hist(df.get("audio_duration"), "Audio duration (s)", "Duration (s)", output_dir / "duration_hist.png")
    plot_hist(df.get("confidence"), "Confidence distribution", "Confidence", output_dir / "confidence_hist.png")

    if "product" in df:
        plot_bar_counts(df["product"], "Predicted product counts", output_dir / "product_counts.png", max_classes)
    if "quantity" in df:
        plot_bar_counts(df["quantity"], "Predicted quantity counts", output_dir / "quantity_counts.png", max_classes)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate monitoring report from inference logs")
    parser.add_argument("--log", default="logs/inference.log", help="Path to inference log (JSON lines)")
    parser.add_argument("--output-dir", default="artifacts/monitoring", help="Where to store report")
    parser.add_argument("--max-classes", type=int, default=30, help="Max classes per bar plot")
    args = parser.parse_args()

    summary = build_report(Path(args.log), Path(args.output_dir), args.max_classes)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
