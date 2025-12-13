from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_metadata(raw_dir: Path, metadata_files: list[str]) -> pd.DataFrame:
    frames = []
    for name in metadata_files:
        src = raw_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Metadata file missing: {src}")
        frames.append(pd.read_csv(src))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)

    # Buang kolom duplikat yang muncul setelah rename
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Normalisasi kolom
    rename_map = {
        "produk": "product",
        "product_id": "product",
        "file_audio": "path",
        "file_path": "path",
        "file_path_abs": "path_abs",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Tentukan path audio
    if "path" not in df.columns and "path_abs" in df.columns:
        df["path"] = df["path_abs"]
    if "path" not in df.columns:
        raise ValueError("Tidak menemukan kolom path/file_audio/file_path di metadata.")

    # Drop baris tanpa product/quantity
    missing_cols = [c for c in ["product", "quantity"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib hilang di metadata gabungan: {missing_cols}")
    before = len(df)
    df = df.dropna(subset=["product", "quantity", "path"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropping {dropped} rows tanpa product/quantity/path")

    if len(df) == 0:
        raise ValueError("Metadata kosong setelah pembersihan. Periksa kolom product/quantity/path.")

    # Isi path absolut jika ada path_rel
    def resolve_path(p_raw: str):
        p = Path(p_raw)
        if p.is_absolute():
            return p.as_posix()
        return (raw_dir / p).resolve().as_posix()

    df["path"] = df["path"].astype(str).apply(resolve_path)
    return df


def save_metadata(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def build_and_save_label_maps(df: pd.DataFrame, product_path: Path, qty_path: Path) -> None:
    product_series = pd.Series(df["product"].astype(str)).squeeze()
    qty_series = pd.Series(df["quantity"].astype(str)).squeeze()
    product_vals = sorted(product_series.unique())
    qty_vals = sorted(qty_series.unique())
    product_map = {v: i for i, v in enumerate(product_vals)}
    qty_map = {v: i for i, v in enumerate(qty_vals)}
    product_path.parent.mkdir(parents=True, exist_ok=True)
    product_path.write_text(json.dumps(product_map, indent=2), encoding="utf-8")
    qty_path.write_text(json.dumps(qty_map, indent=2), encoding="utf-8")


def main(config_path: str = "configs/data.yaml") -> None:
    cfg = load_config(Path(config_path))
    raw_dir = Path(cfg.get("raw_dir", ".."))
    metadata_files: list[str] = cfg.get("metadata_files", [])
    out_dir = Path(cfg.get("output_dir", "data/processed"))
    df = build_metadata(raw_dir, metadata_files)
    save_metadata(df, out_dir / "metadata.csv")

    label_maps = cfg.get("label_maps", {})
    product_path = Path(label_maps.get("product", out_dir / "product2id.json"))
    qty_path = Path(label_maps.get("quantity", out_dir / "qty2id.json"))
    build_and_save_label_maps(df, product_path, qty_path)
    print(f"Saved label maps to {product_path} and {qty_path}")


if __name__ == "__main__":
    main()
