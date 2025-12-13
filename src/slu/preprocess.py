from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def waveform_to_logmel(waveform: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int, win_length: int, power: float) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, power=power)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    return logmel.astype(np.float32)


def process_row(row: pd.Series, cfg: dict, out_dir: Path, raw_root: Path, min_dur: float, max_dur: float) -> tuple[Path | None, float | None, int | None]:
    wav_rel = Path(row["path"]) if Path(row["path"]).is_absolute() else raw_root / row["path"]
    if not wav_rel.exists():
        return None, None, None
    waveform, _ = librosa.load(wav_rel, sr=cfg["sample_rate"], mono=True)
    duration = waveform.shape[0] / cfg["sample_rate"]
    if cfg.get("trim_top_db") is not None:
        waveform, _ = librosa.effects.trim(waveform, top_db=cfg["trim_top_db"])
        duration = waveform.shape[0] / cfg["sample_rate"]
    if duration < min_dur or duration > max_dur:
        return None, duration, None
    logmel = waveform_to_logmel(
        waveform,
        sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        win_length=cfg["win_length"],
        power=cfg.get("power", 2.0),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{row['id']}.npy"
    np.save(target, logmel)
    frames = int(logmel.shape[1])
    return target, duration, frames


def preprocess_metadata(metadata_path: Path, cfg_path: Path, raw_root: Path | None = None) -> Path:
    cfg = load_config(cfg_path)
    df = pd.read_csv(metadata_path)
    # Pastikan kolom wajib ada; jika id belum ada, buat dari index
    if "id" not in df.columns:
        df["id"] = range(len(df))
    required_cols = {"id", "path", "product", "quantity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib hilang di metadata: {missing}")

    raw_root = raw_root or Path(cfg.get("raw_dir", ".."))
    feature_dir = Path(cfg.get("cache_dir", "data/features"))
    min_dur = cfg.get("min_duration", 0.5)
    max_dur = cfg.get("max_duration", 5.0)

    results = df.apply(lambda r: process_row(r, cfg, feature_dir, raw_root, min_dur, max_dur), axis=1)
    df["feature_path"], df["duration"], df["frames"] = zip(*results)
    before = len(df)
    df = df.dropna(subset=["feature_path"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropping {dropped} rows (missing audio or duration out of range)")

    out_path = metadata_path.parent / "metadata_features.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main(metadata_path: str = "data/processed/metadata.csv", cfg_path: str = "configs/preprocess.yaml") -> None:
    preprocess_metadata(Path(metadata_path), Path(cfg_path))


if __name__ == "__main__":
    main()
