from __future__ import annotations

from pathlib import Path
import json


def save_artifact(model_path: Path, config_path: Path, label_map_path: Path, metrics: dict, registry_dir: Path = Path("artifacts/registry")) -> Path:
    registry_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "model_path": model_path.as_posix(),
        "config_path": config_path.as_posix(),
        "label_map_path": label_map_path.as_posix(),
        "metrics": metrics,
    }
    target = registry_dir / "latest.json"
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return target
