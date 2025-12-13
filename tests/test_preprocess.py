from pathlib import Path
from slu import preprocess


def test_load_config(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("a: 1", encoding="utf-8")
    cfg = preprocess.load_config(cfg_path)
    assert cfg["a"] == 1
