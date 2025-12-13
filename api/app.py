from __future__ import annotations

import io
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schema import HealthResponse, PredictResponse
from slu.models.m1_cnn_transformer import CNNTransformerSLU
from slu.models.m2_transformer_tiny import TransformerTinySLU

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "artifacts" / "registry" / "latest.json"
DEFAULT_PREPROCESS_CFG = PROJECT_ROOT / "configs" / "preprocess.yaml"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SLU Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise RuntimeError(f"Registry not found: {REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text())


@lru_cache(maxsize=1)
def load_label_maps(product_path: Path, qty_path: Path) -> Tuple[dict, dict, dict, dict]:
    product_map = json.loads(product_path.read_text())
    qty_map = json.loads(qty_path.read_text())
    inv_product = {int(v): k for k, v in product_map.items()}
    inv_qty = {int(v): k for k, v in qty_map.items()}
    return product_map, qty_map, inv_product, inv_qty


def build_model(model_cfg_path: Path, num_products: int, num_quantities: int) -> torch.nn.Module:
    cfg = load_yaml(model_cfg_path)
    name = cfg.get("model_name", "cnn_transformer")
    if name == "cnn_transformer":
        model = CNNTransformerSLU(
            num_products=num_products,
            num_quantities=num_quantities,
            d_model=cfg.get("transformer", {}).get("d_model", 256),
            nhead=cfg.get("transformer", {}).get("nhead", 4),
            num_layers=cfg.get("transformer", {}).get("num_layers", 2),
        )
    elif name == "transformer_tiny":
        model = TransformerTinySLU(
            num_products=num_products,
            num_quantities=num_quantities,
            d_model=cfg.get("transformer", {}).get("d_model", 128),
            nhead=cfg.get("transformer", {}).get("nhead", 4),
            num_layers=cfg.get("transformer", {}).get("num_layers", 2),
        )
    else:
        raise ValueError(f"Unknown model_name: {name}")
    return model


@lru_cache(maxsize=1)
def load_model_cached() -> Tuple[torch.nn.Module, dict, dict, dict, dict]:
    reg = load_registry()
    model_path = PROJECT_ROOT / reg["model_path"] if not Path(reg["model_path"]).is_absolute() else Path(reg["model_path"])
    config_path = PROJECT_ROOT / reg["config_path"] if not Path(reg["config_path"]).is_absolute() else Path(reg["config_path"])
    label_map_path = PROJECT_ROOT / reg["label_map_path"] if not Path(reg["label_map_path"]).is_absolute() else Path(reg["label_map_path"])

    # label_map_path expected to contain product2id and qty2id
    label_dir = label_map_path.parent
    product_map_path = label_dir / "product2id.json"
    qty_map_path = label_dir / "qty2id.json"
    product_map, qty_map, inv_product, inv_qty = load_label_maps(product_map_path, qty_map_path)

    model = build_model(config_path, num_products=len(product_map), num_quantities=len(qty_map))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device, product_map, qty_map, {"product": inv_product, "quantity": inv_qty}


def load_audio_to_logmel(data: bytes, preprocess_cfg: Path = DEFAULT_PREPROCESS_CFG) -> tuple[np.ndarray, int]:
    cfg = load_yaml(preprocess_cfg)
    sr = cfg.get("sample_rate", 16000)
    trim_top_db = cfg.get("trim_top_db", 30)
    n_mels = cfg.get("n_mels", 128)
    n_fft = cfg.get("n_fft", 1024)
    hop_length = cfg.get("hop_length", 256)
    win_length = cfg.get("win_length", 1024)
    power = cfg.get("power", 2.0)
    min_dur = cfg.get("min_duration", 0.5)
    max_dur = cfg.get("max_duration", 5.0)

    buf = io.BytesIO(data)
    waveform, _ = librosa.load(buf, sr=sr, mono=True)
    if trim_top_db is not None:
        waveform, _ = librosa.effects.trim(waveform, top_db=trim_top_db)
    duration = waveform.shape[0] / sr
    if duration < min_dur or duration > max_dur:
        raise ValueError(f"Audio duration {duration:.2f}s di luar rentang {min_dur}-{max_dur}s")
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=power,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    frames = logmel.shape[1]
    return logmel.astype(np.float32), frames


def infer(logmel: np.ndarray):
    model, device, _, _, inv_maps = load_model_cached()
    tensor = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(device)
    lengths = torch.tensor([logmel.shape[1]], device=device)
    with torch.no_grad():
        outputs = model(tensor)
        prod_logits = outputs["product"]
        qty_logits = outputs["quantity"]
        prod_prob = torch.softmax(prod_logits, dim=-1)
        qty_prob = torch.softmax(qty_logits, dim=-1)
        prod_idx = int(prod_prob.argmax(dim=-1).item())
        qty_idx = int(qty_prob.argmax(dim=-1).item())
        prod_conf = float(prod_prob.max().item())
        qty_conf = float(qty_prob.max().item())
    return {
        "product": inv_maps["product"].get(prod_idx, str(prod_idx)),
        "quantity": inv_maps["quantity"].get(qty_idx, str(qty_idx)),
        "confidence": min(prod_conf, qty_conf),
    }


def log_event(event: dict) -> None:
    timestamp = int(time.time() * 1000)
    path = LOG_DIR / "inference.log"
    line = {"ts": timestamp, **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if file.content_type not in {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/flac", "audio/mp3", "audio/wave"}:
        raise HTTPException(status_code=400, detail="Unsupported audio type")
    data = await file.read()
    t0 = time.perf_counter()
    try:
        logmel, _ = load_audio_to_logmel(data)
        result = infer(logmel)
        latency = time.perf_counter() - t0
        log_event(
            {
                "event": "predict",
                "status": "ok",
                "latency_ms": round(latency * 1000, 2),
                "product": result["product"],
                "quantity": result["quantity"],
                "confidence": result["confidence"],
            }
        )
    except Exception as exc:  # pragma: no cover
        latency = time.perf_counter() - t0
        log_event(
            {
                "event": "predict",
                "status": "error",
                "latency_ms": round(latency * 1000, 2),
                "error": str(exc),
            }
        )
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}")
    return PredictResponse(product=result["product"], quantity=result["quantity"], confidence=result["confidence"])
