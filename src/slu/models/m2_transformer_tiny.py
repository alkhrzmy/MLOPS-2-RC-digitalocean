from __future__ import annotations

import torch
import torch.nn as nn


class TransformerTinySLU(nn.Module):
    def __init__(self, num_products: int, num_quantities: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(128, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn = nn.Linear(d_model, 1)
        self.product_head = nn.Linear(d_model, num_products)
        self.quantity_head = nn.Linear(d_model, num_quantities)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore[override]
        feats = self.proj(x)  # expects B, T, 128 input log-mel slices
        enc = self.encoder(feats)
        weights = torch.softmax(self.attn(enc), dim=1)
        pooled = (enc * weights).sum(dim=1)
        return {
            "product": self.product_head(pooled),
            "quantity": self.quantity_head(pooled),
        }
