from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int]):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.block(x)


class CNNTransformerSLU(nn.Module):
    def __init__(self, num_products: int, num_quantities: int, d_model: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1, 64, (2, 2)),
            ConvBlock(64, 128, (1, 2)),
            ConvBlock(128, 256, (1, 2)),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.proj = nn.Conv1d(256, d_model, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn = nn.Linear(d_model, 1)
        self.product_head = nn.Linear(d_model, num_products)
        self.quantity_head = nn.Linear(d_model, num_quantities)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore[override]
        feats = self.cnn(x).squeeze(2)  # B, C, T
        feats = self.proj(feats).transpose(1, 2)  # B, T, D
        enc = self.encoder(feats)
        weights = torch.softmax(self.attn(enc), dim=1)  # B, T, 1
        pooled = (enc * weights).sum(dim=1)
        return {
            "product": self.product_head(pooled),
            "quantity": self.quantity_head(pooled),
        }
