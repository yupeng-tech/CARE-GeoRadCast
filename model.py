# coding: utf-8
# Defines the CARE-GeoRadCast inference model.
# Usage: Imported by run.py.
# Version : CARE-GeoRadCast v1
# Author: Run Ma,Peng Yu

import math

import torch
import torch.nn as nn


class SpatialFrameEncoder(nn.Module):
    """Encodes each input frame into a spatial descriptor."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(22, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or tuple(history.shape[1:]) != (12, 22, 17, 17):
            raise ValueError(f"history must be [B,12,22,17,17], got {tuple(history.shape)}")
        batch, steps, channels, height, width = history.shape
        encoded = self.layers(history.reshape(batch * steps, channels, height, width).float())
        descriptor = encoded.reshape(batch, steps, 64)
        return descriptor.float()


class TemporalEncoder(nn.Module):
    """Encodes the input sequence with a temporal convolution and LSTM."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(86, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, dropout=dropout)

    def forward(self, fusion: torch.Tensor) -> torch.Tensor:
        if fusion.ndim != 3 or tuple(fusion.shape[1:]) != (12, 86):
            raise ValueError(f"fusion must be [B,12,86], got {tuple(fusion.shape)}")
        convolved = self.temporal_conv(fusion.permute(0, 2, 1).float())
        recurrent, _ = self.lstm(convolved.permute(0, 2, 1).float())
        return recurrent.float()


class LeadSpecificDecoder(nn.Module):
    """Eighteen learned queries and eighteen independent 12-step attentions."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.lead_embedding = nn.Embedding(18, 32)
        self.geometry_embedding = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.query_projection = nn.Linear(64, 256)
        self.key_projection = nn.Linear(256, 256)
        self.per_lead_head = nn.Sequential(
            nn.Linear(320, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, hidden_states: torch.Tensor, future_geometry: torch.Tensor) -> torch.Tensor:
        batch = hidden_states.shape[0]
        lead_ids = torch.arange(18, device=hidden_states.device)
        lead = self.lead_embedding(lead_ids).unsqueeze(0).expand(batch, -1, -1)
        geometry = self.geometry_embedding(future_geometry.float())
        queries = self.query_projection(torch.cat((lead, geometry), dim=2))
        keys = self.key_projection(hidden_states)
        logits = torch.einsum("bld,btd->blt", queries, keys) / math.sqrt(256.0)
        attention = torch.softmax(logits, dim=2)
        contexts = torch.einsum("blt,btd->bld", attention, hidden_states)
        features = torch.cat((contexts, lead, geometry), dim=2)
        return self.per_lead_head(features).squeeze(2).float()


class CAREGeoRadCast(nn.Module):
    def __init__(self, dropout: float = 0.15) -> None:
        super().__init__()
        self.spatial_encoder = SpatialFrameEncoder()
        self.temporal_encoder = TemporalEncoder(dropout)
        self.decoder = LeadSpecificDecoder(dropout)

    def forward(self, history: torch.Tensor, future_geometry: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or tuple(history.shape[1:]) != (12, 22, 17, 17):
            raise ValueError("History must have shape [B,12,22,17,17]")
        if tuple(future_geometry.shape) != (history.shape[0], 18, 3):
            raise ValueError("Future solar geometry must have shape [B,18,3]")
        center = history[:, :, :, 8, 8].float()
        spatial = self.spatial_encoder(history.float())
        fusion = torch.cat((center, spatial), dim=2)
        hidden = self.temporal_encoder(fusion)
        return self.decoder(hidden, future_geometry)
