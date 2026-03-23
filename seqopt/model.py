"""Transformer regressor model and sequence encoding helpers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .constants import AMINO_ACIDS

AA_TO_IDX = {c: i for i, c in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: c for i, c in enumerate(AMINO_ACIDS)}

ALPHABET_SIZE = 20
PAD_TOKEN = 20
CLS_TOKEN = 21
TGT_TOKEN = 22
LIG_TOKEN = 23
VOCAB_SIZE = 24
MAX_POS = 604


class BindingRegressor(nn.Module):
    """Transformer-based regressor for binding free energy."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN)
        self.pos_embedding = nn.Embedding(MAX_POS, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        cls_out = x[:, 0, :]
        return self.head(cls_out).squeeze(-1)


def encode_sequences(
    target_indices: list[int] | torch.Tensor,
    ligand_indices: list[int] | torch.Tensor,
    max_len: int = MAX_POS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode (target, ligand) indices into model inputs."""
    if isinstance(target_indices, torch.Tensor):
        target_indices = target_indices.tolist()
    if isinstance(ligand_indices, torch.Tensor):
        ligand_indices = ligand_indices.tolist()

    tokens = [CLS_TOKEN, TGT_TOKEN] + target_indices + [LIG_TOKEN] + ligand_indices
    seq_len = len(tokens)
    if seq_len > max_len:
        tokens = tokens[:max_len]
        seq_len = max_len

    padding_length = max_len - seq_len
    input_ids = torch.tensor(tokens + [PAD_TOKEN] * padding_length, dtype=torch.long)
    padding_mask = torch.zeros(max_len, dtype=torch.bool)
    padding_mask[seq_len:] = True
    return input_ids, padding_mask


def encode_from_strings(
    target_str: str,
    ligand_str: str,
    max_len: int = MAX_POS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode amino acid strings into model inputs."""
    target_indices = [AA_TO_IDX[c] for c in target_str]
    ligand_indices = [AA_TO_IDX[c] for c in ligand_str]
    return encode_sequences(target_indices, ligand_indices, max_len=max_len)

