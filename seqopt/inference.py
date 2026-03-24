"""Inference helpers for the pretrained binding regressor."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import BindingRegressor, encode_from_strings, encode_sequences


def _resolve_device(device: str | None, model: BindingRegressor | None = None) -> torch.device:
    """Prefer an explicit device, otherwise reuse the model device or fall back to CUDA."""
    if device is not None:
        return torch.device(device)
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_regressor(
    weights_path: str | Path = "regressor.pt",
    device: str | None = None,
) -> BindingRegressor:
    """Load pretrained weights into a BindingRegressor."""
    torch_device = _resolve_device(device)
    model = BindingRegressor()
    state = torch.load(str(weights_path), map_location=torch_device, weights_only=True)
    model.load_state_dict(state)
    model.to(torch_device)
    model.eval()
    return model


def predict_from_strings(
    model: BindingRegressor,
    target_str: str,
    ligand_str: str,
    device: str | None = None,
) -> float:
    """Predict free energy for a target/ligand sequence pair."""
    torch_device = _resolve_device(device, model=model)
    input_ids, padding_mask = encode_from_strings(target_str, ligand_str)
    input_ids = input_ids.unsqueeze(0).to(torch_device)
    padding_mask = padding_mask.unsqueeze(0).to(torch_device)
    with torch.no_grad():
        pred = model(input_ids, padding_mask)
    return float(pred.item())


def predict_batch(
    model: BindingRegressor,
    target_indices: list[int],
    ligand_indices_list: list[list[int]],
    device: str | None = None,
) -> list[float]:
    """Predict free energy for a batch of ligands against one target."""
    torch_device = _resolve_device(device, model=model)
    batch_ids = []
    batch_masks = []
    for lig_idx in ligand_indices_list:
        ids, mask = encode_sequences(target_indices, lig_idx)
        batch_ids.append(ids)
        batch_masks.append(mask)

    input_ids = torch.stack(batch_ids).to(torch_device)
    padding_mask = torch.stack(batch_masks).to(torch_device)
    with torch.no_grad():
        preds = model(input_ids, padding_mask)
    return [float(x) for x in preds.tolist()]

