"""Main optimization algorithm (core optimization loop)."""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F

from .constants import AMINO_ACIDS, ANTITARGET, TARGET
from .model import AA_TO_IDX, BindingRegressor, encode_from_strings
from .optimization import GradientConfig, OptimizationConfig, combine_gradients


def _initial_logits(
    init_ligand: str,
    device: torch.device | str,
    init_bias: float,
    noise_scale: float,
) -> torch.Tensor:
    logits = torch.full((len(init_ligand), len(AMINO_ACIDS)), -init_bias, device=device)
    for pos, residue in enumerate(init_ligand):
        logits[pos, AA_TO_IDX[residue]] = init_bias
    if noise_scale > 0:
        logits = logits + noise_scale * torch.randn_like(logits)
    return logits


def _phase_for_step(
    step: int,
    optimization_mode: str,
    target_block_steps: int,
    antitarget_block_steps: int,
) -> str:
    if optimization_mode == "parallel":
        return "parallel"
    if optimization_mode != "sequential":
        raise ValueError(f"Unsupported optimization_mode: {optimization_mode}")
    cycle_len = target_block_steps + antitarget_block_steps
    if cycle_len <= 0:
        raise ValueError("Sequential optimization requires positive block lengths.")
    return "target_only" if step % cycle_len < target_block_steps else "antitarget_only"


def _scheduled_weight(
    base_weight: float,
    step: int,
    num_steps: int,
    min_frac: float,
    schedule: str,
) -> float:
    if base_weight == 0.0:
        return 0.0
    progress = min(max(step / max(num_steps - 1, 1), 0.0), 1.0)
    if schedule == "constant":
        multiplier = 1.0
    elif schedule == "linear_decay":
        multiplier = max(min_frac, 1.0 - progress)
    elif schedule == "linear_ramp":
        multiplier = min_frac + (1.0 - min_frac) * progress
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")
    return base_weight * multiplier


def run_optimization(
    model: BindingRegressor,
    init_ligand: str,
    config: OptimizationConfig | None = None,
    gradient_config: GradientConfig | None = None,
    target_sequence: str = TARGET,
    antitarget_sequence: str = ANTITARGET,
    device: str | None = None,
) -> dict:
    """Run one optimization trajectory and return per-step outputs."""
    config = config or OptimizationConfig()
    gradient_config = gradient_config or GradientConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = model.to(torch_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    num_steps = config.num_steps
    ligand_len = len(init_ligand)

    candidates: list[str] = []
    target_preds: list[float] = []
    antitarget_preds: list[float] = []
    entropy_values: list[float] = []
    entropy_weights: list[float] = []
    phases: list[str] = []

    target_ids, target_mask = encode_from_strings(target_sequence, init_ligand)
    anti_ids, anti_mask = encode_from_strings(antitarget_sequence, init_ligand)
    target_ids = target_ids.unsqueeze(0).to(torch_device)
    target_mask = target_mask.unsqueeze(0).to(torch_device)
    anti_ids = anti_ids.unsqueeze(0).to(torch_device)
    anti_mask = anti_mask.unsqueeze(0).to(torch_device)

    target_lig_start = 3 + len(target_sequence)
    target_lig_end = target_lig_start + ligand_len
    anti_lig_start = 3 + len(antitarget_sequence)
    anti_lig_end = anti_lig_start + ligand_len

    aa_embed_table = model.token_embedding.weight[:20].detach()
    target_base_token_embeds = model.token_embedding(target_ids).detach()
    anti_base_token_embeds = model.token_embedding(anti_ids).detach()
    target_positions = torch.arange(target_ids.shape[1], device=torch_device).unsqueeze(0)
    anti_positions = torch.arange(anti_ids.shape[1], device=torch_device).unsqueeze(0)
    target_pos_embeds = model.pos_embedding(target_positions).detach()
    anti_pos_embeds = model.pos_embedding(anti_positions).detach()

    logits = _initial_logits(
        init_ligand=init_ligand,
        device=torch_device,
        init_bias=config.init_bias,
        noise_scale=config.init_noise_scale,
    ).requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=config.lr)

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        probs = F.softmax(logits, dim=-1)
        soft_ligand_embeds = probs @ aa_embed_table
        soft_ligand_embeds.retain_grad()

        target_token_embeds = target_base_token_embeds.clone()
        target_token_embeds[0, target_lig_start:target_lig_end, :] = soft_ligand_embeds
        target_token_embeds = target_token_embeds * math.sqrt(model.d_model)
        x_target = target_token_embeds + target_pos_embeds
        x_target = model.transformer(x_target, src_key_padding_mask=target_mask)
        pred_target = model.head(x_target[:, 0, :]).squeeze(-1)

        anti_token_embeds = anti_base_token_embeds.clone()
        anti_token_embeds[0, anti_lig_start:anti_lig_end, :] = soft_ligand_embeds
        anti_token_embeds = anti_token_embeds * math.sqrt(model.d_model)
        x_anti = anti_token_embeds + anti_pos_embeds
        x_anti = model.transformer(x_anti, src_key_padding_mask=anti_mask)
        pred_anti = model.head(x_anti[:, 0, :]).squeeze(-1)

        grad_target = torch.autograd.grad(pred_target, soft_ligand_embeds, retain_graph=True)[0]
        grad_anti = torch.autograd.grad(pred_anti, soft_ligand_embeds, retain_graph=True)[0]

        phase = _phase_for_step(
            step=step,
            optimization_mode=config.optimization_mode,
            target_block_steps=config.target_block_steps,
            antitarget_block_steps=config.antitarget_block_steps,
        )
        phases.append(phase)
        if phase == "parallel":
            combined_grad = combine_gradients(grad_target, grad_anti, gradient_config, step=step, num_steps=num_steps)
        elif phase == "target_only":
            combined_grad = combine_gradients(
                grad_target, torch.zeros_like(grad_anti), gradient_config, step=step, num_steps=num_steps
            )
        else:
            combined_grad = combine_gradients(
                torch.zeros_like(grad_target), grad_anti, gradient_config, step=step, num_steps=num_steps
            )

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        entropy_weight = _scheduled_weight(
            base_weight=config.entropy_weight,
            step=step,
            num_steps=num_steps,
            min_frac=config.entropy_min_frac,
            schedule=config.entropy_schedule,
        )
        if config.entropy_mode == "max":
            entropy_objective = -entropy
        elif config.entropy_mode == "min":
            entropy_objective = entropy
        else:
            raise ValueError(f"Unsupported entropy_mode: {config.entropy_mode}")

        use_entropy_reg = entropy_weight > 0.0
        soft_ligand_embeds.backward(gradient=-combined_grad, retain_graph=use_entropy_reg)
        if use_entropy_reg:
            (entropy_weight * entropy_objective).backward()
        optimizer.step()

        step_ids = logits.argmax(dim=-1)
        step_ligand = "".join(AMINO_ACIDS[i] for i in step_ids.tolist())
        candidates.append(step_ligand)
        target_preds.append(float(pred_target.detach().item()))
        antitarget_preds.append(float(pred_anti.detach().item()))
        entropy_values.append(float(entropy.detach().item()))
        entropy_weights.append(float(entropy_weight))

    return {
        "candidates_traj": candidates,
        "target_preds": target_preds,
        "antitarget_preds": antitarget_preds,
        "entropy_values": entropy_values,
        "entropy_weights": entropy_weights,
        "phases": phases,
        "optimization_config": asdict(config),
        "gradient_config": asdict(gradient_config),
    }

