"""Main optimization algorithm (core optimization loop)."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from .constants import AMINO_ACIDS, ANTITARGET, TARGET
from .model import AA_TO_IDX, BindingRegressor, encode_from_strings
from .optimization import GradientConfig, OptimizationConfig, combine_gradients

logger = logging.getLogger(__name__)


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


def _emit_progress(message: str, *args: object) -> None:
    if logger.isEnabledFor(logging.INFO):
        logger.info(message, *args)
    else:
        print(message % args, flush=True)


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
    return run_optimization_batch(
        model=model,
        init_ligands=[init_ligand],
        config=config,
        gradient_config=gradient_config,
        target_sequence=target_sequence,
        antitarget_sequence=antitarget_sequence,
        device=device,
    )[0]


def run_optimization_batch(
    model: BindingRegressor,
    init_ligands: list[str],
    config: OptimizationConfig | None = None,
    gradient_config: GradientConfig | None = None,
    target_sequence: str = TARGET,
    antitarget_sequence: str = ANTITARGET,
    device: str | None = None,
) -> list[dict]:
    """Run optimization for multiple ligands in one batched forward pass."""
    config = config or OptimizationConfig()
    gradient_config = gradient_config or GradientConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if not init_ligands:
        raise ValueError("init_ligands must contain at least one sequence.")
    ligand_len = len(init_ligands[0])
    if ligand_len == 0:
        raise ValueError("Ligand sequences must be non-empty.")
    if any(len(ligand) != ligand_len for ligand in init_ligands):
        raise ValueError("All ligands in init_ligands must have the same length for batched optimization.")

    model = model.to(torch_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    num_steps = config.num_steps
    batch_size = len(init_ligands)
    if config.log_progress and config.log_every <= 0:
        raise ValueError("log_every must be > 0 when log_progress is enabled.")

    candidates: list[list[str]] = [[] for _ in range(batch_size)]
    target_preds: list[list[float]] = [[] for _ in range(batch_size)]
    antitarget_preds: list[list[float]] = [[] for _ in range(batch_size)]
    entropy_values: list[list[float]] = [[] for _ in range(batch_size)]
    entropy_weights: list[list[float]] = [[] for _ in range(batch_size)]
    phases: list[list[str]] = [[] for _ in range(batch_size)]

    target_encoded = [encode_from_strings(target_sequence, ligand) for ligand in init_ligands]
    anti_encoded = [encode_from_strings(antitarget_sequence, ligand) for ligand in init_ligands]
    target_ids = torch.stack([ids for ids, _ in target_encoded], dim=0).to(torch_device)
    target_mask = torch.stack([mask for _, mask in target_encoded], dim=0).to(torch_device)
    anti_ids = torch.stack([ids for ids, _ in anti_encoded], dim=0).to(torch_device)
    anti_mask = torch.stack([mask for _, mask in anti_encoded], dim=0).to(torch_device)

    target_lig_start = 3 + len(target_sequence)
    target_lig_end = target_lig_start + ligand_len
    anti_lig_start = 3 + len(antitarget_sequence)
    anti_lig_end = anti_lig_start + ligand_len

    aa_embed_table = model.token_embedding.weight[:20].detach()
    target_base_token_embeds = model.token_embedding(target_ids).detach()
    anti_base_token_embeds = model.token_embedding(anti_ids).detach()
    target_positions = torch.arange(target_ids.shape[1], device=torch_device).unsqueeze(0).expand(batch_size, -1)
    anti_positions = torch.arange(anti_ids.shape[1], device=torch_device).unsqueeze(0).expand(batch_size, -1)
    target_pos_embeds = model.pos_embedding(target_positions).detach()
    anti_pos_embeds = model.pos_embedding(anti_positions).detach()
    sqrt_d_model = math.sqrt(model.d_model)

    logits = torch.stack(
        [
            _initial_logits(
                init_ligand=ligand,
                device=torch_device,
                init_bias=config.init_bias,
                noise_scale=config.init_noise_scale,
            )
            for ligand in init_ligands
        ],
        dim=0,
    ).requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=config.lr)
    run_start = time.perf_counter()
    if config.log_progress:
        _emit_progress(
            "Optimization start: mode=%s steps=%d batch_size=%d lr=%.4g device=%s",
            config.optimization_mode,
            num_steps,
            batch_size,
            config.lr,
            torch_device,
        )

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        probs = F.softmax(logits, dim=-1)
        soft_ligand_embeds = probs @ aa_embed_table
        soft_ligand_embeds.retain_grad()

        target_token_embeds = target_base_token_embeds.clone()
        target_token_embeds[:, target_lig_start:target_lig_end, :] = soft_ligand_embeds

        anti_token_embeds = anti_base_token_embeds.clone()
        anti_token_embeds[:, anti_lig_start:anti_lig_end, :] = soft_ligand_embeds

        joint_token_embeds = torch.cat([target_token_embeds, anti_token_embeds], dim=0) * sqrt_d_model
        joint_pos_embeds = torch.cat([target_pos_embeds, anti_pos_embeds], dim=0)
        joint_mask = torch.cat([target_mask, anti_mask], dim=0)
        joint_x = joint_token_embeds + joint_pos_embeds
        joint_x = model.transformer(joint_x, src_key_padding_mask=joint_mask)
        joint_preds = model.head(joint_x[:, 0, :]).squeeze(-1)
        pred_target = joint_preds[:batch_size]
        pred_anti = joint_preds[batch_size:]

        ones = torch.ones_like(pred_target)
        grad_target = torch.autograd.grad(
            pred_target,
            soft_ligand_embeds,
            grad_outputs=ones,
            retain_graph=True,
        )[0]
        grad_anti = torch.autograd.grad(
            pred_anti,
            soft_ligand_embeds,
            grad_outputs=ones,
            retain_graph=True,
        )[0]

        phase = _phase_for_step(
            step=step,
            optimization_mode=config.optimization_mode,
            target_block_steps=config.target_block_steps,
            antitarget_block_steps=config.antitarget_block_steps,
        )
        for sample_phases in phases:
            sample_phases.append(phase)
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

        per_sample_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean(dim=-1)
        entropy = per_sample_entropy.mean()
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
        for idx, sample_ids in enumerate(step_ids.tolist()):
            step_ligand = "".join(AMINO_ACIDS[i] for i in sample_ids)
            candidates[idx].append(step_ligand)
            target_preds[idx].append(float(pred_target[idx].detach().item()))
            antitarget_preds[idx].append(float(pred_anti[idx].detach().item()))
            entropy_values[idx].append(float(per_sample_entropy[idx].detach().item()))
            entropy_weights[idx].append(float(entropy_weight))

        should_log = config.log_progress and (
            step == 0 or step == num_steps - 1 or ((step + 1) % config.log_every == 0)
        )
        if should_log:
            steps_done = step + 1
            elapsed = time.perf_counter() - run_start
            avg_step_time = elapsed / max(steps_done, 1)
            eta_seconds = max(num_steps - steps_done, 0) * avg_step_time
            throughput = batch_size / max(avg_step_time, 1e-12)
            _emit_progress(
                "Step %d/%d phase=%s elapsed=%.1fs avg_step=%.3fs throughput=%.1f ligands/s eta=%.1fs "
                "target_mean=%.4f anti_mean=%.4f entropy_mean=%.4f",
                steps_done,
                num_steps,
                phase,
                elapsed,
                avg_step_time,
                throughput,
                eta_seconds,
                float(pred_target.mean().detach().item()),
                float(pred_anti.mean().detach().item()),
                float(per_sample_entropy.mean().detach().item()),
            )

    optimization_config_dict = asdict(config)
    gradient_config_dict = asdict(gradient_config)
    return [
        {
            "candidates_traj": candidates[idx],
            "target_preds": target_preds[idx],
            "antitarget_preds": antitarget_preds[idx],
            "entropy_values": entropy_values[idx],
            "entropy_weights": entropy_weights[idx],
            "phases": phases[idx],
            "optimization_config": optimization_config_dict,
            "gradient_config": gradient_config_dict,
        }
        for idx in range(batch_size)
    ]

