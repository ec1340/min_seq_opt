"""Optimization configs, gradient composition, and result summaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GradientConfig:
    norm: bool = False
    norm_type: str = "tensor"
    norm_rescale: bool = False
    cosine_filter: bool = False
    noise: bool = False
    noise_scale: float = 0.05
    noise_min_frac: float = 0.1
    alpha_val: float = 1.0
    beta_val: float = 1.0
    eps: float = 1e-8


@dataclass
class OptimizationConfig:
    num_steps: int = 100
    lr: float = 0.1
    init_bias: float = 5.0
    init_noise_scale: float = 1e-2
    optimization_mode: str = "parallel"
    target_block_steps: int = 25
    antitarget_block_steps: int = 25
    entropy_weight: float = 0.0
    entropy_min_frac: float = 0.0
    entropy_schedule: str = "linear_decay"
    entropy_mode: str = "max"


@dataclass
class ResultSummary:
    final_ligand: str
    final_target: float
    final_antitarget: float
    constraint_margin: float
    feasible: bool
    comparison_score: float


def combine_gradients(
    grad_target,
    grad_anti,
    config: GradientConfig,
    step: int | None = None,
    num_steps: int | None = None,
) -> object:
    """Combine target and anti-target gradients under configurable rules."""
    import torch  # type: ignore[reportMissingImports]

    if config.norm:
        if config.norm_type == "tensor":
            target_scale = grad_target.norm().detach()
            anti_scale = grad_anti.norm().detach()
            grad_target = grad_target / (target_scale + config.eps)
            grad_anti = grad_anti / (anti_scale + config.eps)
        elif config.norm_type == "position":
            target_scale = grad_target.norm(dim=-1, keepdim=True).detach()
            anti_scale = grad_anti.norm(dim=-1, keepdim=True).detach()
            grad_target = grad_target / (target_scale + config.eps)
            grad_anti = grad_anti / (anti_scale + config.eps)
        else:
            raise ValueError(f"Unsupported norm_type: {config.norm_type}")

        if config.norm_rescale:
            grad_target = target_scale * grad_target
            grad_anti = anti_scale * grad_anti

    if config.cosine_filter:
        flat_t = grad_target.reshape(-1)
        flat_a = grad_anti.reshape(-1)
        dot = torch.dot(flat_t, flat_a)
        if dot < 0:
            proj_coeff = dot / (torch.dot(flat_t, flat_t) + config.eps)
            grad_anti = grad_anti - proj_coeff * grad_target

    combined = (-config.alpha_val * grad_target) + (config.beta_val * grad_anti)

    if config.noise:
        if num_steps is not None and num_steps > 1 and step is not None:
            progress = min(max(step / (num_steps - 1), 0.0), 1.0)
            schedule = max(config.noise_min_frac, 1.0 - progress)
        else:
            schedule = 1.0
        combined = combined + (config.noise_scale * schedule) * torch.randn_like(combined)

    return combined


def summarize_result(result: dict, antitarget_threshold: float = -2.0) -> ResultSummary:
    """Compute summary statistics from an optimization run."""
    final_target = float(result["target_preds"][-1])
    final_anti = float(result["antitarget_preds"][-1])
    final_ligand = str(result["candidates_traj"][-1])
    constraint_margin = final_anti - antitarget_threshold
    feasible = final_anti > antitarget_threshold
    penalty = max(0.0, antitarget_threshold - final_anti)
    comparison_score = final_target + 10.0 * penalty
    return ResultSummary(
        final_ligand=final_ligand,
        final_target=final_target,
        final_antitarget=final_anti,
        constraint_margin=constraint_margin,
        feasible=feasible,
        comparison_score=comparison_score,
    )


def run_optimization(*args, **kwargs) -> dict:
    """Backward-compatible import path for the main optimization loop."""
    from .algorithm import run_optimization as _run_optimization

    return _run_optimization(*args, **kwargs)

