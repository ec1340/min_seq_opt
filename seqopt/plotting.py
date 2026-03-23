"""Plotting helpers for seqopt experiment outputs."""

from __future__ import annotations


def _get_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc
    return plt


def plot_trajectory(result: dict, title: str = "Optimization trajectory") -> None:
    """Plot target and anti-target predictions over optimization steps."""
    plt = _get_pyplot()
    steps = list(range(len(result["target_preds"])))
    plt.figure(figsize=(8, 4.8))
    plt.plot(steps, result["target_preds"], label="target", linewidth=2)
    plt.plot(
        steps,
        result["antitarget_preds"],
        label="anti-target",
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel("Step")
    plt.ylabel("Predicted free energy")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bar(labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    """Plot a simple bar chart for experiment comparisons."""
    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length")
    plt = _get_pyplot()
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_length_curve(
    lengths: list[int],
    means: list[float],
    title: str = "Length vs performance",
) -> None:
    """Plot average score as a function of ligand length."""
    if len(lengths) != len(means):
        raise ValueError("lengths and means must have the same length")
    plt = _get_pyplot()
    plt.figure(figsize=(8, 4.8))
    plt.plot(lengths, means, marker="o", linewidth=2)
    plt.xlabel("Ligand length")
    plt.ylabel("Mean comparison score (lower is better)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_histogram(
    values: list[float],
    bins: int = 15,
    title: str = "Score distribution",
    xlabel: str = "Comparison score",
    ylabel: str = "Count",
) -> None:
    """Plot a histogram for score distributions (e.g., large search runs)."""
    plt = _get_pyplot()
    plt.figure(figsize=(8, 4.8))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

