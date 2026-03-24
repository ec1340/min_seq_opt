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


def plot_score_summary(
    labels: list[str],
    means: list[float],
    stds: list[float] | None = None,
    title: str = "Configuration comparison",
    counts: list[int] | None = None,
    sort_by_score: bool = True,
) -> None:
    """Plot score comparison with uncertainty and sample counts."""
    n = len(labels)
    if len(means) != n:
        raise ValueError("labels and means must have the same length")
    if stds is not None and len(stds) != n:
        raise ValueError("stds must have the same length as labels")
    if counts is not None and len(counts) != n:
        raise ValueError("counts must have the same length as labels")
    if n == 0:
        raise ValueError("labels must be non-empty")

    order = sorted(range(n), key=lambda i: means[i]) if sort_by_score else list(range(n))
    labels_ord = [labels[i] for i in order]
    means_ord = [means[i] for i in order]
    stds_ord = [stds[i] for i in order] if stds is not None else None
    counts_ord = [counts[i] for i in order] if counts is not None else None

    plt = _get_pyplot()
    y_pos = list(range(n))
    fig, ax_score = plt.subplots(1, 1, figsize=(8.8, max(4.8, 0.55 * n + 2.2)))

    ax_score.barh(y_pos, means_ord, xerr=stds_ord, capsize=4)
    ax_score.set_yticks(y_pos)
    ax_score.set_yticklabels(labels_ord)
    ax_score.invert_yaxis()
    ax_score.set_xlabel("Comparison score (lower is better)")
    ax_score.set_title(title)
    ax_score.grid(axis="x", alpha=0.3)

    if counts_ord is not None:
        x_span = max(abs(v) for v in means_ord) or 1.0
        text_offset = 0.03 * x_span
        for yi, (xval, count) in enumerate(zip(means_ord, counts_ord)):
            ax_score.text(xval + text_offset, yi, f"n={count}", va="center", fontsize=9)

    fig.tight_layout()
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

