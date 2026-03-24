"""seqopt: sequence optimization utilities for ligand design."""

from .constants import AMINO_ACIDS, ANTITARGET, TARGET
from .sampling import DEFAULT_WEIGHTED_CHOICES, sample_sequence, sample_sequences

__all__ = [
    "AMINO_ACIDS",
    "ANTITARGET",
    "TARGET",
    "DEFAULT_WEIGHTED_CHOICES",
    "sample_sequence",
    "sample_sequences",
]

try:
    from .oracle import OracleClient

    __all__.append("OracleClient")
except ModuleNotFoundError:
    pass

try:
    from .plotting import plot_bar, plot_histogram, plot_length_curve, plot_score_summary, plot_trajectory

    __all__.extend(
        [
            "plot_trajectory",
            "plot_bar",
            "plot_score_summary",
            "plot_length_curve",
            "plot_histogram",
        ]
    )
except ModuleNotFoundError:
    # matplotlib is optional for plotting helpers.
    pass

try:
    from .algorithm import run_optimization, run_optimization_batch, run_optimization_chunked
    from .inference import load_regressor, predict_batch, predict_from_strings
    from .model import BindingRegressor, encode_from_strings, encode_sequences
    from .optimization import (
        GradientConfig,
        OptimizationConfig,
        ResultSummary,
        combine_gradients,
        summarize_result,
    )

    __all__.extend(
        [
            "BindingRegressor",
            "GradientConfig",
            "OptimizationConfig",
            "ResultSummary",
            "combine_gradients",
            "encode_sequences",
            "encode_from_strings",
            "load_regressor",
            "predict_from_strings",
            "predict_batch",
            "run_optimization",
            "run_optimization_batch",
            "run_optimization_chunked",
            "summarize_result",
        ]
    )
except ModuleNotFoundError:
    # torch is optional for basic imports (sampling/oracle/constants).
    pass
