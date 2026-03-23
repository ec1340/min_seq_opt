"""seqopt: sequence optimization utilities for ligand design."""

from .constants import AMINO_ACIDS, ANTITARGET, TARGET
from .sampling import DEFAULT_WEIGHTED_CHOICES, sample_sequence

__all__ = [
    "AMINO_ACIDS",
    "ANTITARGET",
    "TARGET",
    "DEFAULT_WEIGHTED_CHOICES",
    "sample_sequence",
]

try:
    from .oracle import OracleClient

    __all__.append("OracleClient")
except ModuleNotFoundError:
    pass

try:
    from .algorithm import run_optimization
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
            "summarize_result",
        ]
    )
except ModuleNotFoundError:
    # torch is optional for basic imports (sampling/oracle/constants).
    pass
