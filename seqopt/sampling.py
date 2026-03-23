"""Sampling utilities for ligand sequences."""

from __future__ import annotations

import random

DEFAULT_WEIGHTED_CHOICES: list[tuple[str, float]] = [
    ("W", 10),
    ("Y", 10),
    ("F", 7),
    ("L", 5),
    ("I", 4),
    ("V", 4),
    ("A", 3),
    ("M", 2),
    ("E", 5),
    ("D", 4),
    ("Q", 4),
    ("N", 4),
    ("S", 3),
    ("T", 3),
    ("K", 2),
    ("R", 1),
    ("H", 1),
    ("G", 2),
    ("P", 2),
    ("C", 1),
]
UNIFORM_CHOICES: list[tuple[str, float]] = [
    (letter, 1.0) for letter, _ in DEFAULT_WEIGHTED_CHOICES
]


def sample_sequence(
    length: int,
    rng: random.Random | None = None,
    choices: list[tuple[str, float]] | None = None,
    uniform_weights: bool = False,
) -> str:
    """Sample one amino acid sequence with weighted or uniform choices."""
    if length <= 0:
        raise ValueError("length must be > 0")
    rng = rng or random.Random()
    if choices is None:
        choices = UNIFORM_CHOICES if uniform_weights else DEFAULT_WEIGHTED_CHOICES
    letters = [letter for letter, _ in choices]
    weights = [weight for _, weight in choices]
    return "".join(rng.choices(letters, weights=weights, k=length))


def sample_sequences(
    count: int,
    length: int,
    seed: int | None = None,
    choices: list[tuple[str, float]] | None = None,
    uniform_weights: bool = False,
) -> list[str]:
    """Sample multiple sequences with optional seeding."""
    if count <= 0:
        raise ValueError("count must be > 0")
    rng = random.Random(seed)
    return [
        sample_sequence(
            length=length,
            rng=rng,
            choices=choices,
            uniform_weights=uniform_weights,
        )
        for _ in range(count)
    ]

