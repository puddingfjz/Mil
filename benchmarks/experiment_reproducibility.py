"""Shared process-level seeding for Artifact experiment entry points.

The historical scripts fixed several local RNGs to zero but left scheduler
sampling on Python's process-global RNG.  The Artifact runner exports one
explicit seed so every process can initialize the same RNG families without
enabling deterministic GPU kernels (which would change timing semantics).
"""

from __future__ import annotations

import os
import random
from typing import Mapping, Optional

import numpy as np


SEED_ENVIRONMENT_VARIABLE = "MIL_EXPERIMENT_SEED"
MAX_SEED = 2**32 - 1


def experiment_seed(
    environment: Optional[Mapping[str, str]] = None,
    default: int = 0,
) -> int:
    """Return the validated Artifact seed, preserving historical seed zero."""

    source = os.environ if environment is None else environment
    raw = source.get(SEED_ENVIRONMENT_VARIABLE)
    if raw is None:
        seed = default
    else:
        try:
            seed = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{SEED_ENVIRONMENT_VARIABLE} must be an integer, got {raw!r}"
            ) from exc
    if isinstance(seed, bool) or not 0 <= seed <= MAX_SEED:
        raise ValueError(
            f"{SEED_ENVIRONMENT_VARIABLE} must be between 0 and {MAX_SEED}, got {seed!r}"
        )
    return seed


def seed_process(seed: Optional[int] = None, seed_torch: bool = False) -> int:
    """Seed Python/NumPy and optionally PyTorch in the current process."""

    selected = experiment_seed() if seed is None else experiment_seed({}, default=seed)
    random.seed(selected)
    np.random.seed(selected)
    if seed_torch:
        import torch

        torch.manual_seed(selected)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(selected)
    return selected
