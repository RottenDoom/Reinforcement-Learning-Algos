"""
Learning-rate and beta schedulers shared across algorithms.

All schedulers are stateless functions — callers pass the current counts or
episode index and get back a scalar.
"""

from __future__ import annotations

import numpy as np


def lr_schedule(visit_count: int, omega: float = 0.8) -> float:
    """
    Count-based learning rate.

    alpha_t = (1 / visit_count) ^ omega

    ``visit_count`` is initialised to 1 (not 0) by convention so that the
    first update uses alpha = 1.0 when omega = 1.0.
    """
    return (1.0 / visit_count) ** omega


def beta_schedule(
    episode: int,
    n_episodes: int,
    beta_min: float = 0.1,
    beta_max: float = 7.0,
) -> float:
    """
    Linear beta schedule for entropy-regularised Q-learning.

    beta_t = beta_min + episode * (beta_max - beta_min) / n_episodes
    """
    return beta_min + episode * (beta_max - beta_min) / n_episodes
