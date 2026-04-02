"""
Metrics for comparing value estimates and policies against the optimal.

All metrics skip invalid states and the terminal state (flat index 36).
"""

from __future__ import annotations

import numpy as np

from environment.gridworld import policy_eval

# Blocked flat indices + terminal
_INV_ST: list[int] = [9, 17, 25, 33, 34, 42, 50, 12, 20, 28, 29, 30, 38, 46, 54]
_TERMINAL: int = 36

_VALID_STATES: list[int] = [
    s for s in range(64) if s not in _INV_ST and s != _TERMINAL
]

# Optimal values from Fox (2016), used as ground truth
_OPTIMAL_GRID = np.array([
    [4.4098, 3.8096, 3.8096, 3.8096, 3.8096, 4.4098, 4.9311, 5.3873],
    [4.4098,      0, 3.0288, 3.0288,      0, 4.4098, 4.9311, 5.3873],
    [4.8806,      0, 2.1455, 2.1455,      0, 5.0336, 5.0336, 5.4940],
    [5.2439,      0, 2.0294, 1.0000,      0,      0,      0, 5.1445],
    [4.9307,      0,      0, 1.0000,      0, 1.0000,      0, 4.7615],
    [4.4761, 4.4761,      0, 1.0000, 1.0000, 1.0000,      0, 4.3139],
    [4.3638, 3.7411,      0, 2.0988, 2.0988, 2.0988,      0, 3.7411],
    [4.3638, 3.7411, 3.0170, 3.0170, 3.0170, 3.0170, 3.0170, 3.7411],
])
V_STAR: np.ndarray = _OPTIMAL_GRID.ravel()   # shape (64,)


def signed_error(V_est: np.ndarray) -> float:
    """
    Mean signed relative error over valid states.

    (1/|valid|) * sum_s  (V_est[s] - V_star[s]) / V_star[s]

    Positive  => over-estimate (optimistic); negative => under-estimate.
    """
    total = sum(
        (V_est[s] - V_STAR[s]) / V_STAR[s]
        for s in _VALID_STATES
    )
    return total / len(_VALID_STATES)


def abs_error(V_est: np.ndarray) -> float:
    """
    Mean absolute relative error over valid states.

    (1/|valid|) * sum_s  |V_est[s] - V_star[s]| / V_star[s]
    """
    total = sum(
        abs(V_est[s] - V_STAR[s]) / V_STAR[s]
        for s in _VALID_STATES
    )
    return total / len(_VALID_STATES)


def policy_eval_error(pi: np.ndarray) -> float:
    """
    Absolute relative error of the value function of policy ``pi``.

    Evaluates pi via the model matrices and compares against V_star.

    Args:
        pi: shape (64, 9), pi(a|s).

    Returns:
        scalar error.
    """
    V_pi = policy_eval(pi)
    total = sum(
        abs(V_pi[s] - V_STAR[s]) / V_STAR[s]
        for s in _VALID_STATES
    )
    return total / len(_VALID_STATES)
