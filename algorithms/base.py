"""Abstract base class that every algorithm must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    """
    Shared interface for all RL agents in this project.

    State is always passed as a flat integer index (row * 8 + col).
    """

    @abstractmethod
    def select_action(self, state: int) -> int:
        """Return an action index in [0, 8] for the given flat state."""
        ...

    @abstractmethod
    def update(self, s: int, a: int, cost: float, s_prime: int, done: bool) -> None:
        """Process one environment transition and update internal tables."""
        ...

    @abstractmethod
    def get_value_estimate(self) -> np.ndarray:
        """
        Return V_est, shape (64,).

        Convention by algorithm type:
          value-based    : min_a Q[s, a]
          G-learning     : softmax value V_er[s]
          policy gradient: critic V[s]
        """
        ...

    @abstractmethod
    def get_policy(self) -> np.ndarray:
        """Return pi(a|s), shape (64, 9)."""
        ...

    @abstractmethod
    def reset_episode(self) -> None:
        """Called at the start of each episode; reset traces, epsilon, etc."""
        ...
