"""Q-learning (Watkins 1989) — off-policy TD control."""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent
from utils.schedule import lr_schedule


class QLearningAgent(Agent):
    """
    Tabular Q-learning for a cost-minimisation MDP.

    TD target: cost + gamma * min_a Q[s', a]

    Exploration: epsilon-greedy, epsilon resets to ``epsilon_start`` at the
    beginning of each episode and decays by ``epsilon_decay`` after every step.

    Visit counts reset each episode (matching the reference notebook).

    Args:
        rng:           seeded numpy Generator for reproducibility.
        gamma:         discount factor.
        omega:         LR exponent; alpha = (1 / visit_count) ^ omega.
        epsilon_start: initial exploration rate per episode.
        epsilon_decay: multiplicative decay applied after each step.
        n_states:      number of states (64 for the gridworld).
        n_actions:     number of actions (9 for the gridworld).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        gamma: float = 0.85,
        omega: float = 0.8,
        epsilon_start: float = 0.4,
        epsilon_decay: float = 0.9,
        n_states: int = 64,
        n_actions: int = 9,
    ) -> None:
        self.rng = rng
        self.gamma = gamma
        self.omega = omega
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.n_states = n_states
        self.n_actions = n_actions

        # Q table and per-episode visit counts (both initialised to 0 / 1)
        self.Q: np.ndarray = np.zeros((n_states, n_actions))
        self._visit_count: np.ndarray = np.ones((n_states, n_actions))
        self._epsilon: float = epsilon_start

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset epsilon and visit counts at the start of each episode."""
        self._epsilon = self.epsilon_start
        self._visit_count = np.ones((self.n_states, self.n_actions))

    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection (greedy = argmin Q)."""
        if self.rng.random() < self._epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmin(self.Q[state, :]))

    # ------------------------------------------------------------------
    def update(
        self, s: int, a: int, cost: float, s_prime: int, done: bool
    ) -> None:
        """
        One-step Q-learning update, then decay epsilon.

        The visit count is incremented before computing alpha so that the
        first update in an episode uses alpha = (1/2)^omega (count: 1 -> 2).
        """
        self._visit_count[s, a] += 1
        alpha = lr_schedule(int(self._visit_count[s, a]), self.omega)

        target = cost if done else cost + self.gamma * float(np.min(self.Q[s_prime, :]))
        self.Q[s, a] += alpha * (target - self.Q[s, a])

        self._epsilon *= self.epsilon_decay

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """Return V_est[s] = min_a Q[s, a], shape (64,)."""
        return np.min(self.Q, axis=1)

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        """
        Return the greedy policy as a one-hot matrix, shape (n_states, n_actions).

        Ties are broken by taking the first argmin (numpy default).
        """
        pi = np.zeros((self.n_states, self.n_actions))
        pi[np.arange(self.n_states), np.argmin(self.Q, axis=1)] = 1.0
        return pi
