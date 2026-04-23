"""Double Q-learning (van Hasselt 2010).

Motivation
----------
Standard Q-learning suffers from maximisation bias: using the same samples to
both select and evaluate the greedy action tends to overestimate Q-values.
Double Q-learning decouples these two steps by maintaining two independent
tables Q_A and Q_B.

Update rule (selected uniformly at random each step)
-----------------------------------------------------
With probability 0.5 update Q_A:
    a* = argmin Q_A[s_prime, :]       # action selection
    target = cost + gamma * Q_B[s_prime, a*]   # action evaluation
    Q_A[s, a] += alpha * (target - Q_A[s, a])

Otherwise update Q_B symmetrically (swap roles of A and B).

Exploration policy uses the average (Q_A + Q_B) / 2 so both tables contribute
to behaviour.
"""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent
from utils.schedule import lr_schedule


class DoubleQLearningAgent(Agent):
    """
    Double Q-learning for a cost-minimisation MDP.

    Two Q-tables, Q_A and Q_B, are maintained.  On each step one is chosen at
    random (50/50) to be updated; the other evaluates the target.  Exploration
    uses the mean of both tables.

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

        # Two independent Q-tables, both initialised to zero.
        self.Q_A: np.ndarray = np.zeros((n_states, n_actions))
        self.Q_B: np.ndarray = np.zeros((n_states, n_actions))

        # Separate visit counts per table so each table's learning rate is
        # governed only by how often it was the one being updated.
        self._count_A: np.ndarray = np.ones((n_states, n_actions))
        self._count_B: np.ndarray = np.ones((n_states, n_actions))

        self._epsilon: float = epsilon_start

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset epsilon and visit counts at the start of each episode."""
        self._epsilon = self.epsilon_start
        self._count_A = np.ones((self.n_states, self.n_actions))
        self._count_B = np.ones((self.n_states, self.n_actions))

    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        """Epsilon-greedy over the mean of Q_A and Q_B."""
        if self.rng.random() < self._epsilon:
            return int(self.rng.integers(self.n_actions))
        q_mean = (self.Q_A[state, :] + self.Q_B[state, :]) / 2.0
        return int(np.argmin(q_mean))

    # ------------------------------------------------------------------
    def update(
        self, s: int, a: int, cost: float, s_prime: int, done: bool
    ) -> None:
        """
        Randomly choose which table to update; the other evaluates the target.

        The random coin flip uses ``self.rng`` so runs are reproducible given
        the same seed.
        """
        if self.rng.random() < 0.5:
            # Update Q_A: select action from Q_A, evaluate with Q_B.
            self._count_A[s, a] += 1
            alpha = lr_schedule(int(self._count_A[s, a]), self.omega)
            if done:
                target = cost
            else:
                a_star = int(np.argmin(self.Q_A[s_prime, :]))
                target = cost + self.gamma * float(self.Q_B[s_prime, a_star])
            self.Q_A[s, a] += alpha * (target - self.Q_A[s, a])
        else:
            # Update Q_B: select action from Q_B, evaluate with Q_A.
            self._count_B[s, a] += 1
            alpha = lr_schedule(int(self._count_B[s, a]), self.omega)
            if done:
                target = cost
            else:
                a_star = int(np.argmin(self.Q_B[s_prime, :]))
                target = cost + self.gamma * float(self.Q_A[s_prime, a_star])
            self.Q_B[s, a] += alpha * (target - self.Q_B[s, a])

        self._epsilon *= self.epsilon_decay

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """Return V_est[s] = min_a ((Q_A + Q_B) / 2)[s, :], shape (n_states,)."""
        q_mean = (self.Q_A + self.Q_B) / 2.0
        return np.min(q_mean, axis=1)

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        """
        Return the greedy policy over the mean Q as a one-hot matrix,
        shape (n_states, n_actions).
        """
        pi = np.zeros((self.n_states, self.n_actions))
        q_mean = (self.Q_A + self.Q_B) / 2.0
        pi[np.arange(self.n_states), np.argmin(q_mean, axis=1)] = 1.0
        return pi
