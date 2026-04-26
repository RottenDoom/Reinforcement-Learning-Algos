"""Policy Mirror Descent with KL-divergence Bregman potential.

Overview
--------
Mirror Descent (MD) replaces the Euclidean gradient step of REINFORCE with a
multiplicative update derived from the KL-divergence Bregman potential.
For a softmax policy the MD update at visited state s_t becomes:

    pi_{new}(a|s) proportional to pi_old(a|s) * exp(-eta * G_t * I[a = a_t])

In log-probability space this is simply an additive shift:

    log pi_{new}[s, a_t] -= eta * G_t
    log pi_{new}[s, :]   -= log Z_s    (renormalise)

Properties
----------
Because the update stays within the probability simplex by construction,
Mirror Descent avoids the numerical overflow that can occur with unconstrained
parameter gradient steps (e.g. large theta values after many wall collisions).
The policy is stored as log-probabilities to keep the multiplicative update
numerically stable.
"""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent


class MirrorDescentAgent(Agent):
    """
    Episodic Policy Mirror Descent for a cost-minimisation MDP.

    Collects a full episode before updating.  Call ``finish_episode``
    after the episode ends to apply the mirror step.

    State is stored as log-probabilities ``_log_pi``, shape (n_states, n_actions).
    The initial policy is uniform: log pi = -log(n_actions).

    Args:
        rng:       seeded numpy Generator for reproducibility.
        gamma:     discount factor.
        lr:        mirror-descent step size eta.
        n_states:  number of states (64 for the gridworld).
        n_actions: number of actions (9 for the gridworld).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        gamma: float = 0.85,
        lr: float = 0.005,
        n_states: int = 64,
        n_actions: int = 9,
    ) -> None:
        self.rng = rng
        self.gamma = gamma
        self.lr = lr
        self.n_states = n_states
        self.n_actions = n_actions

        # Uniform starting policy in log-probability space.
        self._log_pi: np.ndarray = np.full(
            (n_states, n_actions), -np.log(n_actions)
        )
        self._buf: list[tuple[int, int, float]] = []

    # ------------------------------------------------------------------
    def _probs(self, s: int) -> np.ndarray:
        """Numerically stable probability vector for state s."""
        lp = self._log_pi[s, :]
        lp = lp - lp.max()
        p = np.exp(lp)
        return p / p.sum()

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        self._buf = []

    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        return int(self.rng.choice(self.n_actions, p=self._probs(state)))

    # ------------------------------------------------------------------
    def update(
        self, s: int, a: int, cost: float, s_prime: int, done: bool
    ) -> None:
        """Buffer the transition; mirror step applied in finish_episode."""
        self._buf.append((s, a, float(cost)))

    # ------------------------------------------------------------------
    def finish_episode(self) -> None:
        """Apply the KL-proximal multiplicative update for each step."""
        G = 0.0
        for s, a, cost in reversed(self._buf):
            G = cost + self.gamma * G
            # Multiplicative update: decrease log-prob of the taken action.
            self._log_pi[s, a] -= self.lr * G
            # Renormalise to maintain a valid log-probability distribution.
            lp = self._log_pi[s, :]
            log_z = np.log(np.sum(np.exp(lp - lp.max()))) + lp.max()
            self._log_pi[s, :] -= log_z

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """No critic; returns zeros (policy-only method)."""
        return np.zeros(self.n_states)

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        pi = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            pi[s, :] = self._probs(s)
        return pi
