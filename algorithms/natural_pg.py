"""Natural Policy Gradient for tabular softmax policies.

Overview
--------
The Natural Policy Gradient (NPG) pre-multiplies the ordinary policy gradient
by the inverse Fisher information matrix F(theta)^{-1}.  For a tabular softmax
policy this has a closed-form simplification: the natural gradient direction at
state s is exactly the advantage function:

    nat_grad[s, :] = F_s^{-1} g_s = A(s, .) = Q(s, .) - V(s)

No explicit Fisher matrix inversion or conjugate-gradient solve is needed
in the tabular case.

Q estimation
------------
Q(s,a) is maintained as a running Monte-Carlo mean over all episodes.  At each
episode only the first occurrence of each (s,a) pair (the earliest G_t) is
used, which corresponds to the first-visit MC estimate.

V(s) is computed analytically from the current policy and Q estimate:
    V(s) = sum_a pi(a|s) * Q(s, a)

Update
------
    theta[s, :] += lr * A(s, .)     for each visited state s
"""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class NaturalPGAgent(Agent):
    """
    Episodic Natural Policy Gradient for a cost-minimisation MDP.

    Q(s,a) is estimated as a running mean of Monte-Carlo returns.
    The natural gradient direction (advantage) is applied once per episode.

    Args:
        rng:       seeded numpy Generator for reproducibility.
        gamma:     discount factor.
        lr:        natural-gradient step size.
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

        self.theta: np.ndarray = np.zeros((n_states, n_actions))
        self._Q_hat: np.ndarray = np.zeros((n_states, n_actions))
        self._Q_cnt: np.ndarray = np.zeros((n_states, n_actions))
        self._buf: list[tuple[int, int, float]] = []

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        self._buf = []

    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        pi = _softmax(self.theta[state, :])
        return int(self.rng.choice(self.n_actions, p=pi))

    # ------------------------------------------------------------------
    def update(
        self, s: int, a: int, cost: float, s_prime: int, done: bool
    ) -> None:
        """Buffer the transition; natural gradient applied in finish_episode."""
        self._buf.append((s, a, float(cost)))

    # ------------------------------------------------------------------
    def finish_episode(self) -> None:
        """Update Q estimates then apply natural gradient (advantage) step."""
        # --- first-visit MC returns ---
        G = 0.0
        ep_Q: dict[tuple[int, int], float] = {}
        for s, a, cost in reversed(self._buf):
            G = cost + self.gamma * G
            ep_Q[(s, a)] = G          # earliest (first-visit) return

        # --- running mean update for Q ---
        for (s, a), G_t in ep_Q.items():
            self._Q_cnt[s, a] += 1
            alpha = 1.0 / self._Q_cnt[s, a]
            self._Q_hat[s, a] += alpha * (G_t - self._Q_hat[s, a])

        # --- natural gradient: theta += lr * A(s, .) ---
        for s in set(s for s, _, _ in self._buf):
            pi_s = _softmax(self.theta[s, :])
            V_s = float(np.dot(pi_s, self._Q_hat[s, :]))
            advantage = self._Q_hat[s, :] - V_s
            self.theta[s, :] += self.lr * advantage

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """Return V(s) = E_{pi}[Q(s,a)] using running Q estimates."""
        V = np.zeros(self.n_states)
        for s in range(self.n_states):
            pi_s = _softmax(self.theta[s, :])
            V[s] = float(np.dot(pi_s, self._Q_hat[s, :]))
        return V

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        pi = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            pi[s, :] = _softmax(self.theta[s, :])
        return pi
