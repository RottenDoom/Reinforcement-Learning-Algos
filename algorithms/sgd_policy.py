"""SGD-based policy gradient: vanilla SGD, heavy-ball momentum, Nesterov.

Overview
--------
All three variants use the same episodic REINFORCE gradient estimate.
After collecting a full episode the gradient at each visited state s is:

    g[s, :] = sum_{t: s_t=s}  G_t * (pi(.|s) - I_{a_t})

(cost-minimisation sign convention: descent on expected cost)

Variant selection
-----------------
momentum=0                        -> vanilla SGD (identical to REINFORCE)
momentum>0, nesterov=False        -> heavy-ball (Polyak) momentum
momentum>0, nesterov=True         -> Nesterov accelerated gradient

The velocity buffer ``_vel`` persists across episodes; only the transition
buffer ``_buf`` is cleared per episode.

Nesterov update
---------------
Following Bengio et al. (2013) the update is computed as:

    v_new = beta * v_old + g
    theta += lr * (beta * v_new + g - v_old)
           = lr * ((beta^2 - 1) * v_old + (1 + beta) * g)

This is equivalent to the look-ahead form when the current gradient is used
at the look-ahead position (see also Sutton & Barto, Appendix A).
"""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class SGDPolicyAgent(Agent):
    """
    Episodic policy gradient with a configurable first-order optimiser.

    Implements vanilla SGD, heavy-ball momentum, and Nesterov momentum
    via the same class.  Select the variant via ``momentum`` and ``nesterov``.

    Args:
        rng:       seeded numpy Generator for reproducibility.
        gamma:     discount factor.
        lr:        learning rate applied to each gradient step.
        momentum:  momentum coefficient beta (0.0 = vanilla SGD).
        nesterov:  if True, use Nesterov look-ahead form (requires momentum>0).
        n_states:  number of states (64 for the gridworld).
        n_actions: number of actions (9 for the gridworld).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        gamma: float = 0.85,
        lr: float = 0.005,
        momentum: float = 0.0,
        nesterov: bool = False,
        n_states: int = 64,
        n_actions: int = 9,
    ) -> None:
        self.rng = rng
        self.gamma = gamma
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.n_states = n_states
        self.n_actions = n_actions

        self.theta: np.ndarray = np.zeros((n_states, n_actions))
        self._vel: np.ndarray = np.zeros((n_states, n_actions))
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
        """Buffer the transition; gradient applied in finish_episode."""
        self._buf.append((s, a, float(cost)))

    # ------------------------------------------------------------------
    def finish_episode(self) -> None:
        """Compute per-state gradients and apply the chosen SGD update."""
        grads: dict[int, np.ndarray] = {}
        G = 0.0
        for s, a, cost in reversed(self._buf):
            G = cost + self.gamma * G
            pi = _softmax(self.theta[s, :])
            g = pi.copy()
            g[a] -= 1.0                         # (pi - I_a): cost-min convention
            if s not in grads:
                grads[s] = np.zeros(self.n_actions)
            grads[s] += G * g

        for s, g_s in grads.items():
            if self.momentum > 0:
                if self.nesterov:
                    v_prev = self._vel[s, :].copy()
                    self._vel[s, :] = self.momentum * self._vel[s, :] + g_s
                    self.theta[s, :] += self.lr * (
                        self.momentum * self._vel[s, :] + g_s - v_prev
                    )
                else:
                    self._vel[s, :] = self.momentum * self._vel[s, :] + g_s
                    self.theta[s, :] += self.lr * self._vel[s, :]
            else:
                self.theta[s, :] += self.lr * g_s

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """No critic; returns zeros (policy-only method)."""
        return np.zeros(self.n_states)

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        pi = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            pi[s, :] = _softmax(self.theta[s, :])
        return pi
