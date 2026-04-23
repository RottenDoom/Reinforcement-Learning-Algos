"""REINFORCE and REINFORCE-with-baseline (Williams 1992).

Overview
--------
REINFORCE is a Monte-Carlo policy gradient algorithm.  It collects an entire
episode, computes the discounted return G_t at each step, then updates a
tabular softmax policy using the policy gradient theorem.

Because it is episodic, ``update`` does not perform a gradient step
immediately.  Instead it buffers transitions and the actual parameter update
happens in ``finish_episode``.  The training loop must call
``finish_episode`` after ``env`` returns ``done=True``.

Policy parameterisation
-----------------------
Tabular softmax over parameters theta, shape (n_states, n_actions):

    pi(a|s) = softmax(theta[s, :])
            = exp(theta[s, a]) / sum_a' exp(theta[s, a'])

The softmax is computed with max-subtraction for numerical stability.

Policy gradient update (cost minimisation sign convention)
----------------------------------------------------------
The standard REINFORCE gradient *maximises* reward; here we *minimise* cost,
so G_t is a cost return (positive = bad).  We therefore apply gradient
*descent*:

    delta_t = G_t  (REINFORCE) or G_t - V[s_t]  (with baseline)

    theta[s_t, a_t] -= lr * delta_t * (1 - pi(a_t|s_t))
    theta[s_t, a  ] += lr * delta_t * pi(a|s_t)   for a != a_t

Equivalently for the full action vector at s_t:

    grad_log_pi = -I_{a_t} + pi[s_t, :]   (negative because cost minimisation)
    theta[s_t, :] -= lr * delta_t * grad_log_pi
                   = lr * delta_t * (I_{a_t} - pi[s_t, :])

Baseline (REINFORCE-with-baseline only)
---------------------------------------
A state-value baseline V[s] reduces variance without introducing bias.
Updated by a simple TD-like step on the Monte-Carlo return:

    V[s_t] += lr_baseline * (G_t - V[s_t])
"""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


class REINFORCEAgent(Agent):
    """
    Tabular REINFORCE (no baseline) for a cost-minimisation MDP.

    Collects a full episode before updating.  Call ``finish_episode``
    after the episode ends to apply the gradient update.

    Args:
        rng:        seeded numpy Generator for reproducibility.
        gamma:      discount factor.
        lr_policy:  policy parameter learning rate.
        n_states:   number of states (64 for the gridworld).
        n_actions:  number of actions (9 for the gridworld).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        gamma: float = 0.85,
        lr_policy: float = 0.01,
        n_states: int = 64,
        n_actions: int = 9,
    ) -> None:
        self.rng = rng
        self.gamma = gamma
        self.lr_policy = lr_policy
        self.n_states = n_states
        self.n_actions = n_actions

        # Softmax policy parameters; uniform at initialisation (all zeros).
        self.theta: np.ndarray = np.zeros((n_states, n_actions))

        # Episode buffer: list of (s, a, cost) tuples.
        self._trajectory: list[tuple[int, int, float]] = []

        # Cached value estimate (updated after each episode finish).
        self._V_est: np.ndarray = np.zeros(n_states)

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Clear the trajectory buffer at the start of each episode."""
        self._trajectory = []

    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        """Sample an action from the current softmax policy pi(a|state)."""
        pi = _softmax(self.theta[state, :])
        return int(self.rng.choice(self.n_actions, p=pi))

    # ------------------------------------------------------------------
    def update(
        self, s: int, a: int, cost: float, s_prime: int, done: bool
    ) -> None:
        """Buffer the transition.  Gradient update happens in finish_episode."""
        self._trajectory.append((s, a, cost))

    # ------------------------------------------------------------------
    def finish_episode(self) -> None:
        """
        Apply the REINFORCE parameter update using the buffered trajectory.

        Computes the discounted return G_t for each step (backward pass),
        then updates theta using gradient descent on the expected cost.
        """
        T = len(self._trajectory)
        if T == 0:
            return

        # Compute discounted returns (cost minimisation: no sign flip needed).
        G = np.zeros(T)
        G[-1] = self._trajectory[-1][2]
        for t in range(T - 2, -1, -1):
            G[t] = self._trajectory[t][2] + self.gamma * G[t + 1]

        # Gradient descent step for each timestep.
        for t, (s, a, _) in enumerate(self._trajectory):
            pi = _softmax(self.theta[s, :])
            # Gradient of log pi w.r.t. theta[s]: I_a - pi (one-hot minus dist)
            # Descent on cost: theta[s] -= lr * G_t * grad_log_pi
            # Equivalently: theta[s] += lr * G_t * (pi - I_a)   sign flipped
            grad = pi.copy()
            grad[a] -= 1.0          # = pi - I_a  (negative grad for descent)
            self.theta[s, :] += self.lr_policy * G[t] * grad

        # Cache value estimate as mean return per state visited.
        self._V_est = np.zeros(self.n_states)
        counts = np.zeros(self.n_states)
        for t, (s, _, _) in enumerate(self._trajectory):
            self._V_est[s] += G[t]
            counts[s] += 1
        mask = counts > 0
        self._V_est[mask] /= counts[mask]

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """
        Return V_est, shape (n_states,).

        For REINFORCE this is the mean return observed at each state across
        the most recent episode — not a learned critic.
        """
        return self._V_est.copy()

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        """Return the softmax policy matrix, shape (n_states, n_actions)."""
        pi = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            pi[s, :] = _softmax(self.theta[s, :])
        return pi


# ---------------------------------------------------------------------------

class REINFORCEWithBaselineAgent(Agent):
    """
    REINFORCE with a learned state-value baseline for a cost-minimisation MDP.

    The baseline V[s] reduces the variance of the gradient estimate without
    introducing bias.  It is updated by a Monte-Carlo target:

        V[s_t] += lr_baseline * (G_t - V[s_t])

    The policy gradient uses the advantage (G_t - V[s_t]) instead of G_t.

    Args:
        rng:          seeded numpy Generator for reproducibility.
        gamma:        discount factor.
        lr_policy:    policy parameter learning rate.
        lr_baseline:  baseline value function learning rate.
        n_states:     number of states (64 for the gridworld).
        n_actions:    number of actions (9 for the gridworld).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        gamma: float = 0.85,
        lr_policy: float = 0.01,
        lr_baseline: float = 0.1,
        n_states: int = 64,
        n_actions: int = 9,
    ) -> None:
        self.rng = rng
        self.gamma = gamma
        self.lr_policy = lr_policy
        self.lr_baseline = lr_baseline
        self.n_states = n_states
        self.n_actions = n_actions

        self.theta: np.ndarray = np.zeros((n_states, n_actions))
        self.V: np.ndarray = np.zeros(n_states)

        self._trajectory: list[tuple[int, int, float]] = []

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Clear the trajectory buffer at the start of each episode."""
        self._trajectory = []

    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        """Sample an action from the current softmax policy pi(a|state)."""
        pi = _softmax(self.theta[state, :])
        return int(self.rng.choice(self.n_actions, p=pi))

    # ------------------------------------------------------------------
    def update(
        self, s: int, a: int, cost: float, s_prime: int, done: bool
    ) -> None:
        """Buffer the transition.  Gradient update happens in finish_episode."""
        self._trajectory.append((s, a, cost))

    # ------------------------------------------------------------------
    def finish_episode(self) -> None:
        """
        Apply the REINFORCE-with-baseline update using the buffered trajectory.

        1. Compute discounted returns G_t (backward pass).
        2. For each step compute advantage delta_t = G_t - V[s_t].
        3. Update baseline: V[s_t] += lr_baseline * delta_t
        4. Update policy:   theta[s_t] += lr_policy * delta_t * (pi - I_a)
        """
        T = len(self._trajectory)
        if T == 0:
            return

        # Discounted returns (backward pass).
        G = np.zeros(T)
        G[-1] = self._trajectory[-1][2]
        for t in range(T - 2, -1, -1):
            G[t] = self._trajectory[t][2] + self.gamma * G[t + 1]

        # Update baseline and policy.
        for t, (s, a, _) in enumerate(self._trajectory):
            delta = G[t] - self.V[s]
            self.V[s] += self.lr_baseline * delta

            pi = _softmax(self.theta[s, :])
            grad = pi.copy()
            grad[a] -= 1.0          # = pi - I_a
            self.theta[s, :] += self.lr_policy * delta * grad

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """Return the learned baseline V, shape (n_states,)."""
        return self.V.copy()

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        """Return the softmax policy matrix, shape (n_states, n_actions)."""
        pi = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            pi[s, :] = _softmax(self.theta[s, :])
        return pi
