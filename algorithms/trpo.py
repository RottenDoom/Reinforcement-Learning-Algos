"""Trust Region Policy Optimisation for tabular softmax policies.

Overview
--------
TRPO (Schulman et al. 2015) constrains each policy update to stay within a
KL-divergence ball of radius delta around the current policy:

    maximise  E[A(s,a)]
    subject to  mean_s KL(pi_old || pi_new) <= delta

For tabular softmax the natural gradient direction is the advantage A(s,.) and
the optimal step size from the quadratic KL approximation is:

    alpha_max = sqrt(2 * delta / sum_s w_s * A_s^T F_s A_s)

where F_s = diag(pi_s) - pi_s pi_s^T is the Fisher information at state s
and w_s = visit_count_s / total_steps is the state visitation weight.

A backtracking line search then halves alpha until the exact mean KL constraint
is satisfied.

Fisher-vector product
---------------------
F_s v_s = pi_s * v_s - pi_s * (pi_s . v_s)   (O(n) without materialising F_s)

Q estimation
------------
Same running first-visit Monte-Carlo mean as NaturalPGAgent.
"""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _kl_state(theta_old_s: np.ndarray, theta_new_s: np.ndarray) -> float:
    """KL(pi_old || pi_new) for one state (numerically stable)."""
    pi_old = _softmax(theta_old_s)
    pi_new = _softmax(theta_new_s)
    return float(np.sum(pi_old * (np.log(pi_old + 1e-12) - np.log(pi_new + 1e-12))))


def _vFv(pi_s: np.ndarray, v_s: np.ndarray) -> float:
    """v^T F_s v = Var_{pi_s}[v] (Fisher-vector product, O(n))."""
    Fv = pi_s * v_s - pi_s * float(np.dot(pi_s, v_s))
    return float(np.dot(v_s, Fv))


class TRPOAgent(Agent):
    """
    Trust Region Policy Optimisation for a cost-minimisation MDP.

    Per episode:
      1. Estimate Q(s,a) via running first-visit Monte-Carlo mean.
      2. Compute advantage A(s,.) = Q(s,.) - V(s) (natural gradient direction).
      3. Compute alpha_max from the quadratic KL approximation.
      4. Backtracking line search: halve alpha until mean KL <= delta.
      5. Apply theta[s,:] += alpha * A(s,:) for all visited states.

    Attributes:
        kl_history:    list of mean KL values accepted each episode.
        alpha_history: list of step sizes accepted each episode.

    Args:
        rng:               seeded numpy Generator for reproducibility.
        gamma:             discount factor.
        delta:             KL trust-region radius.
        max_alpha:         initial step size before backtracking.
        line_search_steps: maximum number of halvings in the line search.
        n_states:          number of states (64 for the gridworld).
        n_actions:         number of actions (9 for the gridworld).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        gamma: float = 0.85,
        delta: float = 0.01,
        max_alpha: float = 1.0,
        line_search_steps: int = 5,
        n_states: int = 64,
        n_actions: int = 9,
    ) -> None:
        self.rng = rng
        self.gamma = gamma
        self.delta = delta
        self.max_alpha = max_alpha
        self.line_search_steps = line_search_steps
        self.n_states = n_states
        self.n_actions = n_actions

        self.theta: np.ndarray = np.zeros((n_states, n_actions))
        self._Q_hat: np.ndarray = np.zeros((n_states, n_actions))
        self._Q_cnt: np.ndarray = np.zeros((n_states, n_actions))
        self._buf: list[tuple[int, int, float]] = []

        self.kl_history: list[float] = []
        self.alpha_history: list[float] = []

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
        """Buffer the transition; TRPO update applied in finish_episode."""
        self._buf.append((s, a, float(cost)))

    # ------------------------------------------------------------------
    def finish_episode(self) -> None:
        """Compute advantage, determine step size, apply constrained update."""
        if not self._buf:
            return

        # --- step 1: first-visit MC returns ---
        G = 0.0
        ep_Q: dict[tuple[int, int], float] = {}
        for s, a, cost in reversed(self._buf):
            G = cost + self.gamma * G
            ep_Q[(s, a)] = G

        for (s, a), G_t in ep_Q.items():
            self._Q_cnt[s, a] += 1
            self._Q_hat[s, a] += (G_t - self._Q_hat[s, a]) / self._Q_cnt[s, a]

        # --- step 2: advantage and visitation weights ---
        T = len(self._buf)
        visited = list(set(s for s, _, _ in self._buf))
        visit_cnt = {s: sum(1 for ss, _, _ in self._buf if ss == s) for s in visited}
        w = {s: visit_cnt[s] / T for s in visited}

        direction = np.zeros_like(self.theta)
        for s in visited:
            pi_s = _softmax(self.theta[s, :])
            V_s = float(np.dot(pi_s, self._Q_hat[s, :]))
            direction[s, :] = self._Q_hat[s, :] - V_s

        # --- step 3: step size from quadratic KL approximation ---
        dFd = sum(
            w[s] * _vFv(_softmax(self.theta[s, :]), direction[s, :])
            for s in visited
        )
        if dFd < 1e-12:
            self.kl_history.append(0.0)
            self.alpha_history.append(0.0)
            return

        alpha = min(self.max_alpha, float(np.sqrt(2.0 * self.delta / dFd)))

        # --- step 4: backtracking line search ---
        theta_old = self.theta.copy()
        for _ in range(self.line_search_steps):
            theta_try = theta_old.copy()
            for s in visited:
                theta_try[s, :] += alpha * direction[s, :]
            kl = sum(
                w[s] * _kl_state(theta_old[s, :], theta_try[s, :])
                for s in visited
            )
            if kl <= self.delta:
                self.theta = theta_try
                self.kl_history.append(kl)
                self.alpha_history.append(alpha)
                return
            alpha *= 0.5

        # Step not accepted; record zero and leave theta unchanged.
        self.kl_history.append(0.0)
        self.alpha_history.append(0.0)

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
