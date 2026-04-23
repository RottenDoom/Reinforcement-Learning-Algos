"""Entropy-regularized Q-learning (G-learning, Fox et al. 2016).

Background
----------
G-learning adds a KL-divergence penalty to the Bellman optimality equation,
encouraging the learned policy to stay close to a prior rho(a|s).  The
optimal policy is a softmax over the G-function (analogous to Q) rather than
an argmin/argmax.

The regularised Bellman target replaces ``min_a Q[s', a]`` with a softmax
value:

    V[s'] = (-1/beta) * log( sum_a rho(a|s') * exp(-beta * G[s', a]) )

For a uniform prior rho = 1/n_actions this simplifies to subtracting the
minimum for numerical stability:

    Q_min = min_a G[s', a]
    exp_vals = exp(-beta * (G[s', a] - Q_min))
    V[s'] = (-1/beta) * (-beta * Q_min + log(sum(exp_vals)))

The induced policy is:

    mu(a|s) = exp_vals / sum(exp_vals)

Beta schedule
-------------
Beta starts low (near 0 = maximally stochastic / high-entropy) and increases
linearly toward beta_max (more deterministic / closer to greedy).  This acts
like an annealing schedule: the agent explores early and exploits later.

    beta_t = beta_min + episode * (beta_max - beta_min) / n_episodes

Note: ``episode`` here is the *current episode index* (0-based), passed in
from the training loop via ``set_episode``.
"""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent
from utils.schedule import lr_schedule, beta_schedule


class EntropyRegQLearningAgent(Agent):
    """
    G-learning / Entropy-Regularized Q-learning for a cost-minimisation MDP.

    The G-function plays the role of Q-values; the policy is a softmin
    (softmax over negated costs) parameterized by the inverse-temperature beta.

    Args:
        rng:           seeded numpy Generator for reproducibility.
        gamma:         discount factor.
        omega:         LR exponent; alpha = (1 / visit_count) ^ omega.
        epsilon_start: epsilon-greedy wrapper on top of softmin policy.
        epsilon_decay: multiplicative decay applied after each step.
        beta_min:      initial inverse-temperature (low = high entropy).
        beta_max:      final inverse-temperature after n_episodes.
        n_episodes:    total training episodes (used for beta schedule).
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
        beta_min: float = 0.1,
        beta_max: float = 7.0,
        n_episodes: int = 1500,
        n_states: int = 64,
        n_actions: int = 9,
    ) -> None:
        self.rng = rng
        self.gamma = gamma
        self.omega = omega
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_episodes = n_episodes
        self.n_states = n_states
        self.n_actions = n_actions

        # G-table (analogous to Q-table), initialised to zero.
        self.Q: np.ndarray = np.zeros((n_states, n_actions))
        self._visit_count: np.ndarray = np.ones((n_states, n_actions))
        self._epsilon: float = epsilon_start

        # Current episode index, updated by set_episode() each episode.
        self._current_episode: int = 0

    # ------------------------------------------------------------------
    def set_episode(self, episode: int) -> None:
        """
        Inform the agent of the current episode index so the beta schedule
        can be computed correctly.  Call this before reset_episode.
        """
        self._current_episode = episode

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset epsilon and visit counts at the start of each episode."""
        self._epsilon = self.epsilon_start
        self._visit_count = np.ones((self.n_states, self.n_actions))

    # ------------------------------------------------------------------
    def _beta(self) -> float:
        """Return the current beta from the linear schedule."""
        return beta_schedule(
            self._current_episode, self.n_episodes, self.beta_min, self.beta_max
        )

    # ------------------------------------------------------------------
    def _softmin_value_and_policy(self, state: int) -> tuple[float, np.ndarray]:
        """
        Compute the entropy-regularised value and policy at ``state``.

        Returns
        -------
        V : float
            Softmin value V[state].
        mu : np.ndarray, shape (n_actions,)
            Softmin policy mu(a|state).

        Notes
        -----
        We subtract the minimum G-value before exponentiating to prevent
        numerical overflow (the shift cancels in both V and mu).
        """
        beta = self._beta()
        g = self.Q[state, :]
        g_min = float(np.min(g))
        exp_vals = np.exp(-beta * (g - g_min))
        sum_exp = float(np.sum(exp_vals))
        v = (-1.0 / beta) * (-beta * g_min + np.log(sum_exp))
        mu = exp_vals / sum_exp
        return v, mu

    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        """
        Epsilon-greedy wrapper over the softmin policy mu(a|state).

        With probability epsilon, choose uniformly at random.
        Otherwise, sample from the softmin distribution mu.
        """
        if self.rng.random() < self._epsilon:
            return int(self.rng.integers(self.n_actions))
        _, mu = self._softmin_value_and_policy(state)
        return int(self.rng.choice(self.n_actions, p=mu))

    # ------------------------------------------------------------------
    def update(
        self, s: int, a: int, cost: float, s_prime: int, done: bool
    ) -> None:
        """
        G-learning update.

        TD target replaces the greedy bootstrap with the softmin value V[s']:

            target = cost + gamma * V[s']
            G[s, a] += alpha * (target - G[s, a])
        """
        self._visit_count[s, a] += 1
        alpha = lr_schedule(int(self._visit_count[s, a]), self.omega)

        if done:
            target = cost
        else:
            v_next, _ = self._softmin_value_and_policy(s_prime)
            target = cost + self.gamma * v_next

        self.Q[s, a] += alpha * (target - self.Q[s, a])
        self._epsilon *= self.epsilon_decay

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """
        Return V_est[s] = softmin value at each state, shape (n_states,).

        This is the entropy-regularised value, not min_a G[s, a].
        """
        v = np.zeros(self.n_states)
        for state in range(self.n_states):
            v[state], _ = self._softmin_value_and_policy(state)
        return v

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        """
        Return the softmin policy mu(a|s) for all states,
        shape (n_states, n_actions).
        """
        pi = np.zeros((self.n_states, self.n_actions))
        for state in range(self.n_states):
            _, mu = self._softmin_value_and_policy(state)
            pi[state, :] = mu
        return pi
