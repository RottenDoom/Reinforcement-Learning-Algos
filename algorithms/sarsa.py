"""SARSA (on-policy TD control)."""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent
from utils.schedule import lr_schedule


class SARSAAgent(Agent):
    """
    Tabular SARSA for a cost-minimisation MDP.

    On-policy TD control: the next action a_prime is sampled from the
    epsilon-greedy policy *before* the update, so both the behaviour and
    target policies are the same.

    TD target: cost + gamma * Q[s_prime, a_prime]

    Exploration: epsilon-greedy, epsilon resets to ``epsilon_start`` at the
    beginning of each episode and decays by ``epsilon_decay`` after every step.

    Visit counts reset each episode (matching Q-learning reference behaviour).

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

        self.Q: np.ndarray = np.zeros((n_states, n_actions))
        self._visit_count: np.ndarray = np.ones((n_states, n_actions))
        self._epsilon: float = epsilon_start

        # SARSA requires knowing a_prime before calling update.
        # The caller selects a_prime via select_action(s_prime) and passes it
        # in the next call to update.  We cache it here so that reset_episode
        # can clear it safely.
        self._next_action: int | None = None

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset epsilon, visit counts, and cached next-action."""
        self._epsilon = self.epsilon_start
        self._visit_count = np.ones((self.n_states, self.n_actions))
        self._next_action = None

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
        One-step SARSA update, then decay epsilon.

        For SARSA the TD target uses Q[s_prime, a_prime] where a_prime is
        drawn from the *current* epsilon-greedy policy.  The caller must
        already have selected a_prime (via select_action) and passed it as
        the action in the *next* call — but since we need it here, we sample
        it inside update and cache it so the caller can retrieve it via
        ``next_action``.

        Concretely, the training loop should be:

            a = agent.select_action(s)
            obs, cost, done, _ = env.step(a)
            agent.update(s, a, cost, s_prime, done)
            a = agent.next_action   # reuse the sampled a_prime
        """
        self._visit_count[s, a] += 1
        alpha = lr_schedule(int(self._visit_count[s, a]), self.omega)

        if done:
            target = cost
            self._next_action = None
        else:
            a_prime = self.select_action(s_prime)
            self._next_action = a_prime
            target = cost + self.gamma * float(self.Q[s_prime, a_prime])

        self.Q[s, a] += alpha * (target - self.Q[s, a])
        self._epsilon *= self.epsilon_decay

    # ------------------------------------------------------------------
    @property
    def next_action(self) -> int | None:
        """The a_prime sampled during the last update (None if episode ended)."""
        return self._next_action

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """Return V_est[s] = min_a Q[s, a], shape (n_states,)."""
        return np.min(self.Q, axis=1)

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        """
        Return the greedy policy as a one-hot matrix, shape (n_states, n_actions).

        Ties broken by first argmin (numpy default).
        """
        pi = np.zeros((self.n_states, self.n_actions))
        pi[np.arange(self.n_states), np.argmin(self.Q, axis=1)] = 1.0
        return pi
