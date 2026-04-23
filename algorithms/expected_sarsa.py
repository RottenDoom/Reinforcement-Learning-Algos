"""Expected SARSA.

Relationship to SARSA and Q-learning
--------------------------------------
- SARSA (on-policy)    : TD target uses Q[s', a'] where a' is *sampled* from pi.
- Q-learning (off-policy): TD target uses min_a Q[s', a] — the greedy action.
- Expected SARSA       : TD target uses the *expectation* of Q[s', :] under pi.

Expected SARSA is on-policy (pi is the same epsilon-greedy policy used to
act) but has lower variance than SARSA because it averages over all actions
rather than sampling one.  In the limit epsilon->0 it reduces to Q-learning.

TD target
---------
    expected_V[s'] = sum_a pi(a|s') * Q[s', a]
    target = cost + gamma * expected_V[s']

where pi is the *current* epsilon-greedy policy at s'.
"""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent
from utils.schedule import lr_schedule


class ExpectedSARSAAgent(Agent):
    """
    Tabular Expected SARSA for a cost-minimisation MDP.

    The TD target is the expected Q-value under the current epsilon-greedy
    policy, which reduces variance compared to standard SARSA without
    introducing off-policy bias.

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

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset epsilon and visit counts at the start of each episode."""
        self._epsilon = self.epsilon_start
        self._visit_count = np.ones((self.n_states, self.n_actions))

    # ------------------------------------------------------------------
    def _epsilon_greedy_probs(self, state: int) -> np.ndarray:
        """
        Return the probability vector of the current epsilon-greedy policy
        at ``state``, shape (n_actions,).

        The greedy action (argmin Q) gets probability 1 - epsilon +
        epsilon / n_actions; all others get epsilon / n_actions.
        """
        probs = np.full(self.n_actions, self._epsilon / self.n_actions)
        greedy = int(np.argmin(self.Q[state, :]))
        probs[greedy] += 1.0 - self._epsilon
        return probs

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
        Expected SARSA update.

        The expected value at s_prime is computed analytically from the
        epsilon-greedy policy probabilities — no additional sample is drawn.
        """
        self._visit_count[s, a] += 1
        alpha = lr_schedule(int(self._visit_count[s, a]), self.omega)

        if done:
            target = cost
        else:
            probs = self._epsilon_greedy_probs(s_prime)
            expected_v = float(np.dot(probs, self.Q[s_prime, :]))
            target = cost + self.gamma * expected_v

        self.Q[s, a] += alpha * (target - self.Q[s, a])
        self._epsilon *= self.epsilon_decay

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """Return V_est[s] = min_a Q[s, a], shape (n_states,)."""
        return np.min(self.Q, axis=1)

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        """
        Return the greedy policy as a one-hot matrix, shape (n_states, n_actions).

        This is the deterministic policy induced by Q, not the stochastic
        epsilon-greedy policy used during training.
        """
        pi = np.zeros((self.n_states, self.n_actions))
        pi[np.arange(self.n_states), np.argmin(self.Q, axis=1)] = 1.0
        return pi
