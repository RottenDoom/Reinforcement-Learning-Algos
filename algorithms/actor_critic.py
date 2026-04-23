"""One-step Actor-Critic and Actor-Critic with eligibility traces AC(lambda).

Overview
--------
Actor-Critic methods combine a *policy* (the actor) with a *value function*
(the critic).  Unlike REINFORCE, they update online (every step) rather than
waiting for the end of the episode, which reduces variance at the cost of
introducing some bias.

Both variants share:
- Tabular softmax actor: parameters theta, shape (n_states, n_actions).
- Tabular linear critic: V, shape (n_states,), representing state values.
- TD error as the learning signal: delta = cost + gamma * V[s'] - V[s].

One-step Actor-Critic
---------------------
Critic update:  V[s]     += lr_critic * delta
Actor update:   theta[s] += lr_actor  * delta * grad_log_pi

grad_log_pi at (s, a_t):  (I_{a_t} - pi[s, :])

With cost minimisation sign convention, descent on cost means:
    theta[s] += lr_actor * delta * (pi[s,:] - I_{a_t})

Wait — let's be precise.  The policy gradient theorem for cost minimisation:

    gradient ascent on -J = gradient descent on J (expected cost)
    => theta[s] -= lr * delta * grad_log_pi
    where grad_log_pi = I_a - pi  (derivative of log-softmax)
    => theta[s] -= lr * delta * (I_a - pi)
    => theta[s] += lr * delta * (pi - I_a)

So the update moves theta away from the taken action when delta > 0 (cost
higher than expected) and toward the taken action when delta < 0.

AC(lambda) — eligibility traces
---------------------------------
Eligibility traces maintain a running record of recently visited (state, action)
pairs, allowing credit assignment to extend multiple steps back in time.

Trace update each step (at state s, action a_t):
    z_v[s]          = gamma * lambda_c * z_v[s] + 1
    z_theta[s, a_t] = gamma * lambda_a * z_theta[s, a_t] + (1 - pi[s, a_t])
    z_theta[s, a  ] = gamma * lambda_a * z_theta[s, a  ] - pi[s, a]  (a != a_t)

Then the global update uses the trace:
    V     += lr_critic * delta * z_v
    theta += lr_actor  * delta * z_theta

Traces are reset to zero at the start of each episode.
"""

from __future__ import annotations

import numpy as np

from algorithms.base import Agent


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


class ActorCriticAgent(Agent):
    """
    One-step (TD) Actor-Critic for a cost-minimisation MDP.

    The critic is a tabular V-function updated by one-step TD error.
    The actor is a tabular softmax policy updated by the same TD error.

    Args:
        rng:           seeded numpy Generator for reproducibility.
        gamma:         discount factor.
        lr_critic:     critic (V) learning rate.
        lr_actor:      actor (theta) learning rate.
        n_states:      number of states (64 for the gridworld).
        n_actions:     number of actions (9 for the gridworld).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        gamma: float = 0.85,
        lr_critic: float = 0.01,
        lr_actor: float = 0.01,
        n_states: int = 64,
        n_actions: int = 9,
    ) -> None:
        self.rng = rng
        self.gamma = gamma
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.n_states = n_states
        self.n_actions = n_actions

        self.V: np.ndarray = np.zeros(n_states)
        self.theta: np.ndarray = np.zeros((n_states, n_actions))

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """No per-episode state to reset for one-step AC."""
        pass

    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        """Sample from the current softmax policy pi(a|state)."""
        pi = _softmax(self.theta[state, :])
        return int(self.rng.choice(self.n_actions, p=pi))

    # ------------------------------------------------------------------
    def update(
        self, s: int, a: int, cost: float, s_prime: int, done: bool
    ) -> None:
        """
        One-step Actor-Critic update.

        Computes the TD error, updates the critic V[s], then updates
        the actor theta[s] for the taken action a.
        """
        # TD error (cost-minimisation: positive delta means cost > expected).
        v_next = 0.0 if done else float(self.V[s_prime])
        delta = cost + self.gamma * v_next - float(self.V[s])

        # Critic update.
        self.V[s] += self.lr_critic * delta

        # Actor update: gradient = pi - I_a (cost minimisation convention).
        pi = _softmax(self.theta[s, :])
        grad = pi.copy()
        grad[a] -= 1.0      # = pi - I_a
        self.theta[s, :] += self.lr_actor * delta * grad

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """Return the critic V, shape (n_states,)."""
        return self.V.copy()

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        """Return the softmax policy matrix, shape (n_states, n_actions)."""
        pi = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            pi[s, :] = _softmax(self.theta[s, :])
        return pi


# ---------------------------------------------------------------------------

class ActorCriticLambdaAgent(Agent):
    """
    Actor-Critic with eligibility traces AC(lambda) for a cost-minimisation MDP.

    Eligibility traces accumulate a decaying record of past (state, action)
    visits, allowing the TD error to propagate credit back multiple steps.

    Separate lambda parameters allow different trace decay rates for the
    critic (lambda_critic) and actor (lambda_actor).

    Args:
        rng:            seeded numpy Generator for reproducibility.
        gamma:          discount factor.
        lr_critic:      critic learning rate.
        lr_actor:       actor learning rate.
        lambda_critic:  trace decay for the critic.
        lambda_actor:   trace decay for the actor.
        n_states:       number of states (64 for the gridworld).
        n_actions:      number of actions (9 for the gridworld).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        gamma: float = 0.85,
        lr_critic: float = 0.01,
        lr_actor: float = 0.01,
        lambda_critic: float = 0.9,
        lambda_actor: float = 0.9,
        n_states: int = 64,
        n_actions: int = 9,
    ) -> None:
        self.rng = rng
        self.gamma = gamma
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.lambda_critic = lambda_critic
        self.lambda_actor = lambda_actor
        self.n_states = n_states
        self.n_actions = n_actions

        self.V: np.ndarray = np.zeros(n_states)
        self.theta: np.ndarray = np.zeros((n_states, n_actions))

        # Eligibility traces; reset each episode.
        self._z_v: np.ndarray = np.zeros(n_states)
        self._z_theta: np.ndarray = np.zeros((n_states, n_actions))

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Zero out eligibility traces at the start of each episode."""
        self._z_v[:] = 0.0
        self._z_theta[:] = 0.0

    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        """Sample from the current softmax policy pi(a|state)."""
        pi = _softmax(self.theta[state, :])
        return int(self.rng.choice(self.n_actions, p=pi))

    # ------------------------------------------------------------------
    def update(
        self, s: int, a: int, cost: float, s_prime: int, done: bool
    ) -> None:
        """
        AC(lambda) update.

        Step 1: Update eligibility traces for state s and action a.
        Step 2: Compute TD error delta.
        Step 3: Apply global update to V and theta using the full trace vectors.
        """
        pi = _softmax(self.theta[s, :])

        # -- Step 1: Update traces -------------------------------------------
        # Standard AC(lambda): decay ALL traces first (every state, every step),
        # then accumulate at the current (s, a).  Decaying only s was wrong —
        # it left other states' traces frozen at large values, causing V to
        # diverge from accumulated unbounded updates.
        gc = self.gamma * self.lambda_critic
        ga = self.gamma * self.lambda_actor

        self._z_v     *= gc          # decay all states
        self._z_v[s]  += 1.0         # accumulate at current state

        # Actor trace follows the cost-minimisation sign convention, which is
        # the NEGATIVE of the reward-maximisation convention.
        #
        # For reward max: accumulate +grad_log_pi = (I_a - pi)
        #   → theta += lr * delta * z  pushes probability UP when delta > 0
        #
        # For cost min:   accumulate -grad_log_pi = (pi - I_a)
        #   → theta += lr * delta * z  pushes probability DOWN when delta > 0
        #   (i.e. the agent learns to avoid actions with unexpectedly high cost)
        #
        # This is consistent with one-step AC which uses grad = (pi - I_a).
        self._z_theta            *= ga          # decay all (state, action) pairs
        self._z_theta[s, :]      += pi          # add pi(a|s) for all actions
        self._z_theta[s, a]      -= 1.0         # subtract 1 for taken action
        # Net effect at (s, a_t):  ga * old + (pi[a_t] - 1)   ← negative, cost-min ✓
        # Net effect at (s, other): ga * old + pi[other]        ← positive, cost-min ✓

        # -- Step 2: TD error -----------------------------------------------
        v_next = 0.0 if done else float(self.V[s_prime])
        delta = cost + self.gamma * v_next - float(self.V[s])

        # -- Step 3: Global update using full trace vectors ------------------
        # z_v and z_theta now carry the correctly decayed credit assignment
        # for all visited (state, action) pairs.
        # delta > 0 means cost exceeded prediction: actor should reduce the
        # probability of the taken action (z_theta[s, a] > 0, so
        # theta[s, a] += lr * delta * positive = pushes softmax away from a).
        self.V     += self.lr_critic * delta * self._z_v
        self.theta += self.lr_actor  * delta * self._z_theta

    # ------------------------------------------------------------------
    def get_value_estimate(self) -> np.ndarray:
        """Return the critic V, shape (n_states,)."""
        return self.V.copy()

    # ------------------------------------------------------------------
    def get_policy(self) -> np.ndarray:
        """Return the softmax policy matrix, shape (n_states, n_actions)."""
        pi = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            pi[s, :] = _softmax(self.theta[s, :])
        return pi
