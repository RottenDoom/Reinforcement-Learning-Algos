"""Tests for algorithm implementations."""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.q_learning import QLearningAgent
from environment.gridworld import GridWorldEnv, INV_ST, TERMINAL_STATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_trivial_mdp(agent: QLearningAgent, n_episodes: int = 3000) -> None:
    """
    Drive the agent on a trivial deterministic MDP.

    States : 0 (start), 1 (terminal)
    Actions: 0 -> cost 1.0, goes to state 1 (done)
             1 -> cost 5.0, goes to state 1 (done)

    Q*(0, 0) = 1.0
    Q*(0, 1) = 5.0
    """
    rng = np.random.default_rng(0)
    costs = {0: 1.0, 1: 5.0}

    for _ in range(n_episodes):
        agent.reset_episode()
        a = agent.select_action(0)
        agent.update(s=0, a=a, cost=costs[a], s_prime=1, done=True)


def _run_gridworld(agent: QLearningAgent, n_episodes: int = 100) -> None:
    """Run the agent on the full gridworld for n_episodes."""
    rng = np.random.default_rng(7)
    env = GridWorldEnv(rng)

    for _ in range(n_episodes):
        agent.reset_episode()
        obs = env.reset()
        s = int(obs[0]) * 8 + int(obs[1])

        for _ in range(300):
            a = agent.select_action(s)
            obs, cost, done, _ = env.step(a)
            s_prime = int(obs[0]) * 8 + int(obs[1])
            agent.update(s, a, cost, s_prime, done)
            s = s_prime
            if done:
                break


# ---------------------------------------------------------------------------
# 1. Q-learning converges on a trivial 2-state 2-action MDP
# ---------------------------------------------------------------------------

class TestQLearningConvergence:
    def test_q_values_converge_to_true_values(self):
        rng = np.random.default_rng(42)
        agent = QLearningAgent(rng=rng, gamma=0.85, n_states=2, n_actions=2)
        _run_trivial_mdp(agent, n_episodes=3000)

        # Both actions must have been visited; Q*(0,0)=1.0, Q*(0,1)=5.0
        assert abs(agent.Q[0, 0] - 1.0) < 0.10, (
            f"Q[0,0]={agent.Q[0,0]:.4f}, expected ~1.0"
        )
        assert abs(agent.Q[0, 1] - 5.0) < 0.10, (
            f"Q[0,1]={agent.Q[0,1]:.4f}, expected ~5.0"
        )

    def test_greedy_policy_selects_best_action(self):
        rng = np.random.default_rng(42)
        agent = QLearningAgent(rng=rng, gamma=0.85, n_states=2, n_actions=2)
        _run_trivial_mdp(agent, n_episodes=3000)

        pi = agent.get_policy()
        assert pi[0, 0] == 1.0, "greedy policy should select action 0 (lowest cost)"
        assert pi[0, 1] == 0.0


# ---------------------------------------------------------------------------
# 2. Q table remains finite after 100 episodes on the Gridworld
# ---------------------------------------------------------------------------

class TestQLearningFiniteness:
    def test_q_values_finite_after_gridworld_run(self):
        rng = np.random.default_rng(0)
        agent = QLearningAgent(rng=rng)
        _run_gridworld(agent, n_episodes=100)

        assert np.all(np.isfinite(agent.Q)), (
            "Q table contains NaN or Inf after gridworld run"
        )

    def test_value_estimate_finite(self):
        rng = np.random.default_rng(1)
        agent = QLearningAgent(rng=rng)
        _run_gridworld(agent, n_episodes=100)

        V = agent.get_value_estimate()
        assert V.shape == (64,)
        assert np.all(np.isfinite(V))


# ---------------------------------------------------------------------------
# 3. Policy distributions sum to 1 for all valid states
# ---------------------------------------------------------------------------

class TestQLearningPolicy:
    def test_policy_sums_to_one_for_valid_states(self):
        rng = np.random.default_rng(2)
        agent = QLearningAgent(rng=rng)
        # No training needed — even fresh Q=0 should give a valid policy
        pi = agent.get_policy()

        valid_states = [
            s for s in range(64) if s not in INV_ST and s != TERMINAL_STATE
        ]
        for s in valid_states:
            total = pi[s, :].sum()
            assert abs(total - 1.0) < 1e-9, (
                f"policy at state {s} sums to {total}, expected 1.0"
            )

    def test_policy_shape(self):
        rng = np.random.default_rng(3)
        agent = QLearningAgent(rng=rng)
        pi = agent.get_policy()
        assert pi.shape == (64, 9)

    def test_policy_non_negative(self):
        rng = np.random.default_rng(4)
        agent = QLearningAgent(rng=rng)
        pi = agent.get_policy()
        assert np.all(pi >= 0.0)
