"""Tests for algorithm implementations."""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.double_q_learning import DoubleQLearningAgent
from algorithms.expected_sarsa import ExpectedSARSAAgent
from algorithms.entropy_reg_q import EntropyRegQLearningAgent
from algorithms.reinforce import REINFORCEAgent, REINFORCEWithBaselineAgent
from algorithms.actor_critic import ActorCriticAgent, ActorCriticLambdaAgent
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


# ---------------------------------------------------------------------------
# SARSA helpers
# ---------------------------------------------------------------------------

def _run_trivial_mdp_sarsa(agent: SARSAAgent, n_episodes: int = 3000) -> None:
    """
    Drive a SARSAAgent on the same trivial 2-state 2-action MDP used for
    Q-learning.  SARSA requires reusing the a_prime cached in agent.next_action.
    """
    costs = {0: 1.0, 1: 5.0}

    for _ in range(n_episodes):
        agent.reset_episode()
        a = agent.select_action(0)
        agent.update(s=0, a=a, cost=costs[a], s_prime=1, done=True)


def _run_gridworld_sarsa(agent: SARSAAgent, n_episodes: int = 100) -> None:
    """Run SARSAAgent on the full gridworld, reusing next_action from update."""
    rng = np.random.default_rng(7)
    env = GridWorldEnv(rng)

    for _ in range(n_episodes):
        agent.reset_episode()
        obs = env.reset()
        s = int(obs[0]) * 8 + int(obs[1])
        a = agent.select_action(s)

        for _ in range(300):
            obs, cost, done, _ = env.step(a)
            s_prime = int(obs[0]) * 8 + int(obs[1])
            agent.update(s, a, cost, s_prime, done)
            s = s_prime
            if done:
                break
            # Reuse the a_prime sampled inside update
            a = agent.next_action


# ---------------------------------------------------------------------------
# 4. SARSA converges on a trivial 2-state 2-action MDP
# ---------------------------------------------------------------------------

class TestSARSAConvergence:
    def test_q_values_converge_to_true_values(self):
        rng = np.random.default_rng(42)
        agent = SARSAAgent(rng=rng, gamma=0.85, n_states=2, n_actions=2)
        _run_trivial_mdp_sarsa(agent, n_episodes=3000)

        assert abs(agent.Q[0, 0] - 1.0) < 0.10, (
            f"Q[0,0]={agent.Q[0,0]:.4f}, expected ~1.0"
        )
        assert abs(agent.Q[0, 1] - 5.0) < 0.10, (
            f"Q[0,1]={agent.Q[0,1]:.4f}, expected ~5.0"
        )

    def test_greedy_policy_selects_best_action(self):
        rng = np.random.default_rng(42)
        agent = SARSAAgent(rng=rng, gamma=0.85, n_states=2, n_actions=2)
        _run_trivial_mdp_sarsa(agent, n_episodes=3000)

        pi = agent.get_policy()
        assert pi[0, 0] == 1.0, "greedy policy should select action 0 (lowest cost)"
        assert pi[0, 1] == 0.0


# ---------------------------------------------------------------------------
# 5. SARSA Q table remains finite on the gridworld
# ---------------------------------------------------------------------------

class TestSARSAFiniteness:
    def test_q_values_finite_after_gridworld_run(self):
        rng = np.random.default_rng(0)
        agent = SARSAAgent(rng=rng)
        _run_gridworld_sarsa(agent, n_episodes=100)

        assert np.all(np.isfinite(agent.Q)), (
            "Q table contains NaN or Inf after gridworld run"
        )

    def test_value_estimate_finite(self):
        rng = np.random.default_rng(1)
        agent = SARSAAgent(rng=rng)
        _run_gridworld_sarsa(agent, n_episodes=100)

        V = agent.get_value_estimate()
        assert V.shape == (64,)
        assert np.all(np.isfinite(V))


# ---------------------------------------------------------------------------
# 6. SARSA policy distributions sum to 1 for valid states
# ---------------------------------------------------------------------------

class TestSARSAPolicy:
    def test_policy_sums_to_one_for_valid_states(self):
        rng = np.random.default_rng(2)
        agent = SARSAAgent(rng=rng)
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
        agent = SARSAAgent(rng=rng)
        pi = agent.get_policy()
        assert pi.shape == (64, 9)

    def test_policy_non_negative(self):
        rng = np.random.default_rng(4)
        agent = SARSAAgent(rng=rng)
        pi = agent.get_policy()
        assert np.all(pi >= 0.0)


# ===========================================================================
# Shared gridworld runner for TD methods (value-based, no finish_episode)
# ===========================================================================

def _run_gridworld(agent, n_episodes: int = 100) -> None:
    """
    Drive any value-based Agent on the gridworld for ``n_episodes`` episodes.
    Handles the SARSA next_action protocol automatically when present.
    """
    rng = np.random.default_rng(7)
    env = GridWorldEnv(rng)
    is_sarsa = hasattr(agent, "next_action")

    for _ in range(n_episodes):
        agent.reset_episode()
        obs = env.reset()
        s = int(obs[0]) * 8 + int(obs[1])
        a = agent.select_action(s)

        for _ in range(300):
            obs, cost, done, _ = env.step(a)
            s_prime = int(obs[0]) * 8 + int(obs[1])
            agent.update(s, a, cost, s_prime, done)
            s = s_prime
            if done:
                break
            a = agent.next_action if is_sarsa else agent.select_action(s)


def _policy_sums_to_one(agent, tol: float = 1e-9) -> None:
    """Assert that policy rows for valid non-terminal states sum to 1."""
    valid = [s for s in range(64) if s not in INV_ST and s != TERMINAL_STATE]
    pi = agent.get_policy()
    for s in valid:
        total = pi[s, :].sum()
        assert abs(total - 1.0) < tol, f"state {s}: sum={total}"


# ===========================================================================
# Double Q-learning
# ===========================================================================

def _run_trivial_mdp_dql(agent: DoubleQLearningAgent, n_episodes: int = 3000) -> None:
    """2-state, 2-action MDP: action 0 costs 1.0, action 1 costs 5.0."""
    costs = {0: 1.0, 1: 5.0}
    for _ in range(n_episodes):
        agent.reset_episode()
        a = agent.select_action(0)
        agent.update(s=0, a=a, cost=costs[a], s_prime=1, done=True)


class TestDoubleQLearningConvergence:
    """Double Q-learning should converge to Q* on the trivial MDP."""

    def test_q_values_converge(self):
        rng = np.random.default_rng(42)
        agent = DoubleQLearningAgent(rng=rng, gamma=0.85, n_states=2, n_actions=2)
        _run_trivial_mdp_dql(agent, n_episodes=5000)

        q_mean = (agent.Q_A + agent.Q_B) / 2.0
        assert abs(q_mean[0, 0] - 1.0) < 0.15, f"Q_mean[0,0]={q_mean[0,0]:.4f}"
        assert abs(q_mean[0, 1] - 5.0) < 0.15, f"Q_mean[0,1]={q_mean[0,1]:.4f}"

    def test_greedy_policy_selects_best_action(self):
        rng = np.random.default_rng(42)
        agent = DoubleQLearningAgent(rng=rng, gamma=0.85, n_states=2, n_actions=2)
        _run_trivial_mdp_dql(agent, n_episodes=5000)
        pi = agent.get_policy()
        assert pi[0, 0] == 1.0
        assert pi[0, 1] == 0.0


class TestDoubleQLearningFiniteness:
    def test_q_tables_finite(self):
        rng = np.random.default_rng(0)
        agent = DoubleQLearningAgent(rng=rng)
        _run_gridworld(agent, n_episodes=100)
        assert np.all(np.isfinite(agent.Q_A))
        assert np.all(np.isfinite(agent.Q_B))

    def test_value_estimate_finite(self):
        rng = np.random.default_rng(1)
        agent = DoubleQLearningAgent(rng=rng)
        _run_gridworld(agent, n_episodes=100)
        V = agent.get_value_estimate()
        assert V.shape == (64,)
        assert np.all(np.isfinite(V))


class TestDoubleQLearningPolicy:
    def test_policy_shape_and_non_negative(self):
        rng = np.random.default_rng(2)
        agent = DoubleQLearningAgent(rng=rng)
        pi = agent.get_policy()
        assert pi.shape == (64, 9)
        assert np.all(pi >= 0.0)

    def test_policy_sums_to_one(self):
        rng = np.random.default_rng(3)
        agent = DoubleQLearningAgent(rng=rng)
        _policy_sums_to_one(agent)


# ===========================================================================
# Expected SARSA
# ===========================================================================

def _run_trivial_mdp_esarsa(agent: ExpectedSARSAAgent, n_episodes: int = 3000) -> None:
    costs = {0: 1.0, 1: 5.0}
    for _ in range(n_episodes):
        agent.reset_episode()
        a = agent.select_action(0)
        agent.update(s=0, a=a, cost=costs[a], s_prime=1, done=True)


class TestExpectedSARSAConvergence:
    def test_q_values_converge(self):
        rng = np.random.default_rng(42)
        agent = ExpectedSARSAAgent(rng=rng, gamma=0.85, n_states=2, n_actions=2)
        _run_trivial_mdp_esarsa(agent, n_episodes=3000)
        assert abs(agent.Q[0, 0] - 1.0) < 0.10, f"Q[0,0]={agent.Q[0,0]:.4f}"
        assert abs(agent.Q[0, 1] - 5.0) < 0.10, f"Q[0,1]={agent.Q[0,1]:.4f}"

    def test_greedy_policy_selects_best_action(self):
        rng = np.random.default_rng(42)
        agent = ExpectedSARSAAgent(rng=rng, gamma=0.85, n_states=2, n_actions=2)
        _run_trivial_mdp_esarsa(agent, n_episodes=3000)
        pi = agent.get_policy()
        assert pi[0, 0] == 1.0
        assert pi[0, 1] == 0.0


class TestExpectedSARSAFiniteness:
    def test_q_finite(self):
        rng = np.random.default_rng(0)
        agent = ExpectedSARSAAgent(rng=rng)
        _run_gridworld(agent, n_episodes=100)
        assert np.all(np.isfinite(agent.Q))

    def test_value_estimate_finite(self):
        rng = np.random.default_rng(1)
        agent = ExpectedSARSAAgent(rng=rng)
        _run_gridworld(agent, n_episodes=100)
        V = agent.get_value_estimate()
        assert np.all(np.isfinite(V))


class TestExpectedSARSAPolicy:
    def test_policy_shape_and_non_negative(self):
        agent = ExpectedSARSAAgent(rng=np.random.default_rng(0))
        pi = agent.get_policy()
        assert pi.shape == (64, 9)
        assert np.all(pi >= 0.0)

    def test_policy_sums_to_one(self):
        agent = ExpectedSARSAAgent(rng=np.random.default_rng(1))
        _policy_sums_to_one(agent)


# ===========================================================================
# Entropy-Regularized Q-learning (G-learning)
# ===========================================================================

def _run_gridworld_glearning(agent: EntropyRegQLearningAgent, n_episodes: int = 100) -> None:
    """Run G-learning; set_episode must be called before each reset_episode."""
    rng = np.random.default_rng(7)
    env = GridWorldEnv(rng)

    for ep in range(n_episodes):
        agent.set_episode(ep)
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


class TestGLearningFiniteness:
    def test_q_finite(self):
        rng = np.random.default_rng(0)
        agent = EntropyRegQLearningAgent(rng=rng)
        _run_gridworld_glearning(agent, n_episodes=100)
        assert np.all(np.isfinite(agent.Q))

    def test_value_estimate_finite(self):
        rng = np.random.default_rng(1)
        agent = EntropyRegQLearningAgent(rng=rng)
        _run_gridworld_glearning(agent, n_episodes=100)
        V = agent.get_value_estimate()
        assert V.shape == (64,)
        assert np.all(np.isfinite(V))


class TestGLearningPolicy:
    def test_policy_shape_and_non_negative(self):
        agent = EntropyRegQLearningAgent(rng=np.random.default_rng(0))
        pi = agent.get_policy()
        assert pi.shape == (64, 9)
        assert np.all(pi >= 0.0)

    def test_policy_sums_to_one(self):
        """Softmin policy must be a valid probability distribution."""
        agent = EntropyRegQLearningAgent(rng=np.random.default_rng(1))
        _run_gridworld_glearning(agent, n_episodes=50)
        pi = agent.get_policy()
        valid = [s for s in range(64) if s not in INV_ST and s != TERMINAL_STATE]
        for s in valid:
            total = pi[s, :].sum()
            assert abs(total - 1.0) < 1e-9, f"state {s}: sum={total}"


# ===========================================================================
# REINFORCE and REINFORCE-with-baseline
# ===========================================================================

def _run_gridworld_reinforce(agent, n_episodes: int = 200) -> None:
    """Drive a REINFORCE agent; calls finish_episode after each episode."""
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

        agent.finish_episode()


class TestREINFORCEFiniteness:
    def test_theta_finite(self):
        rng = np.random.default_rng(0)
        agent = REINFORCEAgent(rng=rng)
        _run_gridworld_reinforce(agent, n_episodes=200)
        assert np.all(np.isfinite(agent.theta))

    def test_value_estimate_finite(self):
        rng = np.random.default_rng(1)
        agent = REINFORCEAgent(rng=rng)
        _run_gridworld_reinforce(agent, n_episodes=200)
        V = agent.get_value_estimate()
        assert V.shape == (64,)
        assert np.all(np.isfinite(V))


class TestREINFORCEPolicy:
    def test_policy_shape_and_non_negative(self):
        agent = REINFORCEAgent(rng=np.random.default_rng(0))
        pi = agent.get_policy()
        assert pi.shape == (64, 9)
        assert np.all(pi >= 0.0)

    def test_policy_sums_to_one(self):
        """Softmax policy must be a valid probability distribution."""
        agent = REINFORCEAgent(rng=np.random.default_rng(1))
        _run_gridworld_reinforce(agent, n_episodes=50)
        pi = agent.get_policy()
        valid = [s for s in range(64) if s not in INV_ST and s != TERMINAL_STATE]
        for s in valid:
            total = pi[s, :].sum()
            assert abs(total - 1.0) < 1e-9, f"state {s}: sum={total}"


class TestREINFORCEBaselineFiniteness:
    def test_theta_and_V_finite(self):
        rng = np.random.default_rng(0)
        agent = REINFORCEWithBaselineAgent(rng=rng)
        _run_gridworld_reinforce(agent, n_episodes=200)
        assert np.all(np.isfinite(agent.theta))
        assert np.all(np.isfinite(agent.V))

    def test_policy_sums_to_one(self):
        agent = REINFORCEWithBaselineAgent(rng=np.random.default_rng(2))
        _run_gridworld_reinforce(agent, n_episodes=50)
        pi = agent.get_policy()
        valid = [s for s in range(64) if s not in INV_ST and s != TERMINAL_STATE]
        for s in valid:
            total = pi[s, :].sum()
            assert abs(total - 1.0) < 1e-9, f"state {s}: sum={total}"


# ===========================================================================
# Actor-Critic and AC(lambda)
# ===========================================================================

class TestActorCriticFiniteness:
    def test_V_and_theta_finite(self):
        rng = np.random.default_rng(0)
        agent = ActorCriticAgent(rng=rng)
        _run_gridworld(agent, n_episodes=100)
        assert np.all(np.isfinite(agent.V))
        assert np.all(np.isfinite(agent.theta))

    def test_value_estimate_finite(self):
        rng = np.random.default_rng(1)
        agent = ActorCriticAgent(rng=rng)
        _run_gridworld(agent, n_episodes=100)
        V = agent.get_value_estimate()
        assert V.shape == (64,)
        assert np.all(np.isfinite(V))


class TestActorCriticPolicy:
    def test_policy_shape_and_non_negative(self):
        agent = ActorCriticAgent(rng=np.random.default_rng(0))
        pi = agent.get_policy()
        assert pi.shape == (64, 9)
        assert np.all(pi >= 0.0)

    def test_policy_sums_to_one(self):
        agent = ActorCriticAgent(rng=np.random.default_rng(1))
        _run_gridworld(agent, n_episodes=50)
        _policy_sums_to_one(agent)


class TestActorCriticLambdaFiniteness:
    def test_V_and_theta_finite(self):
        rng = np.random.default_rng(0)
        agent = ActorCriticLambdaAgent(rng=rng)
        _run_gridworld(agent, n_episodes=100)
        assert np.all(np.isfinite(agent.V))
        assert np.all(np.isfinite(agent.theta))

    def test_value_estimate_finite(self):
        rng = np.random.default_rng(1)
        agent = ActorCriticLambdaAgent(rng=rng)
        _run_gridworld(agent, n_episodes=100)
        V = agent.get_value_estimate()
        assert V.shape == (64,)
        assert np.all(np.isfinite(V))


class TestActorCriticLambdaPolicy:
    def test_policy_shape_and_non_negative(self):
        agent = ActorCriticLambdaAgent(rng=np.random.default_rng(0))
        pi = agent.get_policy()
        assert pi.shape == (64, 9)
        assert np.all(pi >= 0.0)

    def test_policy_sums_to_one(self):
        agent = ActorCriticLambdaAgent(rng=np.random.default_rng(1))
        _run_gridworld(agent, n_episodes=50)
        _policy_sums_to_one(agent)
