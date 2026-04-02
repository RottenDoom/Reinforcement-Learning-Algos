"""Tests for environment/gridworld.py."""

from __future__ import annotations

import numpy as np
import pytest

from environment.gridworld import GridWorldEnv, P, C, policy_eval, INV_ST, TERMINAL_STATE


# ---------------------------------------------------------------------------
# 1. P column sums equal 1.0 for all valid (s, a)
# ---------------------------------------------------------------------------

def test_transition_probs_sum_to_one():
    for s in range(64):
        if s in INV_ST or s == TERMINAL_STATE:
            continue
        for a in range(9):
            total = P[:, s, a].sum()
            assert abs(total - 1.0) < 1e-9, (
                f"P[:, s={s}, a={a}] sums to {total}, expected 1.0"
            )


# ---------------------------------------------------------------------------
# 2. reset() never returns a blocked or terminal state
# ---------------------------------------------------------------------------

def test_reset_never_blocked_or_terminal():
    rng = np.random.default_rng(42)
    env = GridWorldEnv(rng)
    inv_rc = {(rc[0], rc[1]) for rc in [
        [1,1],[1,4],[2,1],[2,4],[3,1],[3,4],[3,5],[3,6],
        [4,1],[4,2],[4,6],[5,2],[5,6],[6,2],[6,6],
    ]}
    for _ in range(500):
        obs = env.reset()
        rc = (int(obs[0]), int(obs[1]))
        assert rc not in inv_rc, f"reset returned blocked state {rc}"
        assert rc != (4, 4), "reset returned terminal state"


# ---------------------------------------------------------------------------
# 3. Stepping from terminal state returns done=True immediately
# ---------------------------------------------------------------------------

def test_terminal_step_is_absorbing():
    rng = np.random.default_rng(0)
    env = GridWorldEnv(rng)
    env.reset_previous((4, 4))
    for a in range(9):
        env.reset_previous((4, 4))
        obs, cost, done, _ = env.step(a)
        assert done, f"done should be True from terminal; action={a}"
        assert int(obs[0]) == 4 and int(obs[1]) == 4


# ---------------------------------------------------------------------------
# 4. Optimal values reproduced by value iteration on P and C to within 1e-2
# ---------------------------------------------------------------------------

def test_value_iteration_matches_optimal():
    from utils.metrics import V_STAR, _VALID_STATES

    optimal_policy = np.zeros((64, 9))
    optimal_policy[:, 8] = 1.0   # placeholder uniform-ish; VI will iterate

    # Run value iteration
    gamma = 0.85
    AvC = np.sum(np.multiply(P, C), axis=0)
    V = np.zeros(64)
    for _ in range(2000):
        tmp1 = np.tile(V, (1, 64 * 9))
        tmp2 = tmp1.reshape((64, 64, 9), order='F')
        Q_sa = AvC + gamma * np.sum(np.multiply(P, tmp2), axis=0)
        V_new = np.min(Q_sa, axis=1)
        if np.max(np.abs(V_new - V)) < 1e-8:
            break
        V = V_new

    for s in _VALID_STATES:
        assert abs(V[s] - V_STAR[s]) < 1e-2, (
            f"V[{s}]={V[s]:.4f} differs from V*={V_STAR[s]:.4f} by "
            f"{abs(V[s] - V_STAR[s]):.4f}"
        )
