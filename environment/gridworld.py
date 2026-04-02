"""
GridWorld environment from Fox (2016).

Exports:
    GridWorldEnv  — the interactive environment
    P             — transition probability matrix, shape (64, 64, 9)
    C             — expected cost matrix,          shape (64, 64, 9)
    policy_eval   — model-based policy evaluator
    INV_ST        — list of blocked flat state indices
    TERMINAL_STATE— flat index of the terminal state (36)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

N_STATES: int = 64
N_ACTIONS: int = 9

# Flat indices of blocked (invalid) states
INV_ST: list[int] = [9, 17, 25, 33, 34, 42, 50, 12, 20, 28, 29, 30, 38, 46, 54]

TERMINAL_STATE: int = 36  # (row=4, col=4)

# Action deltas: action_id -> (row_delta, col_delta)
# 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW, 8=Stay
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1,  0),
    1: (-1,  1),
    2: ( 0,  1),
    3: ( 1,  1),
    4: ( 1,  0),
    5: ( 1, -1),
    6: ( 0, -1),
    7: (-1, -1),
    8: ( 0,  0),
}

# Bidirectional state-index maps
_states: dict[int, tuple[int, int]] = {}
_states_pr: dict[tuple[int, int], int] = {}
_k = 0
for _i in range(8):
    for _j in range(8):
        _states[_k] = (_i, _j)
        _states_pr[(_i, _j)] = _k
        _k += 1


# ---------------------------------------------------------------------------
# Transition and cost matrices (computed once at import time)
# ---------------------------------------------------------------------------

def _build_P_C() -> tuple[np.ndarray, np.ndarray]:
    """
    Build P[s', s, a] and C[s', s, a].

    P  — transition probability, shape (64, 64, 9)
    C  — expected cost,          shape (64, 64, 9)
    """
    P_mat = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    C_mat = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    inv_st_set = set(INV_ST)

    for a in range(N_ACTIONS):
        for s in range(N_STATES):
            if s in inv_st_set:
                continue

            s_0, s_1 = _states[s]

            # Terminal state is absorbing with zero cost
            if s == TERMINAL_STATE:
                P_mat[s, s, a] = 1.0
                C_mat[s, s, a] = 0.0
                continue

            # Intended next position after action
            s_pr_0 = s_0 + ACTION_DELTAS[a][0]
            s_pr_1 = s_1 + ACTION_DELTAS[a][1]

            if a == 8:  # Stay — cost 1, no displacement
                s_pr_0, s_pr_1, s_pr = s_0, s_1, s
                C_mat[s_pr, s, a] = 1.0
            elif s_pr_0 < 0 or s_pr_0 > 7 or s_pr_1 < 0 or s_pr_1 > 7:
                # Wall hit: stay in current cell
                s_pr_0, s_pr_1, s_pr = s_0, s_1, s
            else:
                s_pr = _states_pr[(s_pr_0, s_pr_1)]
                if s_pr == TERMINAL_STATE:
                    P_mat[s_pr, s, a] = 1.0
                    C_mat[s_pr, s, a] = 1.0
                    continue
                if s_pr in inv_st_set:
                    s_pr_0, s_pr_1, s_pr = s_0, s_1, s

            s_pr_0_old, s_pr_1_old, s_pr_old = s_pr_0, s_pr_1, s_pr
            p = 0.0

            # Distribute probability mass for stochastic drift (8 directions)
            for i in range(8):
                s_pr_0 = s_pr_0_old + ACTION_DELTAS[i][0]
                s_pr_1 = s_pr_1_old + ACTION_DELTAS[i][1]
                if 0 <= s_pr_0 <= 7 and 0 <= s_pr_1 <= 7:
                    s_pr = _states_pr[(s_pr_0, s_pr_1)]
                    if s_pr in inv_st_set:
                        P_mat[s_pr, s, a] = 0.0
                    else:
                        manhattan = abs(ACTION_DELTAS[i][0]) + abs(ACTION_DELTAS[i][1])
                        prob = 0.05 if manhattan == 1 else 0.025
                        P_mat[s_pr, s, a] = prob
                        p += prob
                        # Drift adds cost only when it displaces from post-action pos
                        C_mat[s_pr, s, a] = 2.0 if s_pr_old != s else 1.0

            # Remaining probability mass stays at post-action position
            P_mat[s_pr_old, s, a] += 1.0 - p
            C_mat[s_pr_old, s, a] = 1.0

    return P_mat, C_mat


P: np.ndarray
C: np.ndarray
P, C = _build_P_C()


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def policy_eval(policy: np.ndarray) -> np.ndarray:
    """
    Evaluate a policy using the model matrices P and C.

    Args:
        policy: shape (64, 9), pi(a|s) for every state.

    Returns:
        V: shape (64,), expected cost-to-go per state.
    """
    gamma = 0.85
    AvC = np.sum(np.multiply(P, C), axis=0)  # shape (64, 9)
    V = np.zeros(N_STATES)
    V_old = np.full(N_STATES, np.inf)

    for _ in range(500):
        tmp1 = np.tile(V, (1, N_STATES * N_ACTIONS))
        tmp2 = tmp1.reshape((N_STATES, N_STATES, N_ACTIONS), order='F')
        tmp = np.sum(np.multiply(P, tmp2), axis=0)
        V = np.sum(np.multiply(policy, (AvC + gamma * tmp)), axis=1)
        norm_V = np.linalg.norm(V)
        if np.linalg.norm(V - V_old) / (norm_V + 1e-12) < 1e-5:
            break
        V_old = V.copy()

    return V


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GridWorldEnv:
    """
    8x8 GridWorld from Fox (2016).

    All stochastic operations go through the ``rng`` passed at construction so
    that runs are fully reproducible.

    Args:
        rng: a seeded ``np.random.Generator``, e.g.
             ``np.random.default_rng(42)``.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.state: np.ndarray | None = None

        # Blocked cells stored as [row, col] lists (matches original step logic)
        self._inv_state: list[list[int]] = [
            [1, 1], [1, 4],
            [2, 1], [2, 4],
            [3, 1], [3, 4], [3, 5], [3, 6],
            [4, 1], [4, 2], [4, 6],
            [5, 2], [5, 6],
            [6, 2], [6, 6],
        ]

    # ------------------------------------------------------------------
    def _is_blocked(self, row: int, col: int) -> bool:
        return [row, col] in self._inv_state

    def _is_out(self, row: int, col: int) -> bool:
        return row < 0 or row > 7 or col < 0 or col > 7

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Apply ``action`` from the current state.

        Returns:
            obs  : np.array([row, col])
            cost : float  (positive = bad; wall hit adds 1000)
            done : bool
            info : dict   (empty)
        """
        row, col = int(self.state[0]), int(self.state[1])
        row_old, col_old = row, col
        done = False
        info: dict = {}
        cost: float = 0.0
        sigma = 0.2

        # Already at terminal — absorbing
        if [row, col] == [4, 4]:
            return np.array([row, col]), cost, True, info

        # --- Intended action ---
        dr, dc = ACTION_DELTAS[action]
        row += dr
        col += dc
        cost += 1.0
        cost += float(self.rng.normal(0.0, sigma))

        # Check terminal after intended move
        if [row, col] == [4, 4]:
            self.state = np.array([row, col])
            return self.state, cost, True, info

        # Revert if out-of-bounds or blocked; add wall-collision penalty
        if self._is_out(row, col) or self._is_blocked(row, col):
            row, col = row_old, col_old
            cost += 1000.0
            cost += float(self.rng.normal(0.0, sigma))

        row_old, col_old = row, col  # post-action position

        # --- Stochastic drift ---
        r = float(self.rng.uniform(0.0, 1.0))
        if   r <= 0.050:            row -= 1
        elif r <= 0.075:            row -= 1; col += 1
        elif r <= 0.125:            col += 1
        elif r <= 0.150:            row += 1; col += 1
        elif r <= 0.200:            row += 1
        elif r <= 0.225:            row += 1; col -= 1
        elif r <= 0.275:            col -= 1
        elif r <= 0.300:            row -= 1; col -= 1
        # r > 0.30: no drift

        # Drift cost applies only when the agent is actually displaced
        if (row, col) != (row_old, col_old):
            cost += 1.0
            cost += float(self.rng.normal(0.0, sigma))

        # Check terminal after drift
        if [row, col] == [4, 4]:
            self.state = np.array([row, col])
            return self.state, cost, True, info

        # Revert invalid drift (undo the extra drift cost)
        if self._is_out(row, col) or self._is_blocked(row, col):
            row, col = row_old, col_old
            cost -= 1.0
            cost -= float(self.rng.normal(0.0, sigma))

        self.state = np.array([row, col])
        return self.state, cost, done, info

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """
        Uniformly sample a valid (non-blocked, non-terminal) start state.

        Returns:
            np.array([row, col])
        """
        while True:
            idx = int(self.rng.integers(N_STATES))
            row, col = _states[idx]
            if [row, col] not in self._inv_state and [row, col] != [4, 4]:
                break
        self.state = np.array([row, col])
        return self.state

    # ------------------------------------------------------------------
    def reset_previous(self, st: tuple[int, int]) -> np.ndarray:
        """Reset to a specific ``(row, col)`` state (used for paired runs)."""
        self.state = np.array(st)
        return self.state
