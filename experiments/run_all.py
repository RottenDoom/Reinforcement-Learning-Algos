"""Run every algorithm under identical conditions and save results.

Usage
-----
    python experiments/run_all.py                    # all algorithms
    python experiments/run_all.py --algo q_learning  # one algorithm

Output
------
One .npy file per algorithm in experiments/results/:

    {algo_name}_metrics.npy

Each file stores a dict (loaded with np.load(..., allow_pickle=True).item()):
    {
        "signed_err"     : np.ndarray, shape (n_runs, n_episodes),
        "abs_err"        : np.ndarray, shape (n_runs, n_episodes),
        "policy_err"     : np.ndarray, shape (n_runs, n_checkpoints),
        "checkpoint_eps" : np.ndarray, shape (n_checkpoints,),   # 0-based episode idx
        "n_runs"         : int,
        "n_episodes"     : int,
    }

Design notes
------------
- Each run gets a fresh agent and a fresh environment (different seed per run).
- No state is shared between runs.
- Metrics are recorded *after* each episode using the agent's current
  value estimate and policy.
- Policy evaluation is expensive (linear solve on 64x64 system).  We record
  it every ``EVAL_EVERY`` episodes to keep total runtime reasonable.
- Training loop differences are handled by protocol detection (hasattr), not
  subclassing, so the main loop stays readable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Make sure the project root is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.gridworld import GridWorldEnv, INV_ST, TERMINAL_STATE
from utils.metrics import signed_error, abs_error, policy_eval_error
from algorithms.q_learning import QLearningAgent
from algorithms.double_q_learning import DoubleQLearningAgent
from algorithms.sarsa import SARSAAgent
from algorithms.expected_sarsa import ExpectedSARSAAgent
from algorithms.entropy_reg_q import EntropyRegQLearningAgent
from algorithms.reinforce import REINFORCEAgent, REINFORCEWithBaselineAgent
from algorithms.actor_critic import ActorCriticAgent, ActorCriticLambdaAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# How often (in episodes) to compute the expensive policy-eval metric.
EVAL_EVERY: int = 50

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load hyperparameters from default.yaml (or a supplied override path)."""
    if path is None:
        path = Path(__file__).parent / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------
# Each factory is a callable (rng, cfg) -> Agent.
# Adding a new algorithm = adding one entry here; nothing else changes.

def _make_q_learning(rng: np.random.Generator, cfg: dict) -> QLearningAgent:
    return QLearningAgent(
        rng=rng,
        gamma=cfg["environment"]["gamma"],
        omega=cfg["learning_rate"]["omega"],
        epsilon_start=cfg["exploration"]["epsilon_start"],
        epsilon_decay=cfg["exploration"]["epsilon_decay"],
    )


def _make_double_q(rng: np.random.Generator, cfg: dict) -> DoubleQLearningAgent:
    return DoubleQLearningAgent(
        rng=rng,
        gamma=cfg["environment"]["gamma"],
        omega=cfg["learning_rate"]["omega"],
        epsilon_start=cfg["exploration"]["epsilon_start"],
        epsilon_decay=cfg["exploration"]["epsilon_decay"],
    )


def _make_sarsa(rng: np.random.Generator, cfg: dict) -> SARSAAgent:
    return SARSAAgent(
        rng=rng,
        gamma=cfg["environment"]["gamma"],
        omega=cfg["learning_rate"]["omega"],
        epsilon_start=cfg["exploration"]["epsilon_start"],
        epsilon_decay=cfg["exploration"]["epsilon_decay"],
    )


def _make_expected_sarsa(rng: np.random.Generator, cfg: dict) -> ExpectedSARSAAgent:
    return ExpectedSARSAAgent(
        rng=rng,
        gamma=cfg["environment"]["gamma"],
        omega=cfg["learning_rate"]["omega"],
        epsilon_start=cfg["exploration"]["epsilon_start"],
        epsilon_decay=cfg["exploration"]["epsilon_decay"],
    )


def _make_g_learning(rng: np.random.Generator, cfg: dict) -> EntropyRegQLearningAgent:
    return EntropyRegQLearningAgent(
        rng=rng,
        gamma=cfg["environment"]["gamma"],
        omega=cfg["learning_rate"]["omega"],
        epsilon_start=cfg["exploration"]["epsilon_start"],
        epsilon_decay=cfg["exploration"]["epsilon_decay"],
        beta_min=cfg["entropy_q"]["beta_min"],
        beta_max=cfg["entropy_q"]["beta_max"],
        n_episodes=cfg["environment"]["n_episodes"],
    )


def _make_reinforce(rng: np.random.Generator, cfg: dict) -> REINFORCEAgent:
    return REINFORCEAgent(
        rng=rng,
        gamma=cfg["environment"]["gamma"],
        lr_policy=cfg["reinforce"]["lr_policy"],
    )


def _make_reinforce_baseline(
    rng: np.random.Generator, cfg: dict
) -> REINFORCEWithBaselineAgent:
    return REINFORCEWithBaselineAgent(
        rng=rng,
        gamma=cfg["environment"]["gamma"],
        lr_policy=cfg["reinforce"]["lr_policy"],
        lr_baseline=cfg["reinforce"]["lr_baseline"],
    )


def _make_actor_critic(rng: np.random.Generator, cfg: dict) -> ActorCriticAgent:
    return ActorCriticAgent(
        rng=rng,
        gamma=cfg["environment"]["gamma"],
        lr_critic=cfg["actor_critic"]["lr_actor"],   # same lr for both by default
        lr_actor=cfg["actor_critic"]["lr_actor"],
    )


def _make_actor_critic_lambda(
    rng: np.random.Generator, cfg: dict
) -> ActorCriticLambdaAgent:
    return ActorCriticLambdaAgent(
        rng=rng,
        gamma=cfg["environment"]["gamma"],
        lr_critic=cfg["actor_critic"]["lr_actor"],
        lr_actor=cfg["actor_critic"]["lr_actor"],
        lambda_critic=cfg["actor_critic"]["lambda_critic"],
        lambda_actor=cfg["actor_critic"]["lambda_actor"],
    )


# Registry: display_name -> factory
ALGORITHMS: dict[str, Any] = {
    "q_learning":           _make_q_learning,
    "double_q_learning":    _make_double_q,
    "sarsa":                _make_sarsa,
    "expected_sarsa":       _make_expected_sarsa,
    "g_learning":           _make_g_learning,
    "reinforce":            _make_reinforce,
    "reinforce_baseline":   _make_reinforce_baseline,
    "actor_critic":         _make_actor_critic,
    "actor_critic_lambda":  _make_actor_critic_lambda,
}


# ---------------------------------------------------------------------------
# Training-loop helpers
# ---------------------------------------------------------------------------

def _flat(obs: np.ndarray) -> int:
    """Convert (row, col) observation to flat state index."""
    return int(obs[0]) * 8 + int(obs[1])


def run_episode(agent, env: GridWorldEnv, max_steps: int, episode: int) -> None:
    """
    Run one episode with ``agent`` in ``env``.

    Handles three different protocol variants transparently:

    G-learning         : calls ``agent.set_episode(episode)`` before reset.
    SARSA              : reuses ``agent.next_action`` instead of re-selecting.
    REINFORCE variants : calls ``agent.finish_episode()`` after the loop.
    Actor-Critic       : standard online update, no special calls.
    """
    # G-learning needs the episode index to compute beta.
    if hasattr(agent, "set_episode"):
        agent.set_episode(episode)

    agent.reset_episode()
    obs = env.reset()
    s = _flat(obs)

    # Pre-select first action (needed for SARSA's on-policy contract).
    a = agent.select_action(s)

    for _ in range(max_steps):
        obs, cost, done, _ = env.step(a)
        s_prime = _flat(obs)
        agent.update(s, a, cost, s_prime, done)
        s = s_prime
        if done:
            break
        # SARSA reuses the a_prime sampled inside update; all others re-select.
        if hasattr(agent, "next_action") and agent.next_action is not None:
            a = agent.next_action
        else:
            a = agent.select_action(s)

    # REINFORCE variants apply the gradient only after the episode ends.
    if hasattr(agent, "finish_episode"):
        agent.finish_episode()


# ---------------------------------------------------------------------------
# Metric recording helper
# ---------------------------------------------------------------------------

def _record_metrics(
    agent,
    signed_arr: np.ndarray,
    abs_arr: np.ndarray,
    policy_arr: np.ndarray,
    run: int,
    ep: int,
    checkpoint_eps: np.ndarray,
    ckpt_idx_ref: list[int],
) -> None:
    """Record per-episode and checkpoint metrics in-place."""
    V_est = agent.get_value_estimate()
    signed_arr[run, ep] = signed_error(V_est)
    abs_arr[run, ep]    = abs_error(V_est)

    ckpt_idx = ckpt_idx_ref[0]
    if ckpt_idx < len(checkpoint_eps) and ep == checkpoint_eps[ckpt_idx]:
        policy_arr[run, ckpt_idx] = policy_eval_error(agent.get_policy())
        ckpt_idx_ref[0] += 1


def run_algorithm(
    name: str,
    factory,
    cfg: dict,
    base_seed: int = 0,
    eval_every: int = EVAL_EVERY,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run one algorithm for ``n_runs`` independent seeds.

    Returns a result dict (keys: signed_err, abs_err, policy_err,
    checkpoint_eps, n_runs, n_episodes) suitable for saving with np.save.
    """
    n_runs = cfg["environment"]["n_runs"]
    n_eps  = cfg["environment"]["n_episodes"]
    max_st = cfg["environment"]["max_steps_per_episode"]

    checkpoint_eps = np.arange(eval_every - 1, n_eps, eval_every)
    n_ckpt = len(checkpoint_eps)

    signed_arr = np.zeros((n_runs, n_eps))
    abs_arr    = np.zeros((n_runs, n_eps))
    policy_arr = np.zeros((n_runs, n_ckpt))

    t0 = time.time()

    for run in range(n_runs):
        seed      = base_seed + run * 1000
        rng_agent = np.random.default_rng(seed)
        rng_env   = np.random.default_rng(seed + 1)

        agent = factory(rng_agent, cfg)
        env   = GridWorldEnv(rng_env)

        ckpt_ref = [0]   # mutable int in a list so helper can mutate it

        for ep in range(n_eps):
            run_episode(agent, env, max_st, ep)
            _record_metrics(
                agent, signed_arr, abs_arr, policy_arr,
                run, ep, checkpoint_eps, ckpt_ref
            )

        # Force-compute any trailing checkpoint (rounding edge case).
        while ckpt_ref[0] < n_ckpt:
            policy_arr[run, ckpt_ref[0]] = policy_eval_error(agent.get_policy())
            ckpt_ref[0] += 1

        if verbose:
            log.info(
                f"{name:<22}  run {run + 1}/{n_runs}"
                f"  abs_err(final)={abs_arr[run, -1]:.4f}"
                f"  {time.time() - t0:.1f}s"
            )

    return {
        "signed_err":     signed_arr,
        "abs_err":        abs_arr,
        "policy_err":     policy_arr,
        "checkpoint_eps": checkpoint_eps,
        "n_runs":         n_runs,
        "n_episodes":     n_eps,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(algo_filter: str | None = None, config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    selected = (
        {algo_filter: ALGORITHMS[algo_filter]}
        if algo_filter and algo_filter in ALGORITHMS
        else ALGORITHMS
    )
    if algo_filter and algo_filter not in ALGORITHMS:
        log.error(f"Unknown algorithm '{algo_filter}'.  Choices: {list(ALGORITHMS)}")
        sys.exit(1)

    log.info(
        f"Running {len(selected)} algorithm(s)  "
        f"| n_runs={cfg['environment']['n_runs']}  "
        f"| n_episodes={cfg['environment']['n_episodes']}"
    )

    for name, factory in selected.items():
        log.info(f"--- {name} ---")
        results = run_algorithm(name, factory, cfg)
        out_path = RESULTS_DIR / f"{name}_metrics.npy"
        np.save(out_path, results)
        log.info(f"Saved -> {out_path}")

    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL algorithm experiments.")
    parser.add_argument(
        "--algo", type=str, default=None,
        help=f"Single algorithm to run. Choices: {list(ALGORITHMS.keys())}",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file (defaults to configs/default.yaml).",
    )
    args = parser.parse_args()
    main(algo_filter=args.algo, config_path=args.config)
