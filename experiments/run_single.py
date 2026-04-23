"""Single-algorithm run for interactive debugging and quick checks.

Usage
-----
    python experiments/run_single.py --algo q_learning
    python experiments/run_single.py --algo g_learning --runs 3 --episodes 200
    python experiments/run_single.py --algo actor_critic --seed 99 --verbose

Output
------
Prints a per-episode summary table to stdout and optionally saves a results
file to experiments/results/{algo}_debug.npy.

This script is intentionally lightweight — it runs a *single* seed (or the
number you specify) and prints live progress.  Use run_all.py for the full
multi-run experiment.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_all import (
    ALGORITHMS,
    load_config,
    run_episode,
    RESULTS_DIR,
)
from environment.gridworld import GridWorldEnv
from utils.metrics import signed_error, abs_error, policy_eval_error


def run_single(
    algo_name: str,
    n_runs: int = 1,
    n_episodes: int | None = None,
    max_steps: int | None = None,
    seed: int = 0,
    eval_every: int = 50,
    save: bool = False,
    config_path: str | None = None,
) -> None:
    """
    Run one algorithm for ``n_runs`` seeds with live per-episode logging.

    Args:
        algo_name:   Key from the ALGORITHMS registry.
        n_runs:      Number of independent runs (default 1 for debugging).
        n_episodes:  Override config n_episodes.
        max_steps:   Override config max_steps_per_episode.
        seed:        Base RNG seed; run k uses seed + k * 1000.
        eval_every:  How often to print the policy-eval metric (expensive).
        save:        If True, save results to experiments/results/.
        config_path: Optional override for the YAML config path.
    """
    if algo_name not in ALGORITHMS:
        print(f"Unknown algorithm '{algo_name}'.  Choices: {list(ALGORITHMS)}")
        sys.exit(1)

    cfg     = load_config(config_path)
    factory = ALGORITHMS[algo_name]

    n_eps  = n_episodes or cfg["environment"]["n_episodes"]
    max_st = max_steps  or cfg["environment"]["max_steps_per_episode"]

    checkpoint_eps = set(range(eval_every - 1, n_eps, eval_every))

    # Storage (only allocated if save=True, otherwise just print).
    if save:
        ckpt_list  = sorted(checkpoint_eps)
        n_ckpt     = len(ckpt_list)
        signed_arr = np.zeros((n_runs, n_eps))
        abs_arr    = np.zeros((n_runs, n_eps))
        policy_arr = np.zeros((n_runs, n_ckpt))

    print(f"\n{'='*60}")
    print(f"  Algorithm : {algo_name}")
    print(f"  Runs      : {n_runs}")
    print(f"  Episodes  : {n_eps}   max_steps/ep: {max_st}")
    print(f"  Seed base : {seed}    eval_every: {eval_every}")
    print(f"{'='*60}\n")
    print(f"{'ep':>6}  {'signed_err':>12}  {'abs_err':>10}  {'policy_err':>12}  {'t(s)':>6}")
    print("-" * 55)

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        rng_agent = np.random.default_rng(seed + run * 1000)
        rng_env   = np.random.default_rng(seed + run * 1000 + 1)

        agent = factory(rng_agent, cfg)
        env   = GridWorldEnv(rng_env)
        t0    = time.time()

        ckpt_idx = 0
        ckpt_sorted = sorted(checkpoint_eps)

        for ep in range(n_eps):
            run_episode(agent, env, max_st, ep)

            V_est  = agent.get_value_estimate()
            s_err  = signed_error(V_est)
            a_err  = abs_error(V_est)
            p_err_str = "—"

            if ep in checkpoint_eps:
                p_err = policy_eval_error(agent.get_policy())
                p_err_str = f"{p_err:.6f}"
                if save:
                    policy_arr[run, ckpt_idx] = p_err
                    ckpt_idx += 1

            if save:
                signed_arr[run, ep] = s_err
                abs_arr[run, ep]    = a_err

            # Print every eval_every episodes or the last episode.
            if ep in checkpoint_eps or ep == n_eps - 1:
                print(
                    f"{ep + 1:>6}  {s_err:>+12.6f}  {a_err:>10.6f}"
                    f"  {p_err_str:>12}  {time.time() - t0:>6.1f}"
                )

    print(f"\n{'='*60}")
    print(f"Final abs_error (last episode): {a_err:.6f}")
    print(f"{'='*60}\n")

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{algo_name}_debug.npy"
        np.save(out_path, {
            "signed_err":     signed_arr,
            "abs_err":        abs_arr,
            "policy_err":     policy_arr,
            "checkpoint_eps": np.array(ckpt_sorted),
            "n_runs":         n_runs,
            "n_episodes":     n_eps,
        })
        print(f"Results saved -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug run for a single algorithm.")
    parser.add_argument(
        "algo", nargs="?", default="q_learning",
        help=f"Algorithm name. Choices: {list(ALGORITHMS.keys())}",
    )
    parser.add_argument("--algo",      dest="algo_flag", default=None)
    parser.add_argument("--runs",      type=int,   default=1)
    parser.add_argument("--episodes",  type=int,   default=None)
    parser.add_argument("--max-steps", type=int,   default=None)
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--eval-every",type=int,   default=50)
    parser.add_argument("--save",      action="store_true")
    parser.add_argument("--config",    type=str,   default=None)
    args = parser.parse_args()

    # Accept algorithm either as positional arg or --algo flag.
    algo_name = args.algo_flag if args.algo_flag else args.algo

    run_single(
        algo_name=algo_name,
        n_runs=args.runs,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        eval_every=args.eval_every,
        save=args.save,
        config_path=args.config,
    )
