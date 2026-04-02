# LLM Usage Log

All LLM assistance for this project is documented here as required by the project brief.

---

## 2026-04-02 — Environment implementation

**What was requested:**
Port the `GridWorldEnv` class, P/C matrix construction, and `policy_eval` from the reference notebook (`Q__G_Learning.ipynb`) into `environment/gridworld.py`, following the interface and constraints in `CLAUDE.md`.

**What was produced:**
- `environment/gridworld.py`: `GridWorldEnv`, module-level `P` and `C` matrices, `policy_eval`, and supporting constants (`INV_ST`, `TERMINAL_STATE`, `ACTION_DELTAS`, state-index maps).
- `environment/__init__.py`: re-exports all public symbols.

**Changes / manual verification:**
- The only intentional deviation from the notebook: all `np.random.*` calls replaced with `self.rng.*` (a `np.random.Generator` injected at construction) for reproducibility per `CLAUDE.md`.
- Verified `policy_eval` guard `/ (norm_V + 1e-12)` does not change algorithm semantics; original notebook would divide by zero on first iteration if V=0.
- Step dynamics and P/C build logic were copied verbatim.

**Independent decisions:**
- Kept P/C matrix construction at module level (built once on import) rather than lazily.
- Used a set for `_inv_state_set` for O(1) blocked-state checks inside the build loop, but retained the list format in `GridWorldEnv` for the step logic to stay identical to the notebook.

---

## 2026-04-02 — Project directory structure

**What was requested:**
Create the full project directory structure from `CLAUDE.md` (all directories, stub files, config, requirements, gitignore). No algorithm implementations or notebooks yet.

**What was produced:**
- All directories: `algorithms/`, `utils/`, `experiments/configs/`, `experiments/results/`, `notebooks/`, `report/figures/`, `tests/`.
- `algorithms/base.py`: abstract `Agent` interface.
- `algorithms/{q_learning,double_q_learning,sarsa,expected_sarsa,entropy_reg_q,reinforce,actor_critic}.py`: stubs (`raise NotImplementedError`).
- `utils/metrics.py`: full implementation of `signed_error`, `abs_error`, `policy_eval_error`.
- `utils/schedule.py`: `lr_schedule` and `beta_schedule`.
- `utils/plotting.py`: empty stub.
- `experiments/configs/default.yaml`: all hyperparameters from `CLAUDE.md`.
- `experiments/{run_all,run_single}.py`: empty stubs.
- `tests/test_env.py`: all 4 required environment tests.
- `tests/test_algorithms.py`: skeleton with TODO stubs.
- `requirements.txt`, `.gitignore`, `report/llm_usage.md`.

**Changes / manual verification:**
- Confirmed all 11 tests (4 env + 7 algorithm) pass after environment and Q-learning were implemented.
- `default.yaml` values copied verbatim from `CLAUDE.md`.

**Independent decisions:**
- Used `.gitkeep` files to preserve empty tracked directories (`notebooks/`, `results/`, `report/figures/`).

---

## 2026-04-02 — Q-learning implementation

**What was requested:**
Implement `QLearningAgent` in `algorithms/q_learning.py` and write tests in `tests/test_algorithms.py`.

**What was produced:**
- `algorithms/q_learning.py`: full `QLearningAgent` implementing the `Agent` interface. Count-based LR, epsilon-greedy exploration (epsilon resets per episode, decays per step), visit counts reset per episode to match reference notebook.
- `tests/test_algorithms.py`: 7 tests across 3 classes — convergence on a trivial 2-state MDP, finiteness on the gridworld, policy validity.

**Changes / manual verification:**
- Confirmed all 7 new tests pass alongside the 4 existing environment tests (11/11).
- Verified Q*(0,0)=1.0, Q*(0,1)=5.0 on trivial MDP within tolerance 0.10 after 3000 episodes.

**Independent decisions:**
- Visit counts reset per episode (not cumulative across episodes) to match the reference notebook. This means the first update in each episode uses `alpha = (1/2)^0.8 ≈ 0.574` regardless of how many prior episodes have run.
- `select_action` uses `argmin Q` for greedy (cost minimisation); epsilon decays inside `update` so it fires exactly once per step.
- `get_policy` returns a one-hot greedy matrix (not softmax) — the policy Q-learning has actually learned, not its exploration policy.
