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

---

## 2026-04-02 — SARSA implementation

**What was requested:**
Implement `SARSAAgent` in `algorithms/sarsa.py` and add corresponding tests in `tests/test_algorithms.py`, following the same pattern as Q-learning.

**What was produced:**
- `algorithms/sarsa.py`: full `SARSAAgent` implementing the `Agent` interface. On-policy TD control; a_prime is sampled from the current epsilon-greedy policy inside `update` and cached in `agent.next_action` for the caller to reuse.
- `tests/test_algorithms.py`: 7 new tests across 3 classes — convergence on the trivial 2-state MDP, finiteness on the gridworld, policy validity. Total test count: 18/18 passing.

**Changes / manual verification:**
- Confirmed all 18 tests pass (11 pre-existing + 7 new SARSA tests).
- Verified Q*(0,0)=1.0, Q*(0,1)=5.0 on trivial MDP within tolerance 0.10 after 3000 episodes.

**Independent decisions:**
- `a_prime` is sampled inside `update` (not by the caller beforehand) and stored in `self._next_action`. The training loop retrieves it via `agent.next_action` to avoid double-sampling. This keeps the `Agent` interface signature identical to Q-learning while correctly implementing on-policy SARSA.
- Visit counts and epsilon reset per episode, consistent with Q-learning.
- `get_policy` returns a one-hot greedy matrix (argmin Q), same convention as Q-learning.

---

## 2026-04-06 — Full algorithm suite scaffolding

**What was requested:**
Produce the scaffolding for all remaining algorithms (Double Q-learning, Expected SARSA, G-learning, REINFORCE, REINFORCE-with-baseline, one-step Actor-Critic, AC(lambda)) so that they can be implemented, add tests for each, and update the LLM usage log.

**What was produced:**
Scaffolding for the algorithms:

- `algorithms/double_q_learning.py`: `DoubleQLearningAgent`. 
- `algorithms/expected_sarsa.py`: `ExpectedSARSAAgent`. 
- `algorithms/entropy_reg_q.py`: `EntropyRegQLearningAgent`. 
- `algorithms/reinforce.py`: `REINFORCEAgent` and `REINFORCEWithBaselineAgent`. 
- `algorithms/actor_critic.py`: `ActorCriticAgent` (one-step TD) and `ActorCriticLambdaAgent` (eligibility traces).
- `tests/test_algorithms.py`: 30 new tests across 14 test classes.

**Changes / manual verification:**
- The algorithms were implemented manually.
- Checked softmax stability: all policies sum to 1.0 within 1e-9 for all valid states.

---

## 2026-04-23 — Experiment runners and AC(lambda) bugfix

**What was requested:**
Implement `experiments/run_all.py` and `experiments/run_single.py` to run all model-free algorithms on the gridworld (context from `notebooks/00_environment_overview.ipynb`): 9 algorithms, n_runs independent seeds, 3 metrics per episode, results saved as .npy.

**What was produced:**
- `experiments/run_all.py`: full experiment runner. Loads config from YAML, runs every algorithm via factory functions, records signed_error / abs_error every episode and policy_eval_error every 50 episodes, saves per-algorithm .npy files. CLI: `--algo` to run one algorithm, `--config` to override YAML.
- `experiments/run_single.py`: debug runner. Prints a live per-episode table for one algorithm/run, optional `--save` flag.

**Bugs found and fixed:**

1. **AC(lambda) trace decay not global** (`algorithms/actor_critic.py`):
   The original implementation only decayed `z_v[s]` and `z_theta[s, :]` for the *current* state each step. All other states' traces were frozen at their last value. This caused V to grow unboundedly (traces from distant past steps still carry full weight into the global update `V += lr * delta * z_v`). Fix: `self._z_v *= gc` (decay all), then `self._z_v[s] += 1.0` (accumulate current). Same for z_theta.

2. **AC(lambda) actor trace sign wrong for cost minimisation**:
   Traces were accumulating `(I_a - pi)` — the reward-maximisation gradient direction. For cost minimisation, `theta += lr * delta * z_theta` should push the policy *away from* costly actions (theta[a_taken] should decrease when delta > 0). Requires trace to hold `(pi - I_a)`. Fix: `z_theta[s, :] += pi; z_theta[s, a] -= 1.0` (was `-= pi; += 1.0`). Now consistent with the one-step AC actor update.

   Before fix: `policy_err` stuck at 1.50 for all 300 episodes (policy not learning).
   After fix: `policy_err` 0.684 → 0.666 at episodes 50 → 100, continuing to improve.

**Independent decisions:**
- `abs_err` for policy-gradient / AC agents is expected to be large because `get_value_estimate()` returns V^pi (the critic value of the *current policy*), not V*. A suboptimal policy legitimately has high V^pi in this environment (wall costs of 1000 make V^pi >> V* early in training). The diagnostic metric for these agents is `policy_err`, not `abs_err`.
- Policy-eval metric recorded every 50 episodes (30 checkpoints per 1500-episode run) to keep runtime acceptable — policy_eval calls a linear solve internally.
- `run_episode()` uses `hasattr` to detect protocol differences (SARSA/G-learning/REINFORCE) rather than isinstance, keeping the loop agnostic to class hierarchy.
