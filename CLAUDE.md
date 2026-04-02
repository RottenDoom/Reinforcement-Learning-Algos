# CLAUDE.md — RL Gridworld Project

## Project Purpose

Course project implementing and comparing model-free RL control algorithms on a
custom 8x8 Gridworld (Fox et al. 2016). The final deliverables are a codebase
and a comparison report. All algorithms must be comparable under identical
environment instances, seeds, and hyperparameters.

---

## Repository Layout

```
rl_gridworld_project/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── environment/
│   ├── __init__.py
│   └── gridworld.py          # GridWorldEnv, P matrix, C matrix, policy_eval
├── algorithms/
│   ├── __init__.py
│   ├── base.py               # abstract Agent with shared interface
│   ├── q_learning.py
│   ├── double_q_learning.py
│   ├── sarsa.py
│   ├── expected_sarsa.py
│   ├── entropy_reg_q.py      # G-learning; beta schedule lives here
│   ├── reinforce.py          # REINFORCE and REINFORCE-with-baseline
│   └── actor_critic.py       # one-step AC and AC(lambda)
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # signed_error, abs_error, policy_eval_error
│   ├── plotting.py           # all matplotlib figure functions
│   └── schedule.py           # learning rate and beta schedulers
├── experiments/
│   ├── configs/
│   │   └── default.yaml      # single source of truth for all hyperparameters
│   ├── run_all.py            # runs every algorithm, saves .npy results
│   ├── run_single.py         # single-algorithm run for debugging
│   └── results/              # .npy output files, gitignored
├── notebooks/
│   ├── 00_environment_overview.ipynb
│   ├── 01_q_learning_and_entropy_q.ipynb
│   ├── 02_sarsa_variants.ipynb
│   ├── 03_policy_gradient.ipynb
│   ├── 04_actor_critic.ipynb
│   └── 05_final_comparison.ipynb
├── report/
│   ├── figures/              # exported .png/.pdf for the report
│   └── llm_usage.md          # mandatory log of LLM usage
└── tests/
    ├── test_env.py
    └── test_algorithms.py
```

---

## Environment

**Source file:** `environment/gridworld.py`

The environment is the Fox (2016) 8x8 Gridworld. Do not modify its dynamics
without explicit instruction.

**Key facts:**

- States: 64 flat indices (row * 8 + col), 0-indexed.
- Invalid (blocked) flat indices: `[9, 17, 25, 33, 34, 42, 50, 12, 20, 28, 29, 30, 38, 46, 54]`
- Terminal state: flat index 36, i.e. (row=4, col=4). Zero cost, absorbing.
- Actions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW, 8=Stay. Total 9 actions.
- Intended move succeeds; then stochastic drift applies with probability 0.30
  (cardinal directions 0.05 each, diagonal directions 0.025 each).
- Cost per step: 1 + Normal(0, 0.2). Wall collision adds penalty 1000 + Normal(0, 0.2).
- Drift adds cost 1 + Normal(0, 0.2) only when drift actually displaces the agent.
- `env.reset()` samples uniformly from valid non-terminal states.
- `env.reset_previous((row, col))` resets to a specific state (used when two
  algorithms must start from the same state in the same episode).
- `env.step(action)` returns `(obs, cost, done, info)` where obs is
  `np.array([row, col])`.

**Model matrices** (also in `environment/gridworld.py`):
- `P[s_prime, s, a]`: transition probability, shape (64, 64, 9).
- `C[s_prime, s, a]`: expected cost, shape (64, 64, 9).
- These are deterministic matrices used only for model-based policy evaluation,
  not by the learning algorithms.

**Optimal values** (hardcoded, used as ground truth for metrics):

```python
optimal = np.array([
    [4.4098, 3.8096, 3.8096, 3.8096, 3.8096, 4.4098, 4.9311, 5.3873],
    [4.4098,      0, 3.0288, 3.0288,      0, 4.4098, 4.9311, 5.3873],
    [4.8806,      0, 2.1455, 2.1455,      0, 5.0336, 5.0336, 5.4940],
    [5.2439,      0, 2.0294, 1.0000,      0,      0,      0, 5.1445],
    [4.9307,      0,      0, 1.0000,      0, 1.0000,      0, 4.7615],
    [4.4761, 4.4761,      0, 1.0000, 1.0000, 1.0000,      0, 4.3139],
    [4.3638, 3.7411,      0, 2.0988, 2.0988, 2.0988,      0, 3.7411],
    [4.3638, 3.7411, 3.0170, 3.0170, 3.0170, 3.0170, 3.0170, 3.7411],
])
optimal_flat = optimal.ravel()   # shape (64,), index by flat state
```

Invalid and terminal states have value 0 in this array and must be skipped in
all metric computations.

---

## Hyperparameters (default.yaml is authoritative)

```yaml
environment:
  gamma: 0.85
  n_episodes: 1500
  max_steps_per_episode: 300
  n_runs: 10

learning_rate:
  omega: 0.8        # alpha_t = (1 / visit_count[s,a]) ^ omega

exploration:
  epsilon_start: 0.4
  epsilon_decay: 0.9   # per step within an episode

entropy_q:
  beta_min: 0.1
  beta_max: 7.0        # linearly scheduled over n_episodes

sarsa_lambda:
  lambda_val: 0.9

actor_critic:
  lambda_critic: 0.9
  lambda_actor: 0.9
  lr_actor: 0.01

reinforce:
  lr_policy: 0.01
  lr_baseline: 0.1
```

Do not hardcode any of these values inside algorithm files. Load them from the
config. Algorithm files may accept them as constructor arguments.

---

## Algorithm Specifications

All algorithms share:
- Count-based learning rate: `alpha_t = (1 / n_t(s, a)) ^ 0.8` where `n_t` is
  initialized to 1 (not 0) to avoid division by zero.
- Initial Q/G/V tables: all zeros.
- Uniform prior for entropy-reg Q: `rho(a|s) = 1/9` for all valid (s,a).
- Exploration: epsilon-greedy with epsilon starting at 0.4 and decaying by
  factor 0.9 per step within each episode.

### Q-learning
TD target: `cost + gamma * min_a Q[s_prime, a]`
Update: `Q[s, a] += alpha * (target - Q[s, a])`

### Double Q-learning
Maintain `Q_A` and `Q_B`, both shape (64, 9).
With prob 0.5: select greedy action from `Q_A`, evaluate with `Q_B`.
Exploration policy uses `(Q_A + Q_B) / 2`.

### SARSA
On-policy. Sample `a_prime` from epsilon-greedy policy before update.
TD target: `cost + gamma * Q[s_prime, a_prime]`

### Expected SARSA
TD target: `cost + gamma * sum_a pi(a|s_prime) * Q[s_prime, a]`
where pi is the current epsilon-greedy policy at s_prime.

### Entropy-Regularized Q-learning (G-learning)
Beta is linearly scheduled: `beta_t = beta_min + t * (beta_max - beta_min) / n_episodes`

Softmax value at state s:
```python
Q_min = np.min(Q[s, :])
exp_vals = np.exp(-beta * (Q[s, :] - Q_min))   # subtract min for stability
V[s] = (-1/beta) * (-beta * Q_min + np.log(np.sum(exp_vals)))
mu[s, :] = exp_vals / np.sum(exp_vals)
```

TD target uses V[s_prime] (not min Q):
`target = cost + gamma * V[s_prime]`
Update: `Q[s, a] += alpha * (target - Q[s, a])`

The policy mu[s,:] is used for action selection (the epsilon-greedy wrapper
still applies on top of mu).

### REINFORCE
Episode-based. Tabular softmax policy with parameters theta shape (64, 9).

```
pi(a|s) = softmax(theta[s, :])
```

After collecting a full episode `{(s_t, a_t, cost_t)}`:
```
G_t = sum_{k=t}^{T} gamma^{k-t} * cost_k
theta[s_t, a_t] += lr * G_t * (1 - pi(a_t|s_t))
theta[s_t, a]   -= lr * G_t * pi(a|s_t)   for all a != a_t
```

REINFORCE-with-baseline additionally maintains V shape (64,):
```
delta_t = G_t - V[s_t]
V[s_t]  += lr_baseline * delta_t
Use delta_t in place of G_t for the policy update.
```

### One-step Actor-Critic
At every step compute TD error:
```
delta = cost + gamma * V[s_prime] - V[s]
V[s]  += lr_critic * delta
theta[s_t, a_t] += lr_actor * delta * (1 - pi(a_t|s_t))
theta[s_t, a]   -= lr_actor * delta * pi(a|s_t)   for all a != a_t
```

### Actor-Critic with eligibility traces AC(lambda)
Maintain trace vectors `z_v` (shape 64) and `z_theta` (shape 64x9).

```
z_v[s]           = gamma * lambda_critic * z_v[s] + 1
z_theta[s, a_t]  = gamma * lambda_actor  * z_theta[s, a_t] + (1 - pi(a_t|s))
z_theta[s, a]    = gamma * lambda_actor  * z_theta[s, a]   - pi(a|s)   for a != a_t

delta = cost + gamma * V[s_prime] - V[s]

V     += lr_critic * delta * z_v
theta += lr_actor  * delta * z_theta
```

Reset traces to zero at the start of each episode.

---

## Metrics

Defined in `utils/metrics.py`. All metrics skip invalid states and terminal
state (flat index 36). Let `valid_states` be all flat indices not in
`inv_st` and not equal to 36.

```python
inv_st = [9, 17, 25, 33, 34, 42, 50, 12, 20, 28, 29, 30, 38, 46, 54]
```

**Signed relative error** (tracks optimism/pessimism of value estimate):
```
(1/|valid|) * sum_s (V_est[s] - V_star[s]) / V_star[s]
```

**Absolute relative error** (tracks convergence regardless of sign):
```
(1/|valid|) * sum_s |V_est[s] - V_star[s]| / V_star[s]
```

**Policy evaluation error** (quality of the greedy policy induced):
```
V_pi = policy_eval(greedy_policy_from_Q_or_G)
(1/|valid|) * sum_s |V_pi[s] - V_star[s]| / V_star[s]
```

For value-based algorithms, `V_est[s] = min_a Q[s,a]`.
For G-learning, `V_est[s] = V_er[s]` (the softmax value).
For policy gradient / actor-critic, `V_est[s] = V[s]` (the critic).

---

## Base Agent Interface

Every algorithm must implement this interface (defined in `algorithms/base.py`):

```python
class Agent:
    def select_action(self, state: int) -> int: ...
    def update(self, s, a, cost, s_prime, done) -> None: ...
    def get_value_estimate(self) -> np.ndarray: ...  # shape (64,), V_est per state
    def get_policy(self) -> np.ndarray: ...          # shape (64, 9), pi(a|s)
    def reset_episode(self) -> None: ...             # reset traces, epsilon, etc.
```

State passed to all methods is the flat integer index, not (row, col).

---

## Coding Conventions

- Python 3.10+. No gym/gymnasium dependency anywhere.
- numpy only for numerics. No PyTorch or JAX unless explicitly instructed.
- Type hints on all public functions.
- No print statements inside algorithm update loops. Use logging at DEBUG level
  if needed.
- All random operations go through a seeded `np.random.Generator` passed at
  construction time, never through the global `np.random` state, so that runs
  are reproducible.
- Flat state indices everywhere inside algorithm code. Convert (row, col) to
  flat index immediately after receiving obs from env: `s = obs[0] * 8 + obs[1]`.
- When a numerical overflow risk exists in softmax (e.g., in G-learning or
  softmax policy), always subtract the max/min before exponentiating.

---

## Testing Requirements

`tests/test_env.py` must verify:
1. Row sums of `P[:, s, a]` equal 1.0 for all valid (s,a).
2. `env.reset()` never returns a blocked or terminal state.
3. Stepping from terminal state returns done=True immediately.
4. Optimal values reproduced by value iteration on P and C to within 1e-2.

`tests/test_algorithms.py` must verify:
1. Each algorithm converges on a trivial 2-state 2-action MDP with known Q-star.
2. Q tables remain finite (no NaN or Inf) after 1500 episodes on the Gridworld.
3. Policy distributions sum to 1 for all valid states.

---

## LLM Usage Log

All LLM assistance must be documented in `report/llm_usage.md` with:
- Date
- What was requested
- What was produced
- What was changed or verified manually
- What decision was made independently

This file is a project deliverable.

---

## What NOT to Do

- Do not modify `GridWorldEnv.step()` or the P/C matrix construction logic.
- Do not share mutable state between algorithm instances in `run_all.py`.
- Do not use `np.random.seed()` globally; use per-run `np.random.default_rng(seed)`.
- Do not implement any algorithm by wrapping another algorithm's class; keep
  them independent for clarity during viva Q&A.
- Do not average results in-place during runs; store all runs first, then
  compute statistics.