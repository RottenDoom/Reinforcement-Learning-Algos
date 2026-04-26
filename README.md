# Reinforcement Learning Algorithms

Course project implementation for the course `MEL7118/MCL775 Reinforcement Learning`. This project compares various algorithms taught in classes.

---

## Setup

Python **3.10+** required.

```bash
git clone https://github.com/RottenDoom/Reinforcement-Learning-Algos
cd Reinforcement-Learning-Algos

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `pyyaml`, `scipy`, `pytest`.  No deep-learning framework needed.

---

## Running the Experiments

### Full benchmark (all 15 algorithms, 10 seeds each)

```bash
python experiments/run_all.py
```

Results are written to `experiments/results/` as `.npy` files, one per algorithm.  Expected runtime: **15–30 minutes** on a modern CPU.

### Single algorithm

```bash
python experiments/run_all.py --algo q_learning
```

Available names:

```
q_learning        double_q_learning   g_learning
sarsa             expected_sarsa
reinforce         reinforce_baseline
actor_critic      actor_critic_lambda
vanilla_sgd       sgd_momentum        sgd_nesterov
mirror_descent    natural_pg          trpo
```

### Quick debug run (single seed, live output)

```bash
# 3 seeds, 200 episodes
python experiments/run_single.py --algo g_learning --runs 3 --episodes 200

# specific seed with verbose per-episode table
python experiments/run_single.py --algo trpo --seed 42 --verbose
```

---

## Reproducing the Report Figures

1. Run the full benchmark (above).
2. Open and execute `notebooks/05_final_comparison.ipynb`.  
   All figures are saved to `report/figures/` and referenced by `main.tex`.
---

## Running the Tests

```bash
# full suite (71 tests)
pytest tests/ -v

# environment only
pytest tests/test_env.py -v

# algorithm implementations only
pytest tests/test_algorithms.py -v
```
---

## Hyperparameters

All hyperparameters live in `experiments/configs/default.yaml`.  Edit that file before re-running `run_all.py`; no algorithm source file needs to change.

Key parameters:

```yaml
environment:
  gamma: 0.85          # discount factor
  n_episodes: 1500
  n_runs: 10

exploration:
  epsilon_start: 0.2   # tuned from 0.4
  epsilon_decay: 0.9   # per-step decay within each episode

entropy_q:
  beta_min: 0.01       # G-learning temperature schedule
  beta_max: 7.0

actor_critic:
  lambda_critic: 0.0   # traces disabled — hurt performance on this env
  lambda_actor:  0.0
  lr_actor:      0.005

trpo:
  delta: 0.01          # KL trust-region radius
  line_search_steps: 5
```

---

## Reference

Environment based on:  
> Fox, Pakman, Tishby. *Taming the Noise in Reinforcement Learning via Soft Updates.* UAI 2016.
> The `Q__G_Learning.ipynb` was used as a reference for implementing the agents and scaffolding.
