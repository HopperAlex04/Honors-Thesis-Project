## 1. Software and code revision

- **Repository:** Honors-Thesis-Project (Cuttle RL agent).
- **Python dependencies:** install from the project root with `pip install -r requirements.txt`.
- **Reference commit:** runs in this bundle record `git_commit` **fba06f99b792** in `run_metadata.json` / `experiment_metadata.json`. For closest reproduction, check out that commit before training. Newer commits may still reproduce if training logic and defaults are unchanged, but behavior can drift.

Training is started through **`experiment_manager.py`**, which materializes per-run `hyperparams_config.json` and invokes **`train.py`**. The RNG seed used at runtime is always **`random_seed` in that run’s `hyperparams_config.json`** (not only the experiment-level `seeds` list).

## 2. What gets fixed for reproducibility

`train.py` sets, for a given integer seed:

- Python `random`, NumPy, PyTorch CPU, and (if available) CUDA seeds, plus deterministic cuDNN settings when CUDA is used.

See `set_random_seeds()` in `train.py`. The experiment manager assigns `config["random_seed"] = run_info["seed"]` when building each run’s config.

## 3. Recreating the twenty training runs

Each row is one run. After creating an experiment, run:

```bash
python experiment_manager.py run
```

from the project root (with the target experiment selected as documented in `README_EXPERIMENTS.md` or via `.current_experiment` under `experiments/`).

### 3.1 PPO — eight-run batch (four scale-11 + four large-hidden)

Matches `experiment_20260324_074635_ppo` in this bundle. Seeds are **interleaved**: scale-11 run *i* uses `seeds[2*i]`, large-hidden run *i* uses `seeds[2*i+1]` (see `create_ppo_experiment.py`).

```bash
python create_ppo_experiment.py --backbones both --runs 4 \
  --seeds 15796 861 76821 54887 6266 82387 37195 87499
```

That produces `ppo_scale11_run_01`…`04` and `ppo_large_hidden_run_01`…`04` with the seeds above.

### 3.2 PPO — fifth large-hidden run (same seed as first scale-11 in the batch)

```bash
python create_ppo_experiment.py --backbones large_hidden --runs 1 --seeds 15796
```

This matches `experiment_20260326_091129_ppo` (one run, seed **15796**). Experiment-level metadata in this bundle was corrected so `experiment_metadata.json` and `runs_status.json` agree with `run_metadata.json` and `hyperparams_config.json` for that seed.

### 3.3 PPO — fifth scale-11 run

```bash
python create_ppo_experiment.py --backbones scale_11 --runs 1 --seeds 36844
```

Matches `experiment_20260326_103652_ppo` (`ppo_scale_11_run_01`).

### 3.4 Double DQN — large-hidden embedding (five runs)

One experiment per seed (default script uses 40 rounds and `use_double_dqn: true` under peak-style hyperparameters):

```bash
python create_double_dqn_peak_conditions.py --network large_hidden --seed 1936185084
python create_double_dqn_peak_conditions.py --network large_hidden --seed 365573104
python create_double_dqn_peak_conditions.py --network large_hidden --seed 363460169
python create_double_dqn_peak_conditions.py --network large_hidden --seed 225174597
python create_double_dqn_peak_conditions.py --network large_hidden --seed 1111291925
```

### 3.5 Double DQN — scale-11 embedding (five runs)

```bash
python create_double_dqn_peak_conditions.py --network scale_11 --seed 1222253554
python create_double_dqn_peak_conditions.py --network scale_11 --seed 543462345
python create_double_dqn_peak_conditions.py --network scale_11 --seed 834271479
python create_double_dqn_peak_conditions.py --network scale_11 --seed 35972596
python create_double_dqn_peak_conditions.py --network scale_11 --seed 1680283109
```

### 3.6 Seed summary (all twenty)

| Family | Seeds (in the same order as `Commands.txt` metrics paths) |
|--------|-------------------------------------------------------------|
| ppolh | 861, 54887, 82387, 87499, 15796 |
| ppos11 | 15796, 76821, 6266, 37195, 36844 |
| lhdqn | 1936185084, 365573104, 363460169, 225174597, 1111291925 |
| s11dqn | 1222253554, 543462345, 834271479, 35972596, 1680283109 |

**Note:** Seed **15796** is intentionally reused for ppos11’s first batch run and for the extra single large-hidden PPO run.

## 4. Recreating exported figures

From the **project root** (where `export_graphs.py`, `config_loader.py`, and `metrics_reader.py` live), run the commands in **`Commands.txt`**. They call:

```bash
python export_graphs.py --metrics <five paths to metrics_logs> --output .../FinalData/exported_graphs/<family> --format svg
```

Adjust absolute paths if the repository lives somewhere other than `/run/media/alexh/HDDPrim/ThesisStuff/Honors-Thesis-Project`. The default dashboard stat list is `config/stats_dashboard.json` next to `export_graphs.py`.

## 5. Layout notes for researchers

- **`ppolh/` and `ppos11/`** each contain a **full copy** of `experiment_20260324_074635_ppo` (eight runs: both architectures). For thesis comparisons, use only the five runs per family listed in **`Commands.txt`**; the other four runs in each folder are duplicate architecture from the shared experiment design.
- **`lhdqn/`** and **`s11dqn/`** hold five separate single-run experiments each (one seed per experiment directory).
- Primary artifacts per run: `metrics_logs/*.jsonl`, `models/`, `run_metadata.json`, `hyperparams_config.json`.

## 6. Verifying a run’s seed

Always check the run directory:

- `runs/<run_id>/hyperparams_config.json` → `"random_seed"`
- `runs/<run_id>/run_metadata.json` → `"seed"`

These should match. If `experiment_metadata.json` or `runs_status.json` ever disagree with them, treat the **hyperparams / run_metadata** pair as the source of truth for what was actually trained.
