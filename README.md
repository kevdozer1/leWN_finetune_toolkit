# lewm-finetune

A small, boring, reliable tool for fine-tuning a pretrained
[LeWorldModel](https://github.com/galilai-group/stable-worldmodel) (LeWM)
checkpoint on your own HDF5 episode dataset.

One YAML config. One CLI with three commands: `train`, `eval`, `inspect`.
No auxiliary heads, no dataset-specific hooks, no ablation scaffolding.

## What this is

- **Who it's for:** researchers and engineers who already have a pretrained
  LeWM checkpoint and an HDF5 dataset of robot/manipulation episodes, and
  want to fine-tune without building their own training loop.
- **What you get:** a single `lewm-finetune` CLI that trains, evaluates, and
  validates datasets against a canonical contract; self-describing run
  output directories that can be reloaded as pretrained models.
- **Loss:** the plain LeWM objective —
  `pred_mse(pred_emb, target_emb) + sigreg_weight * SIGReg(emb)`. Nothing
  else.
- **Model size:** LeWM ViT-tiny is ~18M parameters and fits comfortably on
  a single consumer GPU; the default path is full fine-tuning.

## What this is NOT

This repo deliberately does **not** try to be:

- a **dataset platform** — bring your own HDF5 file in the format documented
  in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)
- a **benchmark zoo** — no bundled datasets, leaderboards, or task suites
- a **GUI** — everything runs from the command line
- a **many-format training framework** — one loss, one dataset format, one
  training loop
- a **parameter-efficient finetuning framework** — no LoRA, no adapters, no
  prompt tuning. The model is small enough that full fine-tuning is the
  simple default.
- a **research dump** — no experiment matrices, no auxiliary heads, no
  ablation runners. If you want to extend the loss with auxiliary
  supervision, fork and edit `_lewm_forward` in [src/lewm_finetune/train.py](src/lewm_finetune/train.py).

## Install

> ### STOP — READ THIS FIRST
>
> **`pip install -e ".[train]"` alone is NOT sufficient.** You must also
> install `stable-worldmodel` from source *before* installing the training
> extras. The PyPI wheel of `stable-worldmodel` 0.0.6 is missing the entire
> LeWM code path that this tool depends on (`stable_worldmodel.wm.utils.load_pretrained`,
> `stable_worldmodel.wm.lewm.module.Embedder`, and
> `stable_worldmodel.wm.loss.SIGReg`). If you skip the source install, `pip`
> will look happy but `lewm-finetune train` will crash at runtime inside the
> import chain.
>
> Tested against upstream commit **`ba10600`** of
> [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
> (tested 2026-04-09). Later commits may or may not work.

Follow the steps below **in order**:

```bash
# 1. Clone this repo.
git clone https://github.com/kevdozer1/leWN_finetune_toolkit.git lewm-finetune
cd lewm-finetune

# 2. Create and activate a Python 3.10+ virtual environment.
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install torch for your platform first (CPU or CUDA build).
#    See https://pytorch.org/get-started/locally/
pip install torch

# 4. REQUIRED: install stable-worldmodel from source.
#    The PyPI wheel is missing the LeWM code path this tool relies on.
#    Pin to the tested commit if you want guaranteed compatibility:
pip install -e "git+https://github.com/galilai-group/stable-worldmodel.git@ba10600#egg=stable-worldmodel"
#    (or drop @ba10600 to track main, at your own risk)

# 5. Install lewm-finetune and the rest of the training stack.
pip install -e ".[train]"
```

The `[train]` extras bring in `lightning` and `stable-pretraining`. They
also declare `stable-worldmodel>=0.0.6` as a dependency, but pip will
respect the editable install from step 4 and **not** replace it with the
broken PyPI wheel — as long as you do step 4 first.

If you ran step 5 before step 4 and already have the broken PyPI wheel
installed, uninstall it and redo:

```bash
pip uninstall stable-worldmodel
# then run steps 4 and 5 above
```

### Lightweight install (inspect-only)

If you only want to validate HDF5 files and don't need to train, you can
skip the training stack entirely:

```bash
pip install -e .     # base deps only: numpy, h5py, pyyaml, tqdm
lewm-finetune inspect ./my_dataset.h5
```

This avoids torch, lightning, and `stable-worldmodel` entirely.

## The three commands

### 1. `lewm-finetune inspect <path>` — validate a dataset

Before training, verify your HDF5 file matches the canonical dataset
contract:

```bash
lewm-finetune inspect ./data/datasets/my_dataset.h5
```

Prints shapes, dtypes, episode counts, and any contract violations. Exits
non-zero if the file is invalid. Use `--json` for machine-readable output,
`--no-strict` to keep exit 0 on issues (useful in aggregated CI).

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the exact format.

### 2. `lewm-finetune train --config <yaml>` — fine-tune

```bash
lewm-finetune train --config configs/minimal.yaml
```

Reads the config, loads the pretrained checkpoint, adapts the action
encoder if needed, trains for `max_epochs`, and writes artifacts to
`runs/<timestamp>_<run_name>/`.

### 3. `lewm-finetune eval --run-dir <dir>` — measure validation loss

```bash
lewm-finetune eval --run-dir runs/20260408_101530_finetune
```

Reloads the finetuned weights, reproduces the exact same `seed`-based
90/10 split used in training, and writes `metrics.json` into the run
directory.

## Configuration

The minimal config is four lines:

```yaml
# configs/minimal.yaml
pretrained_path: ./checkpoints/lewm-pretrained
dataset_name: my_dataset
data_cache_dir: ./data
action_dim: 7
```

See [configs/example.yaml](configs/example.yaml) for every supported knob.
Only `pretrained_path`, `dataset_name`, and `data_cache_dir` are required;
everything else has a default in [src/lewm_finetune/config.py](src/lewm_finetune/config.py).

### Small-GPU knobs

LeWM is small, so in most cases you can train at full precision with
batch 16 on a single 8GB card. If you still need to shrink:

```yaml
batch_size: 4              # per-step batch
grad_accum_steps: 4        # effective batch = 4 * 4 = 16
precision: bf16-mixed      # default; use 16-mixed on older GPUs, 32-true on CPU
```

`grad_accum_steps > 1` is wired through to
`pl.Trainer(accumulate_grad_batches=...)` — see the `Trainer` call in
[src/lewm_finetune/train.py](src/lewm_finetune/train.py).

## Run output layout

Every run produces a self-describing directory:

```
runs/<timestamp>_<run_name>/
├── config_snapshot.yaml        # exact config used (defaults + overrides)
├── metadata.json               # seed, params, wall clock, versions
├── summary.txt                 # human-readable summary of metadata.json
├── metrics.json                # written by `lewm-finetune eval`
└── checkpoints/
    └── final/
        ├── weights.pt          # finetuned model state_dict
        └── config.json         # copied from the pretrained checkpoint
```

The `final/` folder can be reloaded directly by `load_pretrained`:

```python
from stable_worldmodel.wm.utils import load_pretrained

model = load_pretrained("runs/20260408_101530_finetune/checkpoints/final")
```

See [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) for details on
pretrained-checkpoint input formats and finetuned-checkpoint output
layout.

## Dataset format

The canonical format is a single HDF5 file with five required keys
(`pixels`, `action`, `observation`, `ep_len`, `ep_offset`). Any extra
keys are tolerated and ignored. See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)
for the exact layout and a worked example of writing one from scratch.

Always run `lewm-finetune inspect` on your dataset before training.

## Known limitations

- **Source-install dependency on `stable-worldmodel`.** The PyPI wheel of
  `stable-worldmodel` 0.0.6 is missing the LeWM code path (`wm/utils.py`,
  `wm/loss.py`, `wm/lewm/`). Training and eval will not work without the
  upstream editable install described in *Install*. Tested against commit
  `ba10600`; later commits may drift.
- **No resumable training.** Only the model `state_dict` is persisted;
  optimizer and scheduler state are not. Fine for short fine-tuning runs,
  not for multi-day jobs. There is currently no `--resume` flag.
- **No multi-GPU or distributed training.** The `Trainer` is hard-coded to
  `devices=1` and has not been tested with `devices>1`, DDP, FSDP, or
  multi-node. Users who need multi-GPU should fork and wire their own
  `pl.Trainer` arguments.
- **HDF5 only.** No folder-of-JPEGs loader, no NPZ loader, no video-file
  loader. Convert upstream into the format documented in
  [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).
- **Single loss.** Plain LeWM objective only — no auxiliary supervision,
  no LoRA/adapters, no parameter-efficient methods. If you need more,
  fork and edit `_lewm_forward` in
  [src/lewm_finetune/train.py](src/lewm_finetune/train.py).
- **Windows console Unicode (cosmetic).** `stable-pretraining` logs via
  loguru with emoji characters that Windows' default console codepage
  cannot encode, producing occasional `UnicodeEncodeError` lines during
  training. These are cosmetic only — training itself still succeeds and
  all run artifacts are written correctly. If it bothers you, set
  `PYTHONIOENCODING=utf-8` or run inside Windows Terminal.

## Repository layout

```
lewm-finetune/
├── README.md
├── pyproject.toml
├── LICENSE
├── configs/
│   ├── minimal.yaml         # smallest valid config
│   ├── example.yaml         # annotated config with every knob
│   └── smoke.yaml           # tiny end-to-end smoke config
├── docs/
│   ├── DATA_FORMAT.md       # canonical HDF5 dataset contract
│   └── CHECKPOINTS.md       # pretrained input + finetuned output format
├── scripts/
│   ├── finetune.py          # thin shim: `lewm-finetune train`
│   ├── evaluate.py          # thin shim: `lewm-finetune eval`
│   └── validate_dataset.py  # thin shim: `lewm-finetune inspect`
├── src/lewm_finetune/
│   ├── __init__.py          # torch-free public surface (config, validate)
│   ├── cli.py               # unified `lewm-finetune` entry point
│   ├── config.py            # YAML loading + DEFAULTS
│   ├── validate.py          # HDF5 dataset validator (torch-free)
│   ├── data.py              # HDF5Dataset wiring + transforms
│   ├── checkpointing.py     # writes weights.pt + config.json
│   ├── train.py             # forward function, load_model, train()
│   ├── evaluate.py          # evaluate() on a run's val split
│   └── utils.py
└── tests/
    └── test_smoke.py        # torch-free config + validator tests
```

## v0.1.0 notes

This is an early, honest initial release. Scope and expectations:

- **Verified end-to-end** on real data: `inspect` → `train` → `eval`, using
  a pretrained LeWM ViT-tiny checkpoint and a real HDF5 episode dataset.
  1-epoch training completed; finetuned weights reloaded and evaluated
  successfully.
- **Tested against upstream `stable-worldmodel` commit** `ba10600` (tested
  2026-04-09). If upstream drifts, this tool may break until it is updated.
- **Biggest caveat:** you must install `stable-worldmodel` from source
  first. See *Install* above. There is no workaround until upstream ships
  a PyPI wheel that contains the LeWM code path.
- **Scope:** exactly three commands (`train`, `eval`, `inspect`), one
  dataset format, one loss function, one training loop. Future versions
  will stay within this scope.

See [CHANGELOG.md](CHANGELOG.md) for change history.

## License

MIT. See [LICENSE](LICENSE).
