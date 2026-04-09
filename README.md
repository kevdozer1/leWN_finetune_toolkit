# lewm-finetune

Fine-tune a pretrained [LeWorldModel](https://github.com/galilai-group/stable-worldmodel) checkpoint on your own HDF5 episode dataset.

One YAML config. Three CLI commands: `train`, `eval`, `inspect`. This is a minimal finetuning harness, not a benchmark suite, dataset platform, or general world-model framework.

Official LeWM checkpoints and datasets: [quentinll/lewm on Hugging Face](https://huggingface.co/collections/quentinll/lewm)

---

## Install

> **Important:** `stable-worldmodel` must be installed from source — the PyPI wheel (0.0.6) is missing the LeWM code path this tool depends on. Do this before installing the training extras or `pip` will silently pull in the broken wheel. Tested against commit `ba10600`.

```bash
# 1. Clone and create a virtualenv
git clone https://github.com/kevdozer1/leWN_finetune_toolkit.git lewm-finetune
# note: the GitHub repo name differs from the package name; the installed CLI is lewm-finetune
cd lewm-finetune

# macOS / Linux
python -m venv .venv && source .venv/bin/activate
# Windows PowerShell: python -m venv .venv && .venv\Scripts\Activate.ps1

# 2. Install torch for your platform
pip install torch  # see https://pytorch.org/get-started/locally/

# 3. Install stable-worldmodel from source (do this before step 4)
pip install -e "git+https://github.com/galilai-group/stable-worldmodel.git@ba10600#egg=stable-worldmodel"

# 4. Install lewm-finetune and the rest of the training stack
pip install -e ".[train]"
```

Confirm the source install is visible (the PyPI wheel and the source install share version `0.0.6`, so version alone does not tell you which one is loaded):

```bash
python -c "from stable_worldmodel.wm.utils import load_pretrained; print('stable-worldmodel OK')"
```

If you accidentally installed the PyPI wheel first:

```bash
pip uninstall -y stable-worldmodel
pip install -e "git+https://github.com/galilai-group/stable-worldmodel.git@ba10600#egg=stable-worldmodel"
pip install -e ".[train]"
```

**Inspect-only install** (no torch required):

```bash
pip install -e .
```

---

## Quickstart

Zero-to-train on a fresh machine, after the install steps above.

Point `pretrained_path` at a HuggingFace repo id to auto-download on first use — e.g. `quentinll/lewm-cube` from the [quentinll/lewm collection](https://huggingface.co/collections/quentinll/lewm). Official LeWM datasets live in the same collection; otherwise bring your own HDF5 matching [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). The loader reads `<data_cache_dir>/datasets/<dataset_name>.h5` — both pieces of the path come from the config.

`dataset_name` is the HDF5 filename stem under `<data_cache_dir>/datasets/` — for example, `data_cache_dir: ./data` and `dataset_name: my_dataset` means the file must exist at `./data/datasets/my_dataset.h5`.

Write a config:

```yaml
# my_run.yaml
pretrained_path: quentinll/lewm-cube
dataset_name: my_dataset
data_cache_dir: ./data
action_dim: 7
```

Then run the three commands:

```bash
lewm-finetune inspect ./data/datasets/my_dataset.h5
lewm-finetune train --config my_run.yaml
lewm-finetune eval --run-dir runs/<timestamp>_finetune
```

---

## Usage

### Validate a dataset

```bash
lewm-finetune inspect ./data/my_dataset.h5
```

Prints shapes, dtypes, episode counts, and any format violations. Torch-free — can be run before installing the training extras. See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the HDF5 contract.

### Train

```bash
lewm-finetune train --config configs/minimal.yaml
```

### Evaluate

```bash
lewm-finetune eval --run-dir runs/20260408_101530_finetune
```

Reproduces the exact train/val split and writes `metrics.json` into the run directory.

---

## Configuration

Minimal config:

```yaml
pretrained_path: ./checkpoints/lewm-pretrained
dataset_name: my_dataset   # expects the file at <data_cache_dir>/datasets/<dataset_name>.h5
data_cache_dir: ./data
action_dim: 7
```

See [configs/example.yaml](configs/example.yaml) for all options. Small-GPU knobs:

```yaml
batch_size: 4
grad_accum_steps: 4   # effective batch = 16
precision: bf16-mixed
```

---

## Run output

```
runs/<timestamp>_<run_name>/
├── config_snapshot.yaml
├── metadata.json
├── summary.txt
├── metrics.json          # written by eval
└── checkpoints/final/
    ├── weights.pt
    └── config.json
```

The `final/` directory can be reloaded directly:

```python
from stable_worldmodel.wm.utils import load_pretrained
model = load_pretrained("runs/20260408_101530_finetune/checkpoints/final")
```

---

## Loss

Plain LeWM objective:

```
pred_mse(pred_emb, target_emb) + sigreg_weight * SIGReg(emb)
```

No auxiliary heads. To extend the loss, edit `_lewm_forward` in [src/lewm_finetune/train.py](src/lewm_finetune/train.py).

---

## Known limitations

- `stable-worldmodel` must be installed from source (see Install)
- No training resumption — only `state_dict` is saved, not optimizer/scheduler state
- Single GPU only (`devices=1`)
- HDF5 dataset format only — see [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)
- Windows console may emit cosmetic `UnicodeEncodeError` lines from loguru; set `PYTHONIOENCODING=utf-8` to suppress

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

---

## License

MIT
