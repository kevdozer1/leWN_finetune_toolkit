# Changelog

All notable changes to `lewm-finetune` will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and this project uses a simple linear version history.

## [0.1.0] — 2026-04-09

Initial public release. Early, honest v0.1.0.

### Added
- `lewm-finetune` CLI with three subcommands: `train`, `eval`, `inspect`.
- YAML-driven config with a four-line minimal example
  (`configs/minimal.yaml`) and a fully annotated reference
  (`configs/example.yaml`).
- Torch-free HDF5 dataset validator (`lewm-finetune inspect`) enforcing
  the canonical five-key contract (`pixels`, `action`, `observation`,
  `ep_len`, `ep_offset`). Supports `--json` and `--no-strict`.
- Self-describing run output directory: `config_snapshot.yaml`,
  `metadata.json`, `summary.txt`, `metrics.json`, and a `checkpoints/final/`
  folder reloadable via `stable_worldmodel.wm.utils.load_pretrained`.
- Small-GPU knobs: `batch_size`, `grad_accum_steps`, and `precision`
  (bf16-mixed by default) wired into `pl.Trainer`.
- Torch-free smoke tests covering config loading, defaults contract,
  and the dataset validator (`tests/test_smoke.py`).
- Documentation: [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the HDF5
  contract and [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) for the
  pretrained-input / finetuned-output layout.

### Verified
- End-to-end `inspect → train → eval` on a real HDF5 episode dataset
  using a pretrained LeWM ViT-tiny checkpoint. 1-epoch training
  completed; finetuned weights reloaded and evaluated successfully.
- Tested against upstream
  [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
  commit **`ba10600`** (tested 2026-04-09).

### Known limitations
- **Source-install dependency on `stable-worldmodel`.** The PyPI wheel
  of `stable-worldmodel` 0.0.6 is missing the LeWM code path
  (`wm/utils.py`, `wm/loss.py`, `wm/lewm/`). You must install
  `stable-worldmodel` from source before installing the `[train]`
  extras. See the *Install* section of the README.
- No resumable training — only the model `state_dict` is persisted.
- No multi-GPU / distributed training — `Trainer` is hard-coded to
  `devices=1` and has not been tested with `devices>1`, DDP, FSDP, or
  multi-node.
- HDF5 only — no folder-of-JPEGs, NPZ, or video loaders.
- Single loss — plain LeWM objective only
  (`pred_mse + sigreg_weight * SIGReg`). No auxiliary heads, no LoRA
  or other PEFT methods.
- Windows console Unicode warnings from `stable-pretraining`'s loguru
  emoji output are cosmetic only; training artifacts are written
  correctly. Mitigate with `PYTHONIOENCODING=utf-8` or Windows Terminal.

### Scope
Exactly three commands, one dataset format, one loss function, one
training loop. Future versions will stay within this scope.
