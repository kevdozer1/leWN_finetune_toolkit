# Dataset format

lewm-finetune reads episodes from a **single HDF5 file** via
`stable_worldmodel.data.dataset.HDF5Dataset`. This is the only supported
format. There is no CSV loader, no NPZ loader, no folder-of-JPEGs loader. If
you want to train on new data, convert it to this format first, then run
``python scripts/validate_dataset.py <file>.h5`` before you train.

## File location

```
<data_cache_dir>/datasets/<dataset_name>.h5
```

`data_cache_dir` and `dataset_name` come from your config file. The
`datasets/` subdirectory is mandatory — it is a `stable_worldmodel`
convention and the loader hard-codes it.

## Required keys

All five of these must be present at the root of the HDF5 file. The validator
(`scripts/validate_dataset.py`) rejects any file missing one of them.

| Key           | Shape                | Dtype      | Description                                                |
|---------------|----------------------|------------|------------------------------------------------------------|
| `pixels`      | `(N, H, W, 3)`       | `uint8`    | RGB frames, HWC. Resized to `image_size` (default 224).    |
| `action`      | `(N, action_dim)`    | `float32`  | Per-frame action. NaNs are replaced with 0 at train time.  |
| `observation` | `(N, obs_dim)`       | `float32`  | Per-frame proprio/state vector. Per-column normalized.     |
| `ep_len`      | `(E,)`               | `int64`    | Length in frames of each episode.                          |
| `ep_offset`   | `(E,)`               | `int64`    | Start index of each episode along the flat `N` axis.       |

Here `N` is the total number of frames across all episodes, and `E` is the
number of episodes. The layout must be **contiguous, cumulative**:

```
ep_offset[0] == 0
ep_offset[i] == sum(ep_len[:i])  for all i in [1, E)
N == ep_offset[E-1] + ep_len[E-1]
```

Episode `i`'s data spans indices `[ep_offset[i], ep_offset[i] + ep_len[i])`
along the first axis of every flat column.

### Per-field notes

- **`pixels`** — HWC uint8 RGB. `HDF5Dataset` automatically permutes to CHW
  when tensorized. Any `H` and `W` work; they get resized at load time. Use
  HDF5 chunking + compression (lz4, blosc) if your episodes are large.
- **`action`** — must be float. The training loop calls
  `torch.nan_to_num(action, 0.0)` before feeding it to the model, so NaN
  entries are silently replaced with zero (useful if your episodes have
  missing actions at boundaries).
- **`observation`** — must be float. Per-column mean/std are fit on the full
  dataset (ignoring rows with any NaN) and applied as `(x - mean) / std`.
  Zero-variance columns are left unscaled to avoid NaN outputs.
- **`ep_len`** / **`ep_offset`** — integer. Arrays of the same length `E`.
  Episodes shorter than `num_steps * frameskip` are skipped entirely by the
  loader, so very short episodes will be silently ignored at training time.

## Optional keys (ignored by training)

Any additional keys in the HDF5 file are **tolerated but ignored** by the
training and evaluation loops. `lewm-finetune` only ever reads the five
required keys. You can carry extra columns (depth maps, masks, task labels,
etc.) in the same file without breaking anything — the validator will list
them as "Extra columns (ignored by training)" so you can see what's there.

## Episode metadata rules

The validator enforces the following invariants. Training will not work if
they are violated; `validate_dataset.py` will tell you exactly which one.

1. **Length agreement** — `len(ep_len) == len(ep_offset) == E`.
2. **Positivity** — every `ep_len[i] > 0`; every `ep_offset[i] >= 0`.
3. **Monotonicity** — `ep_offset` is non-decreasing.
4. **Contiguity** — `ep_offset[i] == cumsum(ep_len)[i-1]` for all `i > 0`.
5. **Flat-column length** — each of `pixels`, `action`, `observation` has at
   least `ep_offset[-1] + ep_len[-1]` rows. Extra trailing rows are allowed
   but produce a warning.

## How to validate a file before training

Run the bundled inspector. It is torch-free, so you can run it on any
machine that has `h5py` installed, even without the full training stack:

```bash
python scripts/validate_dataset.py ./data/datasets/my_dataset.h5
```

Exit codes:
- `0` — file is valid.
- `1` — file is well-formed HDF5 but violates the contract (issues printed).
- `2` — file is missing, not readable as HDF5, or otherwise unopenable.

For CI pipelines or scripted workflows, use `--json` to get a machine-readable
report:

```bash
python scripts/validate_dataset.py ./data/datasets/my_dataset.h5 --json
```

## Example: creating an HDF5 file from scratch

```python
import h5py
import numpy as np

EP_LENS = [200, 300, 500]           # three episodes
N = sum(EP_LENS)                    # 1000 frames total
H, W = 256, 256
ACTION_DIM, OBS_DIM = 7, 9

with h5py.File("data/datasets/my_dataset.h5", "w") as f:
    f.create_dataset("pixels",
                     data=np.zeros((N, H, W, 3), dtype=np.uint8))
    f.create_dataset("action",
                     data=np.zeros((N, ACTION_DIM), dtype=np.float32))
    f.create_dataset("observation",
                     data=np.zeros((N, OBS_DIM), dtype=np.float32))
    f.create_dataset("ep_len",
                     data=np.asarray(EP_LENS, dtype=np.int64))
    f.create_dataset("ep_offset",
                     data=np.cumsum([0] + EP_LENS[:-1]).astype(np.int64))
```

Corresponding config:

```yaml
pretrained_path: ./checkpoints/lewm-pretrained
dataset_name: my_dataset
data_cache_dir: ./data
action_dim: 7
```

## Windowing behaviour

At each step the loader yields a sequence of
`num_steps = history_size + num_preds` contiguous frames from a single
episode. `frameskip` subsamples non-action columns, but `action` is kept at
full rate and reshaped to `(num_steps, action_dim * frameskip)`, so the
**effective action dimension** fed to the model is
`action_dim * frameskip`. lewm-finetune automatically re-initializes the
pretrained action encoder if the effective dim differs from what the
checkpoint expects.

## Train/val split

lewm-finetune uses a deterministic `90 / 10` random split seeded by
`cfg.seed`. Evaluation reads the run's `config_snapshot.yaml` and replays
the exact same split using the same seed, so eval metrics are always
computed against the held-out 10% of windows that training never saw.
