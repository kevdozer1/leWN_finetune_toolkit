# Checkpoints

## Pretrained checkpoints (input)

`cfg.pretrained_path` is handed directly to
`stable_worldmodel.wm.utils.load_pretrained`, which accepts three forms:

1. **Folder** containing exactly one `.pt` file and a `config.json`:

    ```
    ./checkpoints/lewm-pretrained/
        weights.pt
        config.json
    ```

    ```yaml
    pretrained_path: ./checkpoints/lewm-pretrained
    ```

2. **Explicit `.pt` file** with a sibling `config.json` in the same folder:

    ```yaml
    pretrained_path: ./checkpoints/lewm-pretrained/weights_epoch_10.pt
    ```

3. **HuggingFace repo id** in `<user>/<repo>` form. If not already cached
   locally, it is downloaded to the active `stable_worldmodel` cache
   directory:

    ```yaml
    pretrained_path: nice-user/my-worldmodel
    ```

Absolute paths work in all three cases. `config.json` must faithfully
describe the model architecture — `load_pretrained` uses
`hydra.utils.instantiate(config)` to rebuild the model before loading weights.

### Action encoder adaptation

If your dataset's effective action dimensionality
(`cfg.action_dim * cfg.frameskip`) differs from what the pretrained
checkpoint's action encoder expects, lewm-finetune **re-initializes** the
action encoder as a fresh `stable_worldmodel.wm.lewm.module.Embedder` with
the new input dimension. This is a destructive change for that submodule —
it starts training from a random init while the rest of the model stays
pretrained. If you want to preserve the original action encoder, make sure
your `action_dim * frameskip` matches the checkpoint.

## Finetuned checkpoints (output)

Every run writes its outputs under:

```
runs/<timestamp>_<run_name>/
├── config_snapshot.yaml      # the exact config used (defaults + overrides)
├── metadata.json             # seed, sample counts, wall clock, etc.
└── checkpoints/
    └── final/
        ├── weights.pt        # finetuned model state_dict
        └── config.json       # copied from the pretrained checkpoint, with
                              #   action_encoder.input_dim updated if needed
```

The `final/` folder is directly reloadable by `load_pretrained`:

```python
from stable_worldmodel.wm.utils import load_pretrained

model = load_pretrained("runs/20260408_101530_finetune/checkpoints/final")
```

If you set `save_every_epoch: true` in the config, per-epoch checkpoints are
also written to `checkpoints/epoch_<N>/weights.pt` (with the same
`config.json`).

## Notes

- Only the model's `state_dict` is written — optimizer/scheduler state is not
  persisted. lewm-finetune is aimed at short finetuning runs, not resumable
  long runs.
- `config.json` is copied from the pretrained checkpoint's folder. If you
  load a pretrained model from a HuggingFace repo and the cache folder's
  `config.json` is not accessible from the resolved path, you will see a
  warning and the finetuned checkpoint will ship without `config.json` — you
  can still load its weights manually, but not via `load_pretrained` from
  that directory alone.
- `evaluate.py` reads `config_snapshot.yaml` to reproduce the exact
  train/val split from training, so evaluation is deterministic against the
  same seed.
