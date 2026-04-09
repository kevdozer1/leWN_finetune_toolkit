"""Config loading and defaults for lewm-finetune.

A lewm-finetune config is a plain YAML file. Only ``pretrained_path``,
``dataset_name``, and ``data_cache_dir`` are required; everything else has a
sensible default.

Example::

    pretrained_path: ./checkpoints/lewm-cube
    dataset_name: my_robot_episodes
    data_cache_dir: ./data

    action_dim: 7
    max_epochs: 10
    batch_size: 16
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: Default values merged into every config. Keys missing from the user's YAML
#: fall back to these. Anything present in the YAML wins.
DEFAULTS: dict[str, Any] = {
    # --- required-in-spirit (user should override) -------------------------
    "pretrained_path": None,
    "dataset_name": None,
    "data_cache_dir": None,
    # --- dataset / batching -------------------------------------------------
    "action_dim": 7,
    "frameskip": 1,
    "history_size": 3,
    "num_preds": 1,
    "batch_size": 16,
    "num_workers": 0,
    "image_size": 224,
    "normalize_observation": True,
    # --- optimization -------------------------------------------------------
    "max_epochs": 10,
    "lr": 5.0e-5,
    "weight_decay": 1.0e-3,
    "sigreg_weight": 0.09,
    "gradient_clip": 1.0,
    "grad_accum_steps": 1,  # bump this to squeeze large effective batches onto small GPUs
    "precision": "bf16-mixed",
    "seed": 42,
    # --- output -------------------------------------------------------------
    "output_dir": None,  # auto-filled from run_name + timestamp if None
    "run_name": "finetune",
    "save_every_epoch": False,  # if True, save a checkpoint after every epoch
}


_REQUIRED_KEYS = ("pretrained_path", "dataset_name", "data_cache_dir")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and merge it with the defaults.

    Raises ``ValueError`` if any of the required keys are missing after merge.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(user_cfg).__name__}")

    cfg = {**DEFAULTS, **user_cfg}
    _validate(cfg, source=path)
    return cfg


def _validate(cfg: dict[str, Any], source: Path | None = None) -> None:
    missing = [k for k in _REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        where = f" in {source}" if source else ""
        raise ValueError(
            f"Missing required config keys{where}: {missing}. "
            f"See configs/minimal.yaml for the smallest valid config."
        )
    if cfg["history_size"] < 1:
        raise ValueError("history_size must be >= 1")
    if cfg["num_preds"] < 1:
        raise ValueError("num_preds must be >= 1")
    if cfg["frameskip"] < 1:
        raise ValueError("frameskip must be >= 1")
    if cfg.get("grad_accum_steps", 1) < 1:
        raise ValueError("grad_accum_steps must be >= 1")
