"""Dataset and transform construction for lewm-finetune.

All data is read via ``stable_worldmodel.data.dataset.HDF5Dataset``. Only three
columns are ever required: ``pixels``, ``action``, and ``observation``.

See ``docs/DATA_FORMAT.md`` for the expected HDF5 layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def build_dataset(cfg: dict[str, Any]):
    """Build an ``HDF5Dataset`` from a normalized config.

    Returns the raw dataset (with ``transform`` already attached). Split into
    train/val separately via :func:`split_dataset`.
    """
    # Imported lazily so that `import lewm_finetune.config` stays dependency-light.
    from stable_worldmodel.data.dataset import HDF5Dataset

    num_steps = cfg["history_size"] + cfg["num_preds"]
    dataset = HDF5Dataset(
        name=cfg["dataset_name"],
        cache_dir=cfg["data_cache_dir"],
        num_steps=num_steps,
        frameskip=cfg["frameskip"],
        keys_to_load=["pixels", "action", "observation"],
        keys_to_cache=["action", "observation"],
    )
    dataset.transform = build_transform(cfg, dataset)
    return dataset


def build_transform(cfg: dict[str, Any], dataset):
    """Build the ``pixels`` + ``observation`` transform pipeline.

    The image pipeline converts uint8 HWC to ImageNet-normalized float CHW and
    resizes to ``cfg['image_size']``. The observation pipeline fits per-column
    mean/std on the full dataset and applies ``(x - mean) / std``.
    """
    import stable_pretraining as spt

    imagenet_stats = spt.data.dataset_stats.ImageNet
    to_image = spt.data.transforms.ToImage(
        **imagenet_stats, source="pixels", target="pixels"
    )
    resize = spt.data.transforms.Resize(
        cfg["image_size"], source="pixels", target="pixels"
    )
    transforms = [to_image, resize]

    if cfg.get("normalize_observation", True):
        transforms.append(_build_observation_normalizer(dataset))

    return spt.data.transforms.Compose(*transforms)


def _build_observation_normalizer(dataset):
    """Fit a per-column mean/std normalizer on the dataset's observation column."""
    import stable_pretraining as spt

    col = dataset.get_col_data("observation")
    data = torch.from_numpy(np.asarray(col))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    # Guard against zero-variance columns so we don't emit NaNs downstream.
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    def norm_fn(x: torch.Tensor) -> torch.Tensor:
        return ((x - mean) / std).float()

    return spt.data.transforms.WrapTorchTransform(
        norm_fn, source="observation", target="observation"
    )


def split_dataset(dataset, cfg: dict[str, Any], val_fraction: float = 0.1):
    """Deterministic 90/10 (by default) train/val split using the config seed."""
    import stable_pretraining as spt

    train_frac = 1.0 - val_fraction
    rng = torch.Generator().manual_seed(int(cfg["seed"]))
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[train_frac, val_fraction], generator=rng
    )
    return train_set, val_set


def build_dataloaders(cfg: dict[str, Any], train_set, val_set):
    """Build plain PyTorch dataloaders for a train/val split."""
    rng = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
        generator=rng,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        drop_last=False,
    )
    return train_loader, val_loader
