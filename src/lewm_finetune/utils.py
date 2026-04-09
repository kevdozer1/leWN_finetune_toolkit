"""Small helpers shared across lewm_finetune modules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any  # noqa: F401  -- used in type hints below


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    """Return the output directory for a run, creating a fresh timestamped
    directory under ``runs/`` if the config does not specify one."""
    if cfg.get("output_dir"):
        out = Path(cfg["output_dir"])
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("runs") / f"{ts}_{cfg.get('run_name', 'finetune')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def count_params(module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts for a torch module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def write_summary(path: Path, metadata: dict[str, Any]) -> None:
    """Write a short human-readable run summary next to ``metadata.json``.

    Kept intentionally compact — the machine-readable source of truth is
    ``metadata.json`` / ``metrics.json``; this file is just for humans.
    """
    lines = ["lewm-finetune run summary", "=" * 40]
    order = [
        "lewm_finetune_version",
        "run_name",
        "seed",
        "pretrained_path",
        "dataset_name",
        "train_samples",
        "val_samples",
        "max_epochs",
        "batch_size",
        "grad_accum_steps",
        "effective_batch_size",
        "precision",
        "lr",
        "sigreg_weight",
        "history_size",
        "num_preds",
        "total_params",
        "trainable_params",
        "device",
        "training_time_sec",
        "checkpoint_dir",
    ]
    width = max(len(k) for k in order)
    for k in order:
        if k in metadata:
            lines.append(f"{k:<{width}}  {metadata[k]}")
    path.write_text("\n".join(lines) + "\n")
