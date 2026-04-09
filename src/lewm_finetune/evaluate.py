"""Evaluation entry point for lewm-finetune.

Loads a finetuned checkpoint from a run directory, reproduces the exact same
train/val split, and reports validation prediction MSE and SIGReg loss.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from . import __version__


def evaluate(run_dir: str | Path, device: str | None = None) -> dict[str, Any]:
    """Evaluate a finetuned run against its validation split.

    ``run_dir`` must be a directory produced by :func:`lewm_finetune.train.train`
    — it must contain ``config_snapshot.yaml`` and a
    ``checkpoints/final/weights.pt`` (or at least one ``epoch_N`` sibling).
    """
    import numpy as np
    import stable_pretraining as spt
    import torch
    from stable_worldmodel.wm.loss import SIGReg

    from .data import build_dataloaders, build_dataset, split_dataset
    from .train import load_model

    run_dir = Path(run_dir)
    cfg = _load_run_config(run_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model on CPU first, then load finetuned weights, then move to device.
    print(f"Evaluating run: {run_dir}")
    model, _model_config = load_model(cfg)
    weights_path = _resolve_weights(run_dir)
    print(f"  Loading weights: {weights_path}")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Reproduce dataset + split.
    dataset = build_dataset(cfg)
    _, val_set = split_dataset(dataset, cfg)
    _, val_loader = build_dataloaders(cfg, val_set, val_set)  # train unused

    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    sigreg_weight = float(cfg["sigreg_weight"])
    history_size = int(cfg["history_size"])
    num_preds = int(cfg["num_preds"])

    pred_losses: list[float] = []
    sigreg_losses: list[float] = []
    total_losses: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            out = model.encode(batch)
            emb = out["emb"]
            act_emb = out["act_emb"]

            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            tgt_emb = emb[:, num_preds:]
            pred_emb = model.predict(ctx_emb, ctx_act)

            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            sigreg_loss = sigreg(emb.transpose(0, 1))
            total = pred_loss + sigreg_weight * sigreg_loss

            pred_losses.append(pred_loss.item())
            sigreg_losses.append(sigreg_loss.item())
            total_losses.append(total.item())

    result = {
        "lewm_finetune_version": __version__,
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "device": device,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "val_samples": len(val_set),
        "batch_size": int(cfg["batch_size"]),
        "sigreg_weight": float(cfg["sigreg_weight"]),
        "metrics": {
            "pred_mse": float(np.mean(pred_losses)),
            "sigreg_loss": float(np.mean(sigreg_losses)),
            "total_loss": float(np.mean(total_losses)),
        },
    }

    # metrics.json is the canonical machine-readable output. evaluation_results.json
    # mirrors the same content for scripts that expect the older filename.
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    legacy_path = run_dir / "evaluation_results.json"
    with open(legacy_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  pred_mse={result['metrics']['pred_mse']:.6f}")
    print(f"  sigreg_loss={result['metrics']['sigreg_loss']:.6f}")
    print(f"  total_loss={result['metrics']['total_loss']:.6f}")
    print(f"  Saved: {metrics_path}")
    return result


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    snapshot = run_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        raise FileNotFoundError(
            f"No config_snapshot.yaml in {run_dir}. Is this a lewm-finetune "
            "run directory?"
        )
    with open(snapshot) as f:
        return yaml.safe_load(f)


def _resolve_weights(run_dir: Path) -> Path:
    """Pick the best available weights file inside ``run_dir/checkpoints``."""
    ckpt_root = run_dir / "checkpoints"
    final = ckpt_root / "final" / "weights.pt"
    if final.exists():
        return final
    # Fall back to the highest-numbered epoch_N/weights.pt.
    epoch_dirs = sorted(
        (d for d in ckpt_root.glob("epoch_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[1]),
    )
    for d in reversed(epoch_dirs):
        cand = d / "weights.pt"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No weights.pt found under {ckpt_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a lewm-finetune run on its validation split."
    )
    parser.add_argument("--run-dir", required=True, help="Path to a finetune run dir.")
    parser.add_argument("--device", default=None, help="torch device (default: auto).")
    args = parser.parse_args()
    evaluate(args.run_dir, device=args.device)


if __name__ == "__main__":
    main()
