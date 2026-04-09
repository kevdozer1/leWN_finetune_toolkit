"""Training entry point for lewm-finetune.

The single public function :func:`train` takes a normalized config dict and
runs a full finetuning job. The companion script ``scripts/finetune.py`` just
parses a YAML file and calls this.

The training step implements the plain LeWorldModel objective: a prediction
MSE between predicted and target CLS embeddings, plus a SIGReg regularizer on
the embedding distribution. No auxiliary heads, no dataset-specific hooks.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .checkpointing import SaveWeightsCallback
from .config import load_config
from .data import build_dataloaders, build_dataset, split_dataset
from .utils import count_params, resolve_output_dir, write_summary


# ---------------------------------------------------------------------------
# Forward function: plain LeWM loss (pred_mse + sigreg_weight * sigreg).
# Kept at module scope so Lightning can pickle it across workers.
# ---------------------------------------------------------------------------
def _lewm_forward(
    self,  # bound spt.Module instance
    batch: dict,
    stage: str,
    history_size: int,
    num_preds: int,
    sigreg_weight: float,
):
    import torch

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    out = self.model.encode(batch)
    emb = out["emb"]            # (B, T, D)
    act_emb = out["act_emb"]    # (B, T, D_act)

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    out["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    out["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    out["loss"] = out["pred_loss"] + sigreg_weight * out["sigreg_loss"]

    self.log_dict(
        {
            f"{stage}/loss": out["loss"].detach(),
            f"{stage}/pred_loss": out["pred_loss"].detach(),
            f"{stage}/sigreg_loss": out["sigreg_loss"].detach(),
        },
        on_step=(stage == "train"),
        on_epoch=True,
        sync_dist=True,
    )
    return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Load a pretrained LeWorldModel and adapt its action encoder to match
    ``cfg['action_dim'] * cfg['frameskip']``.

    Returns ``(model, model_config_dict)``. The second element is the original
    ``config.json`` from the pretrained checkpoint (with any action encoder
    adjustments applied) so it can be re-saved alongside the finetuned weights.
    """
    from stable_worldmodel.wm.lewm.module import Embedder
    from stable_worldmodel.wm.utils import load_pretrained

    model = load_pretrained(cfg["pretrained_path"])
    model_config = _read_model_config(cfg["pretrained_path"])

    embed_dim = _infer_embed_dim(model)
    effective_action_dim = int(cfg["action_dim"]) * int(cfg["frameskip"])
    orig = int(model.action_encoder.input_dim)
    if orig != effective_action_dim:
        print(
            f"  Reinitializing action_encoder: {orig} -> {effective_action_dim} "
            f"(embed_dim={embed_dim})"
        )
        model.action_encoder = Embedder(
            input_dim=effective_action_dim, emb_dim=embed_dim
        )
        if isinstance(model_config, dict) and "action_encoder" in model_config:
            model_config["action_encoder"]["input_dim"] = effective_action_dim

    total, trainable = count_params(model)
    print(f"  Params: {total:,} total / {trainable:,} trainable")
    return model, model_config


def _infer_embed_dim(model) -> int:
    proj = getattr(model, "projector", None)
    if proj is not None and hasattr(proj, "net"):
        return int(proj.net[0].in_features)
    # Fall back to the action encoder's output embedding dim.
    return int(model.action_encoder.emb_dim)


def _read_model_config(pretrained_path: str) -> dict[str, Any]:
    """Read the pretrained checkpoint's ``config.json``.

    Supports the same path flavors as ``load_pretrained``: a folder, or a
    ``.pt`` file that lives next to ``config.json``.
    """
    p = Path(pretrained_path)
    if p.suffix == ".pt":
        p = p.parent
    cfg_path = p / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    # If the user pointed at an HF repo id, config.json lives in the resolved
    # cache directory — in that case just return an empty dict; the caller
    # will still be able to reload the weights but not via load_pretrained
    # from this specific run directory alone.
    print(
        f"  NOTE: could not find config.json at {cfg_path}; saved checkpoints "
        "will omit it and will only be reloadable via their original path."
    )
    return {}


# ---------------------------------------------------------------------------
# Top-level training entry point
# ---------------------------------------------------------------------------
def train(cfg: dict[str, Any]) -> Path:
    """Run a full finetuning job and return the output directory."""
    import lightning as pl
    import stable_pretraining as spt
    import torch
    from stable_worldmodel.wm.loss import SIGReg

    pl.seed_everything(int(cfg["seed"]), workers=True)

    output_dir = resolve_output_dir(cfg)
    with open(output_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("=" * 60)
    print(f"lewm-finetune  (run='{cfg['run_name']}'  seed={cfg['seed']})")
    print("=" * 60)
    print(f"  Pretrained:  {cfg['pretrained_path']}")
    print(f"  Dataset:     {cfg['dataset_name']} @ {cfg['data_cache_dir']}")
    print(f"  Output dir:  {output_dir}")
    print()

    print("Loading model...")
    model, model_config = load_model(cfg)

    print("\nLoading dataset...")
    dataset = build_dataset(cfg)
    print(f"  Samples: {len(dataset)}")

    train_set, val_set = split_dataset(dataset, cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_set, val_set)
    print(f"  Train: {len(train_set)}  Val: {len(val_set)}")

    print("\nBuilding Lightning module...")
    sigreg = SIGReg(knots=17, num_proj=1024)
    forward_fn = partial(
        _lewm_forward,
        history_size=int(cfg["history_size"]),
        num_preds=int(cfg["num_preds"]),
        sigreg_weight=float(cfg["sigreg_weight"]),
    )
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": {
                "type": "AdamW",
                "lr": float(cfg["lr"]),
                "weight_decay": float(cfg["weight_decay"]),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }
    lit_module = spt.Module(
        model=model,
        sigreg=sigreg,
        forward=forward_fn,
        optim=optimizers,
    )

    print(f"\nStarting training ({cfg['max_epochs']} epochs)...")
    save_cb = SaveWeightsCallback(
        output_dir=output_dir / "checkpoints",
        model_config=model_config,
        save_every_epoch=bool(cfg.get("save_every_epoch", False)),
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    trainer = pl.Trainer(
        max_epochs=int(cfg["max_epochs"]),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg["precision"],
        gradient_clip_val=float(cfg["gradient_clip"]),
        accumulate_grad_batches=int(cfg.get("grad_accum_steps", 1)),
        callbacks=[save_cb],
        num_sanity_val_steps=1,
        enable_checkpointing=False,
        logger=False,
        default_root_dir=str(output_dir),
    )

    t0 = time.time()
    trainer.fit(lit_module, data_module)
    elapsed = time.time() - t0

    total_params, trainable_params = count_params(model)
    effective_batch = int(cfg["batch_size"]) * int(cfg.get("grad_accum_steps", 1))
    metadata = {
        "lewm_finetune_version": __version__,
        "run_name": cfg["run_name"],
        "seed": cfg["seed"],
        "pretrained_path": cfg["pretrained_path"],
        "dataset_name": cfg["dataset_name"],
        "data_cache_dir": cfg["data_cache_dir"],
        "max_epochs": cfg["max_epochs"],
        "batch_size": cfg["batch_size"],
        "grad_accum_steps": cfg.get("grad_accum_steps", 1),
        "effective_batch_size": effective_batch,
        "precision": cfg["precision"],
        "lr": cfg["lr"],
        "sigreg_weight": cfg["sigreg_weight"],
        "history_size": cfg["history_size"],
        "num_preds": cfg["num_preds"],
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_time_sec": round(elapsed, 1),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_dir": str((output_dir / "checkpoints" / "final").resolve()),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    write_summary(output_dir / "summary.txt", metadata)

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Final checkpoint: {output_dir / 'checkpoints' / 'final'}")
    print(f"Run artifacts:    {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finetune a pretrained LeWorldModel on a user dataset."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Override cfg['seed'].")
    parser.add_argument(
        "--output-dir", default=None, help="Override cfg['output_dir']."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    train(cfg)


if __name__ == "__main__":
    main()
