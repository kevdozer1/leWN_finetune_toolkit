"""Checkpoint callback that writes ``weights.pt`` + ``config.json`` pairs.

The layout is the one ``stable_worldmodel.wm.utils.load_pretrained`` expects,
so any run output directory can be loaded straight back as a pretrained model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as pl
import torch


class SaveWeightsCallback(pl.Callback):
    """Save ``weights.pt`` + ``config.json`` after each epoch and/or at the end.

    Parameters
    ----------
    output_dir:
        Base directory to write checkpoints under. Per-epoch checkpoints go to
        ``<output_dir>/epoch_<N>/`` and the final one to ``<output_dir>/final/``.
    model_config:
        The original pretrained model's ``config.json`` content, used verbatim
        so the saved checkpoint can be reloaded by ``load_pretrained``.
    save_every_epoch:
        If True, write a checkpoint at the end of every training epoch.
        If False, only the final checkpoint is written (in ``on_fit_end``).
    """

    def __init__(
        self,
        output_dir: Path,
        model_config: dict[str, Any],
        save_every_epoch: bool = False,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.model_config = model_config
        self.save_every_epoch = save_every_epoch

    # ------------------------------------------------------------------ hooks
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.save_every_epoch or not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        self._save(pl_module, self.output_dir / f"epoch_{epoch}")

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self._save(pl_module, self.output_dir / "final")

    # ----------------------------------------------------------------- helper
    def _save(self, pl_module: pl.LightningModule, ckpt_dir: Path) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.model.state_dict(), ckpt_dir / "weights.pt")
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(self.model_config, f, indent=2)
        print(f"  [checkpoint] {ckpt_dir}")
