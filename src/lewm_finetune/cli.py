"""Unified command-line entry point for lewm-finetune.

Exposes three subcommands, mirroring the very small public surface of this
tool:

    lewm-finetune train    --config configs/minimal.yaml
    lewm-finetune eval     --run-dir runs/20260408_example
    lewm-finetune inspect  ./data/datasets/my_dataset.h5

Each subcommand is a thin shim over the corresponding module-level ``main``
function, so the individual ``scripts/finetune.py`` / ``scripts/evaluate.py``
/ ``scripts/validate_dataset.py`` entry points continue to work identically.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lewm-finetune",
        description=(
            "Fine-tune a pretrained LeWorldModel checkpoint on a user dataset. "
            "See README.md and docs/DATA_FORMAT.md for details."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"lewm-finetune {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train ----------------------------------------------------------------
    p_train = sub.add_parser(
        "train",
        help="Fine-tune a pretrained LeWM on an HDF5 dataset.",
        description="Fine-tune a pretrained LeWM on an HDF5 dataset.",
    )
    p_train.add_argument("--config", required=True, help="Path to YAML config.")
    p_train.add_argument("--seed", type=int, default=None, help="Override cfg['seed'].")
    p_train.add_argument(
        "--output-dir", default=None, help="Override cfg['output_dir']."
    )

    # eval -----------------------------------------------------------------
    p_eval = sub.add_parser(
        "eval",
        help="Evaluate a finetune run against its held-out validation split.",
        description="Evaluate a finetune run against its held-out validation split.",
    )
    p_eval.add_argument(
        "--run-dir",
        required=True,
        help="Path to a run directory produced by `lewm-finetune train`.",
    )
    p_eval.add_argument("--device", default=None, help="torch device (default: auto).")

    # inspect --------------------------------------------------------------
    p_inspect = sub.add_parser(
        "inspect",
        help="Inspect and validate an HDF5 dataset against the canonical contract.",
        description=(
            "Inspect and validate an HDF5 dataset against the canonical contract "
            "defined in docs/DATA_FORMAT.md."
        ),
    )
    p_inspect.add_argument("path", help="Path to the .h5 file.")
    p_inspect.add_argument(
        "--json", action="store_true", help="Emit JSON instead of pretty text."
    )
    p_inspect.add_argument(
        "--no-strict",
        action="store_true",
        help="Exit 0 even when issues are found (they are still printed).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        # Imported lazily so that `lewm-finetune inspect` doesn't need torch.
        try:
            from .train import train as _train
            from .config import load_config
        except ModuleNotFoundError as e:
            _print_train_install_hint(e)
            return 3

        cfg = load_config(args.config)
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.output_dir is not None:
            cfg["output_dir"] = args.output_dir
        _train(cfg)
        return 0

    if args.command == "eval":
        try:
            from .evaluate import evaluate as _evaluate
        except ModuleNotFoundError as e:
            _print_train_install_hint(e)
            return 3

        _evaluate(args.run_dir, device=args.device)
        return 0

    if args.command == "inspect":
        from .validate import main as _validate_main

        extra = [args.path]
        if args.json:
            extra.append("--json")
        if args.no_strict:
            extra.append("--no-strict")
        return _validate_main(extra)

    # argparse already enforces required=True, but be defensive.
    parser.print_help()
    return 2


def _print_train_install_hint(err: ModuleNotFoundError) -> None:
    missing = err.name or "an expected dependency"
    print(
        f"ERROR: {missing} is not installed.\n"
        "\n"
        "The train/eval commands require the '[train]' extras (torch,\n"
        "lightning, stable-pretraining, stable-worldmodel). Install with:\n"
        "\n"
        "    pip install torch                       # pick your CUDA/CPU build first\n"
        "    pip install -e '.[train]'\n"
        "\n"
        "Note: `stable_worldmodel.wm.utils.load_pretrained` is only present\n"
        "in the upstream source tree, not the PyPI wheel. See README.md\n"
        "(\"Install\") for the editable-install workaround.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
