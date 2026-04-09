"""CLI entry point: finetune a pretrained LeWorldModel on a user dataset.

Equivalent to ``lewm-finetune train ...``. This script exists so the repo
works out-of-the-box without ``pip install -e .``.

Usage:
    python scripts/finetune.py --config configs/minimal.yaml
    python scripts/finetune.py --config configs/example.yaml --seed 7
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lewm_finetune.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
