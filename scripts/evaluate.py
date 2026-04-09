"""CLI entry point: evaluate a lewm-finetune run on its validation split.

Equivalent to ``lewm-finetune eval ...``. This script exists so the repo
works out-of-the-box without ``pip install -e .``.

Usage:
    python scripts/evaluate.py --run-dir runs/20260408_101530_finetune
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lewm_finetune.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(["eval", *sys.argv[1:]]))
