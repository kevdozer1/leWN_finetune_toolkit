"""CLI entry point: inspect and validate an HDF5 dataset.

Equivalent to ``lewm-finetune inspect <path>``. This script exists so the
repo works out-of-the-box without ``pip install -e .``.

Note: this script is intentionally named ``validate_dataset.py`` and not
``inspect.py``, because a script called ``inspect.py`` shadows Python's
stdlib ``inspect`` module and breaks downstream imports (e.g. torch).

Usage:
    python scripts/validate_dataset.py ./data/datasets/my_dataset.h5
    python scripts/validate_dataset.py ./data/datasets/my_dataset.h5 --json
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import the validate module directly, not via the package __init__, so this
# script stays torch-free even if the broader package later takes on eager
# imports.
from lewm_finetune.validate import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
