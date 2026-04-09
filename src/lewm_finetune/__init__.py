"""lewm-finetune: minimal finetuning entry point for pretrained LeWorldModel.

The package is intentionally split so the lightweight validator can run
without torch installed. Heavy modules (``data``, ``train``, ``evaluate``)
are imported on demand rather than eagerly here.

Public surface (import directly from the submodule):

    from lewm_finetune.config   import load_config, DEFAULTS
    from lewm_finetune.validate import validate_dataset, ValidationError
    from lewm_finetune.train    import train, load_model
    from lewm_finetune.evaluate import evaluate
"""

from __future__ import annotations

__version__ = "0.1.0"

# Always safe to import (no torch dependency).
from .config import DEFAULTS, load_config
from .validate import ValidationError, validate_dataset

__all__ = [
    "DEFAULTS",
    "ValidationError",
    "__version__",
    "load_config",
    "validate_dataset",
]
