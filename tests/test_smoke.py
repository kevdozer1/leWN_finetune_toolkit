"""Smoke tests that do not require torch / stable_worldmodel / GPU.

These only cover the lightweight surface: config loading, validation, and
the defaults contract. Training and evaluation are exercised via
integration runs, not unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lewm_finetune import DEFAULTS, ValidationError, load_config  # noqa: E402
from lewm_finetune.validate import REQUIRED_KEYS, validate_dataset  # noqa: E402

h5py = pytest.importorskip("h5py")  # validator tests only run if h5py is installed
import numpy as np  # noqa: E402


def test_defaults_contract() -> None:
    """DEFAULTS must include every required key (as ``None``) and sensible
    fallbacks for the rest."""
    for required in ("pretrained_path", "dataset_name", "data_cache_dir"):
        assert required in DEFAULTS
        assert DEFAULTS[required] is None

    # A few knobs that drive the training loop.
    assert DEFAULTS["history_size"] >= 1
    assert DEFAULTS["num_preds"] >= 1
    assert DEFAULTS["batch_size"] > 0
    assert DEFAULTS["max_epochs"] > 0
    assert 0.0 <= DEFAULTS["sigreg_weight"]


def test_load_minimal_config(tmp_path: Path) -> None:
    """The example minimal config parses and satisfies the required-keys rule."""
    cfg_path = REPO_ROOT / "configs" / "minimal.yaml"
    cfg = load_config(cfg_path)
    assert cfg["pretrained_path"] == "./checkpoints/lewm-pretrained"
    assert cfg["dataset_name"] == "my_dataset"
    assert cfg["data_cache_dir"] == "./data"
    # Defaults filled in.
    assert cfg["max_epochs"] == DEFAULTS["max_epochs"]
    assert cfg["seed"] == DEFAULTS["seed"]


def test_load_example_config() -> None:
    """The annotated example config also parses and validates."""
    cfg = load_config(REPO_ROOT / "configs" / "example.yaml")
    assert cfg["run_name"] == "example"
    assert cfg["action_dim"] == 7


def test_missing_required_keys_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("batch_size: 8\n")
    with pytest.raises(ValueError, match="Missing required config keys"):
        load_config(p)


def test_user_overrides_defaults(tmp_path: Path) -> None:
    p = tmp_path / "ok.yaml"
    p.write_text(
        "pretrained_path: x\n"
        "dataset_name: y\n"
        "data_cache_dir: z\n"
        "max_epochs: 99\n"
        "lr: 1.0e-3\n"
    )
    cfg = load_config(p)
    assert cfg["max_epochs"] == 99
    assert cfg["lr"] == 1.0e-3
    # Untouched defaults still present.
    assert cfg["batch_size"] == DEFAULTS["batch_size"]
    assert cfg["sigreg_weight"] == DEFAULTS["sigreg_weight"]


# ---------------------------------------------------------------------------
# Validator tests (torch-free, exercise the canonical dataset contract)
# ---------------------------------------------------------------------------
def _write_valid_h5(path: Path, ep_lens=(4, 6, 5)) -> None:
    total = sum(ep_lens)
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels", data=np.zeros((total, 8, 8, 3), dtype=np.uint8))
        f.create_dataset("action", data=np.zeros((total, 7), dtype=np.float32))
        f.create_dataset("observation", data=np.zeros((total, 5), dtype=np.float32))
        f.create_dataset("ep_len", data=np.asarray(ep_lens, dtype=np.int64))
        offsets = np.concatenate([[0], np.cumsum(ep_lens[:-1])]).astype(np.int64)
        f.create_dataset("ep_offset", data=offsets)


def test_validator_passes_on_valid_file(tmp_path: Path) -> None:
    p = tmp_path / "ok.h5"
    _write_valid_h5(p)
    report = validate_dataset(p)
    assert report["ok"] is True
    assert report["issues"] == []
    assert set(report["keys"].keys()) == set(REQUIRED_KEYS)
    assert report["episodes"]["count"] == 3
    assert report["episodes"]["total_frames"] == 15


def test_validator_flags_missing_keys(tmp_path: Path) -> None:
    p = tmp_path / "missing.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("pixels", data=np.zeros((1, 8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValidationError, match="Missing required keys"):
        validate_dataset(p, strict=True)


def test_validator_flags_bad_offsets(tmp_path: Path) -> None:
    p = tmp_path / "bad_offsets.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("pixels", data=np.zeros((10, 8, 8, 3), dtype=np.uint8))
        f.create_dataset("action", data=np.zeros((10, 7), dtype=np.float32))
        f.create_dataset("observation", data=np.zeros((10, 5), dtype=np.float32))
        f.create_dataset("ep_len", data=np.asarray([4, 6], dtype=np.int64))
        # ep_offset should be [0, 4]; intentionally wrong.
        f.create_dataset("ep_offset", data=np.asarray([0, 7], dtype=np.int64))
    with pytest.raises(ValidationError, match="not consistent with ep_len"):
        validate_dataset(p, strict=True)


def test_validator_flags_ragged_flat_columns(tmp_path: Path) -> None:
    p = tmp_path / "ragged.h5"
    with h5py.File(p, "w") as f:
        # Episodes say 15 frames, but pixels only has 10.
        f.create_dataset("pixels", data=np.zeros((10, 8, 8, 3), dtype=np.uint8))
        f.create_dataset("action", data=np.zeros((15, 7), dtype=np.float32))
        f.create_dataset("observation", data=np.zeros((15, 5), dtype=np.float32))
        f.create_dataset("ep_len", data=np.asarray([5, 5, 5], dtype=np.int64))
        f.create_dataset("ep_offset", data=np.asarray([0, 5, 10], dtype=np.int64))
    with pytest.raises(ValidationError, match="10 rows but episode metadata"):
        validate_dataset(p, strict=True)
