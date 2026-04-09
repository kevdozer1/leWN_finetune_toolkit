"""Dataset validation for lewm-finetune.

Checks a candidate HDF5 file against the canonical dataset contract
described in ``docs/DATA_FORMAT.md``. Deliberately torch-free so it can run
before the training stack is even installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  -- registers compression filters for HDF5
import numpy as np


# ---------------------------------------------------------------------------
# Canonical dataset contract
# ---------------------------------------------------------------------------
REQUIRED_KEYS: tuple[str, ...] = (
    "pixels",
    "action",
    "observation",
    "ep_len",
    "ep_offset",
)

#: Per-key expectations. ``dtype_kind`` is a set of numpy dtype kinds
#: (``"u"`` unsigned int, ``"i"`` signed int, ``"f"`` float). ``ndim`` is the
#: set of acceptable array ranks.
REQUIRED_SPEC: dict[str, dict[str, Any]] = {
    "pixels": {
        "ndim": {4},
        "dtype_kind": {"u", "i"},
        "desc": "RGB frames, shape (N, H, W, 3), uint8 preferred",
    },
    "action": {
        "ndim": {2},
        "dtype_kind": {"f"},
        "desc": "Action vectors, shape (N, action_dim), float32 preferred",
    },
    "observation": {
        "ndim": {2},
        "dtype_kind": {"f"},
        "desc": "Observation vectors, shape (N, obs_dim), float32 preferred",
    },
    "ep_len": {
        "ndim": {1},
        "dtype_kind": {"i", "u"},
        "desc": "Episode lengths, shape (E,), integer",
    },
    "ep_offset": {
        "ndim": {1},
        "dtype_kind": {"i", "u"},
        "desc": "Episode start offsets into flat N axis, shape (E,), integer",
    },
}


class ValidationError(Exception):
    """Raised when dataset validation fails in strict mode."""


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------
def validate_dataset(h5_path: str | Path, strict: bool = True) -> dict[str, Any]:
    """Validate an HDF5 dataset file against the canonical contract.

    Parameters
    ----------
    h5_path:
        Path to the ``.h5`` file.
    strict:
        If ``True`` (default), raise :class:`ValidationError` when the report
        contains any ``issues``. If ``False``, return the report regardless.

    Returns
    -------
    dict
        A machine-readable report with keys ``path``, ``ok``, ``issues``,
        ``warnings``, ``keys``, ``episodes``, ``attributes``, and
        ``extra_keys``.
    """
    h5_path = Path(h5_path)
    issues: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "path": str(h5_path),
        "ok": False,
        "issues": issues,
        "warnings": warnings,
        "keys": {},
        "episodes": {},
        "extra_keys": [],
        "attributes": {},
    }

    if not h5_path.exists():
        raise ValidationError(f"File not found: {h5_path}")

    try:
        f = h5py.File(h5_path, "r")
    except OSError as e:
        raise ValidationError(f"Failed to open {h5_path} as HDF5: {e}") from e

    with f:
        present = list(f.keys())

        # 1. Required keys present.
        missing = [k for k in REQUIRED_KEYS if k not in present]
        if missing:
            issues.append(
                f"Missing required keys: {missing}. "
                f"Required contract: {list(REQUIRED_KEYS)}"
            )

        report["extra_keys"] = sorted(k for k in present if k not in REQUIRED_KEYS)

        # 2. Per-key shape/dtype checks.
        for key in REQUIRED_KEYS:
            if key not in f:
                continue
            ds = f[key]
            spec = REQUIRED_SPEC[key]
            info = {
                "shape": tuple(int(x) for x in ds.shape),
                "dtype": str(ds.dtype),
                "description": spec["desc"],
            }
            report["keys"][key] = info

            if ds.ndim not in spec["ndim"]:
                issues.append(
                    f"'{key}': expected ndim in {sorted(spec['ndim'])}, got "
                    f"{ds.ndim} (shape {tuple(ds.shape)})"
                )
            if ds.dtype.kind not in spec["dtype_kind"]:
                issues.append(
                    f"'{key}': expected dtype kind in "
                    f"{sorted(spec['dtype_kind'])}, got '{ds.dtype}' "
                    f"(kind '{ds.dtype.kind}')"
                )

        # 3. pixels-specific: RGB last axis + uint8 preference.
        if "pixels" in f and f["pixels"].ndim == 4:
            last = int(f["pixels"].shape[-1])
            if last not in (1, 3):
                issues.append(
                    f"'pixels': last axis must be 1 (grayscale) or 3 (RGB), "
                    f"got {last}"
                )
            if f["pixels"].dtype != np.uint8:
                warnings.append(
                    f"'pixels': dtype is {f['pixels'].dtype}; uint8 is strongly "
                    "preferred for disk size and matches HDF5Dataset's "
                    "automatic HWC->CHW permutation."
                )

        # 4. Episode metadata consistency.
        if "ep_len" in f and "ep_offset" in f:
            ep_len = np.asarray(f["ep_len"][:])
            ep_off = np.asarray(f["ep_offset"][:])

            if len(ep_len) != len(ep_off):
                issues.append(
                    f"ep_len has {len(ep_len)} entries but ep_offset has "
                    f"{len(ep_off)} (they must be the same length)"
                )

            E = min(len(ep_len), len(ep_off))
            if E > 0:
                if np.any(ep_len <= 0):
                    issues.append("ep_len contains zero or negative values")
                if np.any(ep_off < 0):
                    issues.append("ep_offset contains negative values")

                if E > 1 and np.any(np.diff(ep_off) < 0):
                    issues.append("ep_offset is not monotonically non-decreasing")

                if int(ep_off[0]) != 0:
                    warnings.append(
                        f"ep_offset[0] = {int(ep_off[0])}, expected 0 "
                        "(most datasets start at 0)"
                    )

                # Contiguity check: ep_offset[i] should equal cumsum(ep_len[:i]).
                expected = np.concatenate([[0], np.cumsum(ep_len[:-1])]).astype(
                    ep_off.dtype
                )
                mismatch = np.where(ep_off[:E] != expected[:E])[0]
                if mismatch.size > 0:
                    first = int(mismatch[0])
                    issues.append(
                        "ep_offset is not consistent with ep_len (expected "
                        f"cumulative-sum layout). First mismatch at episode "
                        f"{first}: ep_offset[{first}]={int(ep_off[first])}, "
                        f"expected {int(expected[first])}"
                    )

                total_frames = int(ep_off[E - 1] + ep_len[E - 1])
                report["episodes"] = {
                    "count": int(E),
                    "total_frames": total_frames,
                    "ep_len_min": int(ep_len.min()),
                    "ep_len_max": int(ep_len.max()),
                    "ep_len_mean": float(ep_len.mean()),
                }

                # Flat-column length alignment with total episode frames.
                for key in ("pixels", "action", "observation"):
                    if key in f:
                        n = int(f[key].shape[0])
                        if n < total_frames:
                            issues.append(
                                f"'{key}' has {n} rows but episode metadata "
                                f"requires at least {total_frames}"
                            )
                        elif n > total_frames:
                            warnings.append(
                                f"'{key}' has {n} rows; episodes cover only "
                                f"{total_frames} (extra rows will be ignored)"
                            )

        # 5. File-level attributes (purely informational).
        for k in f.attrs.keys():
            report["attributes"][k] = _attr_repr(f.attrs[k])

    report["ok"] = len(issues) == 0

    if strict and issues:
        msg = "\n  - ".join(issues)
        raise ValidationError(
            f"Dataset validation failed for {h5_path}:\n  - {msg}"
        )

    return report


def _attr_repr(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def print_report(report: dict[str, Any]) -> None:
    """Print a validation report in a compact, human-readable format."""
    print(f"Dataset: {report['path']}")
    print()

    if report["keys"]:
        print("Required columns:")
        for key in REQUIRED_KEYS:
            info = report["keys"].get(key)
            if info is None:
                print(f"  {key:14s}  [MISSING]")
                continue
            shape_str = "x".join(str(s) for s in info["shape"])
            print(f"  {key:14s}  ({shape_str})  {info['dtype']}")
        print()

    if report.get("extra_keys"):
        print("Extra columns (ignored by training):")
        for k in report["extra_keys"]:
            print(f"  {k}")
        print()

    ep = report.get("episodes") or {}
    if ep:
        print("Episodes:")
        print(f"  count           {ep.get('count')}")
        print(f"  total_frames    {ep.get('total_frames')}")
        if "ep_len_min" in ep:
            print(
                f"  ep_len          min={ep['ep_len_min']} "
                f"max={ep['ep_len_max']} "
                f"mean={ep['ep_len_mean']:.1f}"
            )
        print()

    if report.get("attributes"):
        print("HDF5 attributes:")
        for k, v in report["attributes"].items():
            print(f"  {k}: {v}")
        print()

    if report["warnings"]:
        print("Warnings:")
        for w in report["warnings"]:
            print(f"  [warn] {w}")
        print()

    if report["issues"]:
        print("Issues:")
        for i in report["issues"]:
            print(f"  [FAIL] {i}")
        print()
        print("Result: NOT VALID")
    else:
        print("Result: OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lewm-finetune-inspect",
        description=(
            "Inspect and validate an lewm-finetune HDF5 dataset against the "
            "canonical contract defined in docs/DATA_FORMAT.md."
        ),
    )
    parser.add_argument("path", help="Path to the .h5 file.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON to stdout instead of pretty text.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help=(
            "Exit 0 even when issues are found (they are still printed). "
            "Useful in CI pipelines that want to aggregate reports."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = validate_dataset(args.path, strict=False)
    except ValidationError as e:
        # File-level errors (missing file, not HDF5, etc.) are always fatal.
        print(f"ERROR: {e}")
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    if not report["ok"] and not args.no_strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
