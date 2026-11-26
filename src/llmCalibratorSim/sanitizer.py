"""Strict Sovereignty-safe outbound telemetry sanitizer (Unit 2 / MOE 1)."""

from __future__ import annotations

import math
from typing import Any

from llmCalibratorSim import CATEGORIES, EXT_ENUMS, OUTCOMES

ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "decision_id",
        "ext_enum",
        "pred_cat",
        "actual_cat",
        "outcome",
        "raw_goodness_G",
        "confidence_p",
        "timestamp_s",
    }
)
ALLOWED_EXTS: frozenset[str] = frozenset(EXT_ENUMS)
ALLOWED_CATS: frozenset[str] = frozenset(CATEGORIES)
ALLOWED_OUTCOMES: frozenset[str] = frozenset(OUTCOMES)


class SovereigntyViolationError(ValueError):
    """Raised when an outbound telemetry record violates the strict 8-key schema."""


def _requireStrictInt(value: Any, fieldName: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SovereigntyViolationError(f"{fieldName} must be strict int, got {type(value).__name__}")
    if value < 0:
        raise SovereigntyViolationError(f"{fieldName} must be >= 0, got {value}")
    return int(value)


def _requireStrictFloat(value: Any, fieldName: str, minVal: float | None = None, maxVal: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SovereigntyViolationError(f"{fieldName} must be numeric scalar, got {type(value).__name__}")
    fval = float(value)
    if not math.isfinite(fval):
        raise SovereigntyViolationError(f"{fieldName} must be finite, got {value}")
    if minVal is not None and fval < minVal:
        raise SovereigntyViolationError(f"{fieldName} below minimum {minVal}: {fval}")
    if maxVal is not None and fval > maxVal:
        raise SovereigntyViolationError(f"{fieldName} above maximum {maxVal}: {fval}")
    return fval


def _requireEnum(value: Any, allowed: frozenset[str], fieldName: str) -> str:
    if not isinstance(value, str):
        raise SovereigntyViolationError(f"{fieldName} must be str, got {type(value).__name__}")
    if value not in allowed:
        raise SovereigntyViolationError(f"{fieldName} has disallowed value: {value!r}")
    return value


def sanitizeTelemetryArray(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and copy an outbound telemetry array with zero raw-data egress."""
    if not isinstance(records, list):
        raise SovereigntyViolationError("Telemetry payload must be a list of dicts")

    sanitized: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise SovereigntyViolationError(f"Record {idx} must be a dict")
        keys = set(rec.keys())
        if keys != ALLOWED_KEYS:
            extra = keys - ALLOWED_KEYS
            missing = ALLOWED_KEYS - keys
            raise SovereigntyViolationError(
                f"Record {idx} key mismatch (extra={sorted(extra)}, missing={sorted(missing)})"
            )

        sanitized.append(
            {
                "decision_id": _requireStrictInt(rec["decision_id"], "decision_id"),
                "ext_enum": _requireEnum(rec["ext_enum"], ALLOWED_EXTS, "ext_enum"),
                "pred_cat": _requireEnum(rec["pred_cat"], ALLOWED_CATS, "pred_cat"),
                "actual_cat": _requireEnum(rec["actual_cat"], ALLOWED_CATS, "actual_cat"),
                "outcome": _requireEnum(rec["outcome"], ALLOWED_OUTCOMES, "outcome"),
                "raw_goodness_G": _requireStrictFloat(rec["raw_goodness_G"], "raw_goodness_G"),
                "confidence_p": _requireStrictFloat(
                    rec["confidence_p"], "confidence_p", minVal=0.0, maxVal=1.0
                ),
                "timestamp_s": _requireStrictInt(rec["timestamp_s"], "timestamp_s"),
            }
        )
    return sanitized
