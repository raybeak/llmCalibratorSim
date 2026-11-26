"""Global PID, Per-Group PID, and Sovereignty-Safe Clamped LLM Calibrators (Unit 3)."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from llmCalibratorSim import EXT_ENUMS
from llmCalibratorSim.sanitizer import sanitizeTelemetryArray

VALID_ACTIONS: frozenset[str] = frozenset({"adjust_params", "trigger_replay", "noop"})
DELTA_BOUNDS: dict[str, tuple[float, float]] = {
    "d_theta": (-0.15, 0.15),
    "d_alpha": (-0.20, 0.20),
    "d_beta": (-0.20, 0.20),
    "d_T": (-0.20, 0.20),
}


def _zeroDelta() -> dict[str, float]:
    return {"d_theta": 0.0, "d_alpha": 0.0, "d_beta": 0.0, "d_T": 0.0}


@dataclass
class CalibrationCommand:
    globalDelta: dict[str, float] = field(default_factory=_zeroDelta)
    groupOverrides: dict[str, dict[str, float]] = field(default_factory=dict)
    recommendedAction: str = "noop"
    diagnosis: str = ""
    targetGroups: list[str] = field(default_factory=list)


def _clampDeltaDict(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("Delta block must be a dict")
    out = _zeroDelta()
    for key, (lo, hi) in DELTA_BOUNDS.items():
        if key in raw:
            val = raw[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{key} must be numeric")
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError(f"{key} must be finite")
            out[key] = max(lo, min(hi, fval))
    return out


def clampAndValidateOutput(rawOutput: Any) -> CalibrationCommand:
    """Parse, validate, and hard-clamp calibrator output; fall back safely to noop on any error."""
    try:
        if isinstance(rawOutput, str):
            text = rawOutput.strip()
            if not text:
                return CalibrationCommand(recommendedAction="noop", diagnosis="parse_fallback:empty")
            fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
            if fence:
                text = fence.group(1)
            data = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NaN/Inf")))
        elif isinstance(rawOutput, dict):
            data = rawOutput
        else:
            return CalibrationCommand(recommendedAction="noop", diagnosis="parse_fallback:type")

        if not isinstance(data, dict):
            return CalibrationCommand(recommendedAction="noop", diagnosis="parse_fallback:not_dict")

        globalDelta = _clampDeltaDict(data.get("global_delta", _zeroDelta()))
        overridesRaw = data.get("group_overrides", {})
        if not isinstance(overridesRaw, dict):
            return CalibrationCommand(recommendedAction="noop", diagnosis="parse_fallback:overrides")
        groupOverrides: dict[str, dict[str, float]] = {}
        for ext, deltaBlock in overridesRaw.items():
            if ext in EXT_ENUMS:
                groupOverrides[ext] = _clampDeltaDict(deltaBlock)

        action = data.get("recommended_action", "adjust_params")
        if action not in VALID_ACTIONS:
            action = "noop"

        diagnosis = str(data.get("diagnosis", ""))[:200]
        targetGroups = [
            g for g in data.get("target_groups", list(groupOverrides.keys())) if g in EXT_ENUMS
        ]
        return CalibrationCommand(
            globalDelta=globalDelta,
            groupOverrides=groupOverrides,
            recommendedAction=action,
            diagnosis=diagnosis,
            targetGroups=targetGroups,
        )
    except Exception:
        return CalibrationCommand(recommendedAction="noop", diagnosis="parse_fallback:exception")


class CalibratorBackend(ABC):
    """Pluggable backend receiving strictly sanitized telemetry arrays."""

    @abstractmethod
    def propose(self, sanitizedRecords: list[dict[str, Any]]) -> dict[str, Any] | str:
        raise NotImplementedError


class GlobalPidCalibrator:
    """Baseline 1: Single global P+I controller on (fn_rate - fp_rate)."""

    def __init__(self, kp: float = 0.45, ki: float = 0.08) -> None:
        self.kp = kp
        self.ki = ki
        self._integral = 0.0

    def calibrate(self, records: list[dict[str, Any]]) -> CalibrationCommand:
        if not records:
            return CalibrationCommand(recommendedAction="noop", diagnosis="empty")
        n = len(records)
        fnRate = sum(1 for r in records if r["outcome"] == "FN") / n
        fpRate = sum(1 for r in records if r["outcome"] == "FP") / n
        err = fnRate - fpRate
        self._integral = max(-0.5, min(0.5, self._integral + err))
        dTheta = -(self.kp * err + self.ki * self._integral)
        return clampAndValidateOutput(
            {
                "global_delta": {"d_theta": dTheta, "d_alpha": 0.0, "d_beta": 0.0, "d_T": 0.0},
                "group_overrides": {},
                "recommended_action": "adjust_params",
                "diagnosis": "global_pid",
            }
        )


class PerGroupPidCalibrator:
    """Baseline 2 (Hard Control): Independent P+I controller per ext_enum group."""

    def __init__(self, kp: float = 0.50, ki: float = 0.10) -> None:
        self.kp = kp
        self.ki = ki
        self._integrals: dict[str, float] = {ext: 0.0 for ext in EXT_ENUMS}

    def calibrate(self, records: list[dict[str, Any]]) -> CalibrationCommand:
        if not records:
            return CalibrationCommand(recommendedAction="noop", diagnosis="empty")
        byGroup: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            byGroup[r["ext_enum"]].append(r)

        overrides: dict[str, dict[str, float]] = {}
        for ext, grp in byGroup.items():
            n = len(grp)
            fnRate = sum(1 for r in grp if r["outcome"] == "FN") / n
            fpRate = sum(1 for r in grp if r["outcome"] == "FP") / n
            err = fnRate - fpRate
            self._integrals[ext] = max(-0.5, min(0.5, self._integrals[ext] + err))
            dTheta = -(self.kp * err + self.ki * self._integrals[ext])
            overrides[ext] = {"d_theta": dTheta, "d_alpha": 0.0, "d_beta": 0.0, "d_T": 0.0}

        return clampAndValidateOutput(
            {
                "global_delta": _zeroDelta(),
                "group_overrides": overrides,
                "recommended_action": "adjust_params",
                "diagnosis": "per_group_pid",
            }
        )


class DeterministicRuleOracle(CalibratorBackend):
    """Offline CI oracle that diagnoses drift & structural confusion solely from sanitized arrays."""

    def propose(self, sanitizedRecords: list[dict[str, Any]]) -> dict[str, Any]:
        if not sanitizedRecords:
            return {"recommended_action": "noop", "diagnosis": "empty"}

        byGroup: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in sanitizedRecords:
            byGroup[rec["ext_enum"]].append(rec)

        overrides: dict[str, dict[str, float]] = {}
        targetGroups: list[str] = []
        diagnoses: list[str] = []
        triggerReplay = False
        highFnGroups: list[str] = []

        for ext, grp in byGroup.items():
            n = len(grp)
            fns = [r for r in grp if r["outcome"] == "FN"]
            fps = [r for r in grp if r["outcome"] == "FP"]
            tps = [r for r in grp if r["outcome"] == "TP"]
            fnRate = len(fns) / n
            fpRate = len(fps) / n

            # 1. Detect Structural Label-Swap (high-Goodness confident FP flood in subgroup)
            if fpRate >= 0.16 and len(fps) >= 3:
                meanFpG = sum(r["raw_goodness_G"] for r in fps) / len(fps)
                meanTpG = (sum(r["raw_goodness_G"] for r in tps) / len(tps)) if tps else meanFpG
                if meanFpG >= 0.70 * meanTpG:
                    overrides[ext] = {
                        "d_theta": -0.02,
                        "d_alpha": 0.20,
                        "d_beta": 0.20,
                        "d_T": 0.05,
                    }
                    targetGroups.append(ext)
                    diagnoses.append(f"structural_label_swap:{ext}")
                    triggerReplay = True
                    continue

            # 2. Track elevated FN groups
            if fnRate >= 0.10:
                highFnGroups.append(ext)

        # Distinguish global decay (>= 3 groups with elevated FN) vs subgroup drift (1-2 groups)
        globalDelta = _zeroDelta()
        if len(highFnGroups) >= 3:
            totalN = len(sanitizedRecords)
            gFn = sum(1 for r in sanitizedRecords if r["outcome"] == "FN") / totalN
            gFp = sum(1 for r in sanitizedRecords if r["outcome"] == "FP") / totalN
            globalDelta["d_theta"] = -min(0.15, 0.75 * max(0.0, gFn - gFp))
            globalDelta["d_alpha"] = 0.10
            diagnoses.append("global_decay")
        else:
            for ext in highFnGroups:
                grp = byGroup[ext]
                fnRate = sum(1 for r in grp if r["outcome"] == "FN") / len(grp)
                overrides[ext] = {
                    "d_theta": -min(0.15, 0.60 * fnRate),
                    "d_alpha": 0.15,
                    "d_beta": 0.05,
                    "d_T": 0.0,
                }
                targetGroups.append(ext)
                diagnoses.append(f"subgroup_drift:{ext}")
                triggerReplay = True

        # Mild per-group FP trimming for non-swapped groups
        for ext, grp in byGroup.items():
            if ext in overrides or len(highFnGroups) >= 3:
                continue
            n = len(grp)
            fnRate = sum(1 for r in grp if r["outcome"] == "FN") / n
            fpRate = sum(1 for r in grp if r["outcome"] == "FP") / n
            if abs(fnRate - fpRate) >= 0.06:
                overrides[ext] = {
                    "d_theta": -0.40 * (fnRate - fpRate),
                    "d_alpha": 0.0,
                    "d_beta": 0.0,
                    "d_T": 0.0,
                }

        action = "trigger_replay" if triggerReplay else ("adjust_params" if (overrides or globalDelta["d_theta"] != 0.0) else "noop")
        return {
            "global_delta": globalDelta,
            "group_overrides": overrides,
            "recommended_action": action,
            "target_groups": targetGroups,
            "diagnosis": ";".join(diagnoses) if diagnoses else "nominal",
        }


def _buildCalibratorPrompt(sanitizedRecords: list[dict[str, Any]]) -> str:
    """Build a compact, sovereignty-safe prompt from sanitized telemetry summaries."""
    byGroup: dict[str, dict[str, Any]] = {}
    for ext in EXT_ENUMS:
        grp = [r for r in sanitizedRecords if r["ext_enum"] == ext]
        if not grp:
            continue
        tps = [r for r in grp if r["outcome"] == "TP"]
        fps = [r for r in grp if r["outcome"] == "FP"]
        fns = [r for r in grp if r["outcome"] == "FN"]
        pairs = Counter((r["pred_cat"], r["actual_cat"]) for r in fps).most_common(2)
        byGroup[ext] = {
            "n": len(grp),
            "tp": len(tps),
            "fp": len(fps),
            "fn": len(fns),
            "mean_tp_G": round(sum(r["raw_goodness_G"] for r in tps) / len(tps), 3) if tps else 0.0,
            "mean_fp_G": round(sum(r["raw_goodness_G"] for r in fps) / len(fps), 3) if fps else 0.0,
            "top_fp_pairs": [[p[0][0], p[0][1], p[1]] for p in pairs],
        }

    return (
        "You are a Forward-Forward classifier meta-calibrator. "
        "Analyze ONLY the sanitized per-group telemetry summary below and output ONLY a valid JSON object "
        "(no markdown, no explanation) with keys:\n"
        '{"global_delta":{"d_theta":float,"d_alpha":float,"d_beta":float,"d_T":float},'
        '"group_overrides":{"<ext_enum>":{"d_theta":float,"d_alpha":float,"d_beta":float,"d_T":float}},'
        '"recommended_action":"adjust_params"|"trigger_replay"|"noop",'
        '"target_groups":["<ext_enum>"],'
        '"diagnosis":"short_tag"}\n'
        "Rules:\n"
        "- Ignore minor FP noise (fp/n < 0.15 or fp < 3): return 'noop' if all groups are healthy.\n"
        "- If >=3 groups have high FN, set global_delta.d_theta negative (global_decay) AND still check each group for concentrated top_fp_pairs.\n"
        "- If 1-2 groups have high FN (others healthy), set group_overrides (d_theta negative, d_alpha positive) and recommended_action='trigger_replay' with target_groups.\n"
        "- If any group has fp/n >= 0.15 with concentrated top_fp_pairs, set group_overrides (d_alpha=0.2, d_beta=0.2) and recommended_action='trigger_replay' with target_groups (even during global_decay).\n"
        "- Bounds: d_theta in [-0.15,0.15], d_alpha/d_beta/d_T in [-0.2,0.2].\n"
        f"Summary:\n{json.dumps(byGroup)}"
    )


class PiCliBackend(CalibratorBackend):
    """Unit 4 live LLM backend invoking Pi CLI non-interactively with hermetic flags."""

    def __init__(
        self,
        model: str = "OPENGATEWAY/deepseek/deepseek-v4.1-flash-eco",
        timeoutSec: float = 30.0,
    ) -> None:
        self.model = model
        self.timeoutSec = timeoutSec
        self.totalCalls: int = 0
        self.parseSuccesses: int = 0
        self.clampCount: int = 0

    def propose(self, sanitizedRecords: list[dict[str, Any]]) -> str:
        self.totalCalls += 1
        prompt = _buildCalibratorPrompt(sanitizedRecords)
        cmd = [
            "pi",
            "-p",
            "--no-session",
            "--no-mcp",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-tools",
            "--model",
            self.model,
            "--",
            prompt,
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.timeoutSec,
                check=False,
            )
            if proc.returncode != 0:
                return ""
            return proc.stdout.strip()
        except Exception:
            return ""


class OpenAiCompatibleBackend(CalibratorBackend):
    """Stdlib HTTP backend for Unit 4 (Ollama or envrun-injected OpenAI-compatible API)."""

    def __init__(self, timeoutSec: float = 20.0) -> None:
        self.baseUrl = os.environ.get("LLM_API_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
        self.apiKey = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "qwen2.5:7b")
        self.timeoutSec = timeoutSec

    def propose(self, sanitizedRecords: list[dict[str, Any]]) -> str:
        prompt = (
            "You are a Forward-Forward classifier meta-calibrator. "
            "Analyze the sanitized telemetry array and output ONLY a JSON object with keys: "
            "global_delta, group_overrides, recommended_action (adjust_params|trigger_replay|noop), "
            "target_groups, diagnosis.\nTelemetry:\n"
            + json.dumps(sanitizedRecords[-64:])
        )
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.apiKey:
            headers["Authorization"] = f"Bearer {self.apiKey}"
        req = urllib.request.Request(
            f"{self.baseUrl}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeoutSec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])


class ClampedLlmCalibrator:
    """Enforces Sovereignty Sanitizer before backend call and Hard Clamping after."""

    def __init__(self, backend: CalibratorBackend | None = None) -> None:
        self.backend: CalibratorBackend = backend or DeterministicRuleOracle()
        self.lastParseOk: bool = True

    def calibrate(self, records: list[dict[str, Any]]) -> CalibrationCommand:
        sanitized = sanitizeTelemetryArray(records)
        rawProposal = self.backend.propose(sanitized)
        cmd = clampAndValidateOutput(rawProposal)
        self.lastParseOk = not cmd.diagnosis.startswith("parse_fallback:")
        if isinstance(self.backend, PiCliBackend) and self.lastParseOk:
            self.backend.parseSuccesses += 1
        return cmd
