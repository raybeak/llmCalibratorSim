"""3-Scenario drift environment and Activity-Gated Implicit Feedback (Unit 1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from llmCalibratorSim import CATEGORIES, EXT_ENUMS, FEATURE_DIM

NUM_CLASSES: int = len(CATEGORIES)
NUM_EXTS: int = len(EXT_ENUMS)


@dataclass
class Sample:
    x: np.ndarray
    trueCatIdx: int
    extEnum: str
    goodnessScale: float = 1.0


class DatasetSource(ABC):
    """Abstract dataset source allowing plug-in of synthetic or real pre-computed vectors."""

    @abstractmethod
    def sampleBatch(self, n: int, step: int, scenario: str) -> list[Sample]:
        raise NotImplementedError


class SyntheticClusterDataset(DatasetSource):
    """Deterministic 4-group x 4-class clustered feature stream with 3 controlled failure modes."""

    def __init__(self, seed: int = 42, driftStartStep: int = 100) -> None:
        self.rng = np.random.default_rng(seed)
        self.driftStartStep = driftStartStep

        # Shared semantic centroids in dims 4..31 (well-separated across 4 categories)
        rawCentroids = self.rng.normal(0.0, 1.0, size=(NUM_CLASSES, FEATURE_DIM - NUM_EXTS))
        rawCentroids /= np.linalg.norm(rawCentroids, axis=1, keepdims=True) + 1e-8
        self.baseCentroids = rawCentroids * 1.35

        # Deterministic drift shift vector for doc_hwpx subgroup drift
        shift = self.rng.normal(0.0, 1.0, size=(NUM_CLASSES, FEATURE_DIM - NUM_EXTS))
        shift /= np.linalg.norm(shift, axis=1, keepdims=True) + 1e-8
        self.subgroupShift = shift * 0.95

        self.extEye = np.eye(NUM_EXTS, dtype=np.float64) * 1.15

    def sampleBatch(self, n: int, step: int, scenario: str) -> list[Sample]:
        samples: list[Sample] = []
        afterDrift = step >= self.driftStartStep
        for i in range(n):
            extIdx = i % NUM_EXTS
            extEnum = EXT_ENUMS[extIdx]
            catIdx = int(self.rng.integers(0, NUM_CLASSES))
            semantic = self.baseCentroids[catIdx].copy()
            trueCatIdx = catIdx
            goodnessScale = 1.0

            if afterDrift:
                progress = min(1.0, (step - self.driftStartStep + 1) / 200.0)
                if scenario == "global_decay":
                    goodnessScale = 1.0 - 0.52 * progress
                elif scenario == "subgroup_drift" and extEnum == "doc_hwpx":
                    semantic = semantic + self.subgroupShift[catIdx] * (0.9 + 0.4 * progress)
                    goodnessScale = 0.52
                elif scenario == "structural_label_swap" and extEnum == "doc_slide":
                    trueCatIdx = (catIdx + 1) % NUM_CLASSES

            noisySemantic = semantic + self.rng.normal(0.0, 0.24, size=FEATURE_DIM - NUM_EXTS)
            vec = np.concatenate([self.extEye[extIdx], noisySemantic]).astype(np.float64)
            vec /= float(np.linalg.norm(vec)) + 1e-8
            samples.append(
                Sample(
                    x=vec,
                    trueCatIdx=trueCatIdx,
                    extEnum=extEnum,
                    goodnessScale=goodnessScale,
                )
            )
        return samples


class ActivityGatedFeedback:
    """Confirms implicit TP only after M subsequent user interactions, with configurable label noise."""

    def __init__(self, confirmAfterSteps: int = 3, noiseProb: float = 0.05, seed: int = 42) -> None:
        self.confirmAfterSteps = confirmAfterSteps
        self.noiseProb = noiseProb
        self.rng = np.random.default_rng(seed)
        self._pending: deque[dict[str, Any]] = deque()

    def observe(
        self,
        decisionId: int,
        extEnum: str,
        predCat: str,
        trueCat: str,
        accepted: bool,
        rawGoodness: float,
        confidence: float,
        timestampS: int,
    ) -> list[dict[str, Any]]:
        if not accepted:
            outcome = "FN"
        elif predCat == trueCat:
            outcome = "TP"
        else:
            outcome = "FP"

        # Inject realistic implicit feedback noise (e.g., missed user correction recorded as TP)
        reportedOutcome = outcome
        reportedActual = trueCat
        if self.noiseProb > 0.0 and float(self.rng.random()) < self.noiseProb:
            if outcome in ("FP", "FN"):
                reportedOutcome = "TP"
                reportedActual = predCat

        record = {
            "decision_id": decisionId,
            "ext_enum": extEnum,
            "pred_cat": predCat,
            "actual_cat": reportedActual,
            "outcome": reportedOutcome,
            "raw_goodness_G": round(float(rawGoodness), 5),
            "confidence_p": round(float(confidence), 5),
            "timestamp_s": timestampS,
        }
        self._pending.append(record)

        ready: list[dict[str, Any]] = []
        while len(self._pending) > self.confirmAfterSteps:
            ready.append(self._pending.popleft())
        return ready

    def flush(self) -> list[dict[str, Any]]:
        ready = list(self._pending)
        self._pending.clear()
        return ready
