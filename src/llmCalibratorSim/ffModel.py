"""Pure NumPy 2-Layer Forward-Forward classifier with Jev-like calibrated Softmax (Unit 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from llmCalibratorSim import CATEGORIES, EXT_ENUMS, FEATURE_DIM

NUM_CLASSES: int = len(CATEGORIES)
EPS: float = 1e-8


def _sigmoid(z: float) -> float:
    clipped = max(-30.0, min(30.0, float(z)))
    return 1.0 / (1.0 + float(np.exp(-clipped)))


@dataclass
class GroupParams:
    theta: float = 1.65
    alpha: float = 1.0
    beta: float = 1.0
    temperature: float = 0.20

    def applyDelta(self, delta: dict[str, float]) -> None:
        self.theta = float(np.clip(self.theta + delta.get("d_theta", 0.0), 0.2, 5.0))
        self.alpha = float(np.clip(self.alpha + delta.get("d_alpha", 0.0), 0.2, 3.0))
        self.beta = float(np.clip(self.beta + delta.get("d_beta", 0.0), 0.2, 3.0))
        self.temperature = float(np.clip(self.temperature + delta.get("d_T", 0.0), 0.2, 5.0))

    def copy(self) -> GroupParams:
        return GroupParams(
            theta=self.theta,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
        )


def computeEce(confidences: Sequence[float], correctFlags: Sequence[float], nBins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    if not confidences:
        return 0.0
    confArr = np.asarray(confidences, dtype=np.float64)
    corrArr = np.asarray(correctFlags, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, nBins + 1)
    ece = 0.0
    n = len(confArr)
    for b in range(nBins):
        lo, hi = edges[b], edges[b + 1]
        if b == nBins - 1:
            mask = (confArr >= lo) & (confArr <= hi)
        else:
            mask = (confArr >= lo) & (confArr < hi)
        count = int(np.sum(mask))
        if count > 0:
            acc = float(np.mean(corrArr[mask]))
            avgConf = float(np.mean(confArr[mask]))
            ece += (count / n) * abs(acc - avgConf)
    return float(ece)


class ForwardForwardClassifier:
    """2-Layer local Forward-Forward classifier (36 -> 32 -> 16) with zero backpropagation."""

    def __init__(self, seed: int = 42, hidden1: int = 32, hidden2: int = 16) -> None:
        rng = np.random.default_rng(seed)
        inDim = FEATURE_DIM + NUM_CLASSES
        self.w1: np.ndarray = rng.normal(0.0, 0.75, size=(hidden1, inDim)).astype(np.float64)
        self.b1: np.ndarray = np.zeros(hidden1, dtype=np.float64)
        w2Init = rng.normal(0.0, 0.75, size=(hidden2, hidden1 + NUM_CLASSES)).astype(np.float64)
        self.w2: dict[str, np.ndarray] = {ext: w2Init.copy() for ext in EXT_ENUMS}
        self.b2: dict[str, np.ndarray] = {ext: np.zeros(hidden2, dtype=np.float64) for ext in EXT_ENUMS}
        self.groupParams: dict[str, GroupParams] = {ext: GroupParams() for ext in EXT_ENUMS}
        self._eyeClasses: np.ndarray = np.eye(NUM_CLASSES, dtype=np.float64)

    def copy(self) -> ForwardForwardClassifier:
        clone = ForwardForwardClassifier.__new__(ForwardForwardClassifier)
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = {k: v.copy() for k, v in self.w2.items()}
        clone.b2 = {k: v.copy() for k, v in self.b2.items()}
        clone.groupParams = {k: v.copy() for k, v in self.groupParams.items()}
        clone._eyeClasses = self._eyeClasses
        return clone

    def _forwardSingle(
        self, x: np.ndarray, catIdx: int, extEnum: str
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
        x1Raw = np.concatenate([x, self._eyeClasses[catIdx]])
        x1Hat = x1Raw / (float(np.linalg.norm(x1Raw)) + EPS)
        h1 = np.maximum(0.0, self.w1 @ x1Hat + self.b1)
        g1 = float(np.mean(h1 * h1))

        x2Raw = np.concatenate([h1, self._eyeClasses[catIdx]])
        x2Hat = x2Raw / (float(np.linalg.norm(x2Raw)) + EPS)
        h2 = np.maximum(0.0, self.w2[extEnum] @ x2Hat + self.b2[extEnum])
        g2 = float(np.mean(h2 * h2))
        return x1Hat, h1, g1, x2Hat, h2, g2

    def scoreAllClasses(self, x: np.ndarray, extEnum: str, goodnessScale: float = 1.0) -> np.ndarray:
        xTile = np.broadcast_to(x, (NUM_CLASSES, FEATURE_DIM))
        x1Raw = np.concatenate([xTile, self._eyeClasses], axis=1)
        x1Hat = x1Raw / (np.linalg.norm(x1Raw, axis=1, keepdims=True) + EPS)
        h1 = np.maximum(0.0, x1Hat @ self.w1.T + self.b1)
        g1 = np.mean(h1 * h1, axis=1)

        x2Raw = np.concatenate([h1, self._eyeClasses], axis=1)
        x2Hat = x2Raw / (np.linalg.norm(x2Raw, axis=1, keepdims=True) + EPS)
        h2 = np.maximum(0.0, x2Hat @ self.w2[extEnum].T + self.b2[extEnum])
        g2 = np.mean(h2 * h2, axis=1)
        return (g1 + g2) * goodnessScale

    def predict(
        self,
        x: np.ndarray,
        extEnum: str,
        goodnessScale: float = 1.0,
    ) -> tuple[int, bool, float, float, np.ndarray]:
        params = self.groupParams[extEnum]
        goodnessAll = self.scoreAllClasses(x, extEnum=extEnum, goodnessScale=goodnessScale)
        scaled = goodnessAll / params.temperature
        scaled -= float(np.max(scaled))
        expVals = np.exp(scaled)
        probs = expVals / float(np.sum(expVals))

        bestIdx = int(np.argmax(goodnessAll))
        bestG = float(goodnessAll[bestIdx])
        accepted = bestG >= params.theta
        conf = float(probs[bestIdx])
        return bestIdx, accepted, bestG, conf, probs

    def trainStep(
        self,
        x: np.ndarray,
        trueCatIdx: int,
        extEnum: str,
        lr: float = 0.08,
        weightDecay: float = 5e-4,
        freezeLayer1: bool = False,
    ) -> None:
        """Perform local layer-wise Positive + Hard-Negative FF update without backprop."""
        params = self.groupParams[extEnum]
        layerTheta = params.theta * 0.5

        # Pick Hard Negative (highest Goodness among wrong classes)
        goodnessAll = self.scoreAllClasses(x, extEnum=extEnum)
        goodnessAll[trueCatIdx] = -1e9
        negIdx = int(np.argmax(goodnessAll))

        x1Pos, h1Pos, g1Pos, x2Pos, h2Pos, g2Pos = self._forwardSingle(x, trueCatIdx, extEnum)
        x1Neg, h1Neg, g1Neg, x2Neg, h2Neg, g2Neg = self._forwardSingle(x, negIdx, extEnum)

        if not freezeLayer1:
            p1Pos = _sigmoid(g1Pos - layerTheta)
            p1Neg = _sigmoid(g1Neg - layerTheta)
            norm1 = 2.0 / float(np.sqrt(len(h1Pos)))
            scale1Pos = lr * params.alpha * (1.0 - p1Pos) * norm1
            scale1Neg = lr * params.beta * p1Neg * norm1
            dW1 = scale1Pos * np.outer(h1Pos, x1Pos) - scale1Neg * np.outer(h1Neg, x1Neg)
            db1 = scale1Pos * h1Pos - scale1Neg * h1Neg
            self.w1 = (1.0 - weightDecay) * self.w1 + dW1
            self.b1 = self.b1 + db1

        # Layer 2 (group-local head) local update
        p2Pos = _sigmoid(g2Pos - layerTheta)
        p2Neg = _sigmoid(g2Neg - layerTheta)
        norm2 = 2.0 / float(np.sqrt(len(h2Pos)))
        scale2Pos = lr * params.alpha * (1.0 - p2Pos) * norm2
        scale2Neg = lr * params.beta * p2Neg * norm2
        dW2 = scale2Pos * np.outer(h2Pos, x2Pos) - scale2Neg * np.outer(h2Neg, x2Neg)
        db2 = scale2Pos * h2Pos - scale2Neg * h2Neg
        self.w2[extEnum] = (1.0 - weightDecay) * self.w2[extEnum] + dW2
        self.b2[extEnum] = self.b2[extEnum] + db2

    def replayUpdate(
        self,
        samples: Sequence[tuple[np.ndarray, int, str]],
        lr: float = 0.12,
        epochs: int = 6,
    ) -> None:
        """Targeted local FF replay updating group-local Layer 2 heads without forgetting shared Layer 1."""
        for _ in range(epochs):
            for x, trueIdx, extEnum in samples:
                self.trainStep(x, trueIdx, extEnum, lr=lr, freezeLayer1=True)
