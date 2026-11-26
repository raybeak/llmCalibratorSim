"""5-Seed x 3-Scenario x 3-Calibrator comparative benchmark runner (Unit 3 / MOP 1)."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np

from llmCalibratorSim import CATEGORIES, EXT_ENUMS
from llmCalibratorSim.calibrators import (
    ClampedLlmCalibrator,
    GlobalPidCalibrator,
    PerGroupPidCalibrator,
    PiCliBackend,
)
from llmCalibratorSim.driftEnv import ActivityGatedFeedback, Sample, SyntheticClusterDataset
from llmCalibratorSim.ffModel import ForwardForwardClassifier, computeEce

SCENARIOS: tuple[str, ...] = ("global_decay", "subgroup_drift", "structural_label_swap")
CALIBRATOR_NAMES: tuple[str, ...] = ("global_pid", "per_group_pid", "clamped_llm")


def _buildCalibrator(name: str, liveModel: str | None = None) -> Any:
    if name == "global_pid":
        return GlobalPidCalibrator()
    if name == "per_group_pid":
        return PerGroupPidCalibrator()
    if name == "clamped_llm":
        if liveModel:
            return ClampedLlmCalibrator(backend=PiCliBackend(model=liveModel))
        return ClampedLlmCalibrator()
    raise ValueError(f"Unknown calibrator: {name}")


def _evalBatch(
    model: ForwardForwardClassifier,
    samples: list[Sample],
) -> tuple[float, float]:
    correctFlags: list[float] = []
    confidences: list[float] = []
    for sample in samples:
        predIdx, accepted, _, conf, _ = model.predict(
            sample.x, sample.extEnum, goodnessScale=sample.goodnessScale
        )
        correctFlags.append(1.0 if (accepted and predIdx == sample.trueCatIdx) else 0.0)
        confidences.append(conf)
    acc = sum(correctFlags) / len(correctFlags) if correctFlags else 0.0
    ece = computeEce(confidences, correctFlags)
    return acc, ece


def _runSingleEpisode(
    baseModel: ForwardForwardClassifier,
    anchorBuffer: list[tuple[np.ndarray, int, str]],
    scenario: str,
    calibratorName: str,
    seed: int,
    nCycles: int = 8,
    samplesPerCycle: int = 48,
    liveModel: str | None = None,
) -> dict[str, float]:
    model = baseModel.copy()
    calibrator = _buildCalibrator(calibratorName, liveModel=liveModel)
    env = SyntheticClusterDataset(seed=seed, driftStartStep=100)
    feedback = ActivityGatedFeedback(confirmAfterSteps=3, noiseProb=0.05, seed=seed + 100)

    # Pre-drift baseline accuracy
    preSamples = env.sampleBatch(n=128, step=10, scenario=scenario)
    preAcc, _ = _evalBatch(model, preSamples)

    recentCorrections: deque[tuple[np.ndarray, int, str]] = deque(maxlen=128)
    decisionId = 0
    postAccCycle1 = 0.0
    cycleAccs: list[float] = []
    finalEce = 0.0
    recoveryCycles = float(nCycles)

    for cycle in range(nCycles):
        step = 100 + cycle * 35
        batch = env.sampleBatch(n=samplesPerCycle, step=step, scenario=scenario)
        cycleLogs: list[dict[str, Any]] = []
        correctFlags: list[float] = []
        confidences: list[float] = []

        for sample in batch:
            decisionId += 1
            predIdx, accepted, bestG, conf, _ = model.predict(
                sample.x, sample.extEnum, goodnessScale=sample.goodnessScale
            )
            isCorrect = accepted and (predIdx == sample.trueCatIdx)
            correctFlags.append(1.0 if isCorrect else 0.0)
            confidences.append(conf)

            ready = feedback.observe(
                decisionId=decisionId,
                extEnum=sample.extEnum,
                predCat=CATEGORIES[predIdx],
                trueCat=CATEGORIES[sample.trueCatIdx],
                accepted=accepted,
                rawGoodness=bestG,
                confidence=conf,
                timestampS=1764140000 + decisionId * 10,
            )
            cycleLogs.extend(ready)

            # Store corrected samples locally (never sent externally) + light passive online step
            if not isCorrect:
                recentCorrections.append((sample.x, sample.trueCatIdx, sample.extEnum))
                model.trainStep(sample.x, sample.trueCatIdx, sample.extEnum, lr=0.012, freezeLayer1=True)

        if cycle == nCycles - 1:
            cycleLogs.extend(feedback.flush())

        cycleAcc = sum(correctFlags) / len(correctFlags)
        cycleAccs.append(cycleAcc)
        if cycle == 0:
            postAccCycle1 = cycleAcc
        if cycleAcc >= preAcc - 0.05 and recoveryCycles == float(nCycles):
            recoveryCycles = float(cycle + 1)
        finalEce = computeEce(confidences, correctFlags)

        # Invoke calibrator on cycle telemetry
        cmd = calibrator.calibrate(cycleLogs)
        for ext in EXT_ENUMS:
            model.groupParams[ext].applyDelta(cmd.globalDelta)
        for ext, delta in cmd.groupOverrides.items():
            model.groupParams[ext].applyDelta(delta)

        # Targeted local FF replay on group-local Layer 2 heads (zero interference on other groups)
        if cmd.recommendedAction == "trigger_replay" and recentCorrections:
            targets = set(cmd.targetGroups) if cmd.targetGroups else set(EXT_ENUMS)
            focused = [s for s in recentCorrections if s[2] in targets]
            if focused:
                model.replayUpdate(focused, lr=0.14, epochs=8)

    parseRate = 1.0
    if isinstance(getattr(calibrator, "backend", None), PiCliBackend) and calibrator.backend.totalCalls > 0:
        parseRate = calibrator.backend.parseSuccesses / calibrator.backend.totalCalls

    return {
        "preDriftAcc": preAcc,
        "postDriftAcc": postAccCycle1,
        "finalAcc": float(np.mean(cycleAccs[-2:])),
        "finalEce": finalEce,
        "recoveryCycles": recoveryCycles,
        "parseRate": parseRate,
    }


def runFullBenchmark(
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    nCycles: int = 8,
    liveModel: str | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Run paired multi-seed x 3-scenario x 3-calibrator benchmark."""
    agg: dict[tuple[str, str], list[dict[str, float]]] = {
        (sc, cal): [] for sc in SCENARIOS for cal in CALIBRATOR_NAMES
    }

    for seed in seeds:
        dataset = SyntheticClusterDataset(seed=seed, driftStartStep=100)
        baseModel = ForwardForwardClassifier(seed=seed)
        trainSamples = dataset.sampleBatch(n=640, step=0, scenario="none")
        for _ in range(4):
            for sample in trainSamples:
                baseModel.trainStep(sample.x, sample.trueCatIdx, sample.extEnum, lr=0.08)
        anchorBuffer = [(s.x, s.trueCatIdx, s.extEnum) for s in trainSamples[:128]]

        for scenario in SCENARIOS:
            for calName in CALIBRATOR_NAMES:
                res = _runSingleEpisode(
                    baseModel=baseModel,
                    anchorBuffer=anchorBuffer,
                    scenario=scenario,
                    calibratorName=calName,
                    seed=seed,
                    nCycles=nCycles,
                    liveModel=liveModel if calName == "clamped_llm" else None,
                )
                agg[(scenario, calName)].append(res)

    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, runs in agg.items():
        summary[key] = {
            metric: float(np.mean([r[metric] for r in runs]))
            for metric in ("preDriftAcc", "postDriftAcc", "finalAcc", "finalEce", "recoveryCycles", "parseRate")
        }
    return summary


def main() -> None:
    import sys

    liveModel: str | None = None
    seeds = (42, 43, 44, 45, 46)
    nCycles = 8
    if "--live-pi" in sys.argv:
        idx = sys.argv.index("--live-pi")
        liveModel = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "OPENGATEWAY/deepseek/deepseek-v4.1-flash-eco"
        seeds = (42,)
        nCycles = 6

    t0 = time.perf_counter()
    summary = runFullBenchmark(seeds=seeds, nCycles=nCycles, liveModel=liveModel)
    elapsed = time.perf_counter() - t0
    print("scenario;calibrator;preDriftAcc;postDriftAcc;finalAcc;finalEce;recoveryCycles;parseRate")
    for scenario in SCENARIOS:
        for calName in CALIBRATOR_NAMES:
            m = summary[(scenario, calName)]
            print(
                f"{scenario};{calName};{m['preDriftAcc']:.3f};{m['postDriftAcc']:.3f};"
                f"{m['finalAcc']:.3f};{m['finalEce']:.3f};{m['recoveryCycles']:.2f};{m['parseRate']:.2f}"
            )
    print(f"elapsedSec={elapsed:.3f}")


if __name__ == "__main__":
    main()
