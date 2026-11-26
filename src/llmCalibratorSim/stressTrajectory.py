"""Long-horizon Train/Test trajectory stress test & overshoot finder (GLM vs. PID)."""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from llmCalibratorSim import CATEGORIES, EXT_ENUMS
from llmCalibratorSim.calibrators import (
    ClampedLlmCalibrator,
    PerGroupPidCalibrator,
    PiCliBackend,
)
from llmCalibratorSim.driftEnv import ActivityGatedFeedback, Sample, SyntheticClusterDataset
from llmCalibratorSim.ffModel import ForwardForwardClassifier, computeEce


@dataclass
class TrajectoryPoint:
    cycle: int
    phase: str
    calibrator: str
    trainAcc: float
    testAcc: float
    trainTestGap: float
    testEce: float
    meanTheta: float
    slideTheta: float
    action: str
    diagnosis: str


def _sampleCompoundBatch(
    env: SyntheticClusterDataset,
    n: int,
    cycle: int,
    rng: np.random.Generator,
) -> tuple[str, list[Sample]]:
    """Multi-phase escalating shock trajectory: clean -> subgroup -> swap -> reversal+decay."""
    if cycle < 3:
        phase = "clean"
        samples = env.sampleBatch(n=n, step=10 + cycle * 10, scenario="none")
        return phase, samples

    if cycle < 7:
        phase = "subgroup_drift"
        samples = env.sampleBatch(n=n, step=100 + (cycle - 3) * 55, scenario="subgroup_drift")
        return phase, samples

    if cycle < 11:
        phase = "label_swap"
        samples = env.sampleBatch(n=n, step=100 + (cycle - 7) * 55, scenario="structural_label_swap")
        return phase, samples

    # Phase 4 (cycles 11..13): compound reversal + global decay + doc_md swap
    phase = "compound_shock"
    samples = env.sampleBatch(n=n, step=220 + (cycle - 11) * 40, scenario="global_decay")
    for s in samples:
        if s.extEnum == "doc_md":
            s.trueCatIdx = (s.trueCatIdx + 2) % len(CATEGORIES)
        s.goodnessScale *= 0.72
        if float(rng.random()) < 0.05:
            s.trueCatIdx = int(rng.integers(0, len(CATEGORIES)))
    return phase, samples


def _evalSplit(model: ForwardForwardClassifier, samples: list[Sample]) -> tuple[float, float]:
    correct: list[float] = []
    confs: list[float] = []
    for s in samples:
        predIdx, accepted, _, conf, _ = model.predict(s.x, s.extEnum, goodnessScale=s.goodnessScale)
        correct.append(1.0 if (accepted and predIdx == s.trueCatIdx) else 0.0)
        confs.append(conf)
    acc = sum(correct) / len(correct) if correct else 0.0
    return acc, computeEce(confs, correct)


def runTrajectoryEpisode(
    calibratorName: str,
    calibrator: Any,
    seed: int = 42,
    nCycles: int = 14,
    batchSize: int = 48,
    replayLr: float = 0.14,
    replayEpochs: int = 8,
    unclampedMultiplier: float = 1.0,
) -> list[TrajectoryPoint]:
    """Run a 14-cycle Train + Held-Out Test trajectory and log every cycle."""
    trainEnv = SyntheticClusterDataset(seed=seed, driftStartStep=100)
    testEnv = SyntheticClusterDataset(seed=seed, driftStartStep=100)
    # Advance testEnv RNG independently so test samples are strictly unseen
    testEnv.rng = np.random.default_rng(seed + 9000)
    trainShockRng = np.random.default_rng(seed + 500)
    testShockRng = np.random.default_rng(seed + 9500)

    model = ForwardForwardClassifier(seed=seed)
    preTrain = trainEnv.sampleBatch(n=640, step=0, scenario="none")
    for _ in range(4):
        for s in preTrain:
            model.trainStep(s.x, s.trueCatIdx, s.extEnum, lr=0.08)

    feedback = ActivityGatedFeedback(confirmAfterSteps=3, noiseProb=0.10, seed=seed + 200)
    recentCorrections: deque[tuple[np.ndarray, int, str]] = deque(maxlen=128)
    trajectory: list[TrajectoryPoint] = []
    decisionId = 0

    for cycle in range(nCycles):
        phase, trainBatch = _sampleCompoundBatch(trainEnv, batchSize, cycle, trainShockRng)
        _, testBatch = _sampleCompoundBatch(testEnv, batchSize, cycle, testShockRng)

        # Evaluate held-out test BEFORE cycle updates (true generalization snapshot)
        testAcc, testEce = _evalSplit(model, testBatch)

        cycleLogs: list[dict[str, Any]] = []
        trainCorrect: list[float] = []
        for s in trainBatch:
            decisionId += 1
            predIdx, accepted, bestG, conf, _ = model.predict(
                s.x, s.extEnum, goodnessScale=s.goodnessScale
            )
            isCorrect = accepted and (predIdx == s.trueCatIdx)
            trainCorrect.append(1.0 if isCorrect else 0.0)
            ready = feedback.observe(
                decisionId=decisionId,
                extEnum=s.extEnum,
                predCat=CATEGORIES[predIdx],
                trueCat=CATEGORIES[s.trueCatIdx],
                accepted=accepted,
                rawGoodness=bestG,
                confidence=conf,
                timestampS=1764150000 + decisionId * 10,
            )
            cycleLogs.extend(ready)
            if not isCorrect:
                recentCorrections.append((s.x, s.trueCatIdx, s.extEnum))
                model.trainStep(s.x, s.trueCatIdx, s.extEnum, lr=0.015, freezeLayer1=True)

        if cycle == nCycles - 1:
            cycleLogs.extend(feedback.flush())

        trainAcc = sum(trainCorrect) / len(trainCorrect)
        cmd = calibrator.calibrate(cycleLogs)

        for ext in EXT_ENUMS:
            scaled = {k: v * unclampedMultiplier for k, v in cmd.globalDelta.items()}
            model.groupParams[ext].applyDelta(scaled)
        for ext, delta in cmd.groupOverrides.items():
            scaled = {k: v * unclampedMultiplier for k, v in delta.items()}
            model.groupParams[ext].applyDelta(scaled)

        if cmd.recommendedAction == "trigger_replay" and recentCorrections:
            targets = set(cmd.targetGroups) if cmd.targetGroups else set(EXT_ENUMS)
            focused = [s for s in recentCorrections if s[2] in targets]
            if focused:
                model.replayUpdate(focused, lr=replayLr, epochs=replayEpochs)

        meanTheta = float(np.mean([model.groupParams[e].theta for e in EXT_ENUMS]))
        slideTheta = model.groupParams["doc_slide"].theta
        trajectory.append(
            TrajectoryPoint(
                cycle=cycle,
                phase=phase,
                calibrator=calibratorName,
                trainAcc=trainAcc,
                testAcc=testAcc,
                trainTestGap=trainAcc - testAcc,
                testEce=testEce,
                meanTheta=meanTheta,
                slideTheta=slideTheta,
                action=cmd.recommendedAction,
                diagnosis=cmd.diagnosis[:48],
            )
        )
    return trajectory


def detectOvershoot(points: list[TrajectoryPoint]) -> dict[str, Any]:
    """Detect parameter oscillation and post-peak Train/Test overshoot collapse."""
    thetas = [p.slideTheta for p in points]
    diffs = [thetas[i + 1] - thetas[i] for i in range(len(thetas) - 1)]
    signFlips = sum(
        1
        for i in range(len(diffs) - 1)
        if abs(diffs[i]) > 0.02 and abs(diffs[i + 1]) > 0.02 and (diffs[i] * diffs[i + 1] < 0)
    )
    peakTest = max(p.testAcc for p in points)
    endTest = float(np.mean([p.testAcc for p in points[-2:]]))
    endTrain = float(np.mean([p.trainAcc for p in points[-2:]]))
    maxGap = max(abs(p.trainTestGap) for p in points)
    overshoot = (signFlips >= 2) or ((peakTest - endTest) >= 0.25 and endTest < 0.70)
    return {
        "signFlips": signFlips,
        "peakTestAcc": peakTest,
        "endTrainAcc": endTrain,
        "endTestAcc": endTest,
        "maxTrainTestGap": maxGap,
        "overshoot": overshoot,
    }


def runPidOvershootSweep(seed: int = 42) -> list[tuple[float, float, dict[str, Any]]]:
    """Sweep PerGroupPidCalibrator gains upward until overshoot/oscillation occurs."""
    results: list[tuple[float, float, dict[str, Any]]] = []
    for kp, ki in ((0.50, 0.10), (1.50, 0.30), (3.50, 0.80), (8.00, 2.00)):
        pid = PerGroupPidCalibrator(kp=kp, ki=ki)
        traj = runTrajectoryEpisode(
            calibratorName=f"pid_kp{kp}",
            calibrator=pid,
            seed=seed,
            unclampedMultiplier=max(1.0, kp / 1.5),
        )
        results.append((kp, ki, detectOvershoot(traj)))
    return results


def main() -> None:
    useLiveGlm = "--live-glm" in sys.argv
    seed = 42
    t0 = time.perf_counter()

    print("=== 1. PID GAIN SWEEP UNTIL OVERSHOOT (14-Cycle Compound Trajectory, Train vs Held-Out Test) ===")
    print("kp;ki;endTrainAcc;endTestAcc;peakTestAcc;maxGap;thetaSignFlips;overshoot")
    for kp, ki, stats in runPidOvershootSweep(seed=seed):
        print(
            f"{kp:.2f};{ki:.2f};{stats['endTrainAcc']:.3f};{stats['endTestAcc']:.3f};"
            f"{stats['peakTestAcc']:.3f};{stats['maxTrainTestGap']:.3f};"
            f"{stats['signFlips']};{stats['overshoot']}"
        )

    print("\n=== 2. 14-CYCLE TRAJECTORY: Per-Group PID vs. GLM-5.3 vs. Aggressive-Replay GLM ===")
    configs: list[tuple[str, Any, float, int, float]] = [
        ("per_group_pid", PerGroupPidCalibrator(kp=0.50, ki=0.10), 0.14, 8, 1.0),
    ]
    if useLiveGlm:
        glmBackend = PiCliBackend(model="OPENGATEWAY/z-ai/glm-5.3-flash-ultrafast", timeoutSec=30.0)
        configs.append(("glm_5.3_clamped", ClampedLlmCalibrator(backend=glmBackend), 0.14, 8, 1.0))
        glmAggBackend = PiCliBackend(model="OPENGATEWAY/z-ai/glm-5.3-flash-ultrafast", timeoutSec=30.0)
        configs.append(("glm_5.3_overshoot_push", ClampedLlmCalibrator(backend=glmAggBackend), 0.38, 18, 3.0))
    else:
        configs.append(("oracle_clamped", ClampedLlmCalibrator(), 0.14, 8, 1.0))
        configs.append(("oracle_overshoot_push", ClampedLlmCalibrator(), 0.38, 18, 3.0))

    print("cycle;phase;calibrator;trainAcc;testAcc;gap;testEce;meanTheta;slideTheta;action;diagnosis")
    summaries: list[tuple[str, dict[str, Any]]] = []
    for name, cal, rLr, rEpochs, mult in configs:
        traj = runTrajectoryEpisode(
            calibratorName=name,
            calibrator=cal,
            seed=seed,
            replayLr=rLr,
            replayEpochs=rEpochs,
            unclampedMultiplier=mult,
        )
        summaries.append((name, detectOvershoot(traj)))
        for p in traj:
            print(
                f"{p.cycle};{p.phase};{p.calibrator};{p.trainAcc:.3f};{p.testAcc:.3f};"
                f"{p.trainTestGap:+.3f};{p.testEce:.3f};{p.meanTheta:.2f};{p.slideTheta:.2f};"
                f"{p.action};{p.diagnosis}"
            )

    print("\n=== 3. TRAJECTORY SUMMARY ===")
    print("calibrator;endTrainAcc;endTestAcc;peakTestAcc;maxTrainTestGap;thetaSignFlips;overshoot")
    for name, st in summaries:
        print(
            f"{name};{st['endTrainAcc']:.3f};{st['endTestAcc']:.3f};{st['peakTestAcc']:.3f};"
            f"{st['maxTrainTestGap']:.3f};{st['signFlips']};{st['overshoot']}"
        )
    print(f"elapsedSec={time.perf_counter() - t0:.3f}")


if __name__ == "__main__":
    main()
