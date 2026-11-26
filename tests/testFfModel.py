"""Unit tests for NumPy 2-Layer FF classifier and 3-scenario drift engine (Unit 1 / MOE 2)."""

import unittest

from llmCalibratorSim.driftEnv import (
    ActivityGatedFeedback,
    SyntheticClusterDataset,
)
from llmCalibratorSim.ffModel import ForwardForwardClassifier, computeEce


class TestFfModelAndDrift(unittest.TestCase):
    def testPreDriftConvergenceMeetsMoe2(self) -> None:
        dataset = SyntheticClusterDataset(seed=42)
        model = ForwardForwardClassifier(seed=42)

        trainSamples = dataset.sampleBatch(n=640, step=0, scenario="none")
        for _ in range(4):
            for sample in trainSamples:
                model.trainStep(sample.x, sample.trueCatIdx, sample.extEnum, lr=0.08)

        evalSamples = dataset.sampleBatch(n=320, step=10, scenario="none")
        correctFlags: list[float] = []
        confidences: list[float] = []
        for sample in evalSamples:
            predIdx, accepted, _, conf, _ = model.predict(
                sample.x, sample.extEnum, goodnessScale=sample.goodnessScale
            )
            correctFlags.append(1.0 if (accepted and predIdx == sample.trueCatIdx) else 0.0)
            confidences.append(conf)

        acc = sum(correctFlags) / len(correctFlags)
        ece = computeEce(confidences, correctFlags)
        self.assertGreaterEqual(acc, 0.88, f"Pre-drift accuracy too low: {acc:.3f}")
        self.assertLessEqual(ece, 0.10, f"Pre-drift ECE too high: {ece:.3f}")

    def testDriftScenariosCauseExpectedDegradation(self) -> None:
        dataset = SyntheticClusterDataset(seed=42)
        model = ForwardForwardClassifier(seed=42)
        trainSamples = dataset.sampleBatch(n=640, step=0, scenario="none")
        for _ in range(4):
            for sample in trainSamples:
                model.trainStep(sample.x, sample.trueCatIdx, sample.extEnum, lr=0.08)

        for scenario in ("global_decay", "subgroup_drift", "structural_label_swap"):
            drifted = dataset.sampleBatch(n=240, step=200, scenario=scenario)
            errors = 0
            for sample in drifted:
                predIdx, accepted, _, _, _ = model.predict(
                    sample.x, sample.extEnum, goodnessScale=sample.goodnessScale
                )
                if not accepted or predIdx != sample.trueCatIdx:
                    errors += 1
            errorRate = errors / len(drifted)
            self.assertGreater(errorRate, 0.15, f"{scenario} did not degrade enough: {errorRate:.3f}")

    def testActivityGatedFeedbackDelaysConfirmation(self) -> None:
        gate = ActivityGatedFeedback(confirmAfterSteps=3, noiseProb=0.0, seed=7)
        ready0 = gate.observe(
            decisionId=1,
            extEnum="doc_md",
            predCat="cat_0",
            trueCat="cat_0",
            accepted=True,
            rawGoodness=2.5,
            confidence=0.9,
            timestampS=100,
        )
        self.assertEqual(len(ready0), 0)
        for i in range(2, 5):
            ready = gate.observe(
                decisionId=i,
                extEnum="doc_md",
                predCat="cat_1",
                trueCat="cat_1",
                accepted=True,
                rawGoodness=2.5,
                confidence=0.9,
                timestampS=100 + i,
            )
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0]["decision_id"], 1)
        self.assertEqual(ready[0]["outcome"], "TP")


if __name__ == "__main__":
    unittest.main()
