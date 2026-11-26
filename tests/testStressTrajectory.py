"""Unit tests for Train/Test trajectory split and PID overshoot detection."""

import unittest

from llmCalibratorSim.calibrators import ClampedLlmCalibrator, PerGroupPidCalibrator
from llmCalibratorSim.stressTrajectory import (
    detectOvershoot,
    runPidOvershootSweep,
    runTrajectoryEpisode,
)


class TestStressTrajectory(unittest.TestCase):
    def testPidSweepReachesOvershootAtHighGain(self) -> None:
        sweep = runPidOvershootSweep(seed=42)
        self.assertEqual(len(sweep), 4)
        lowGainStats = sweep[0][2]
        highGainStats = sweep[-1][2]
        self.assertGreaterEqual(highGainStats["signFlips"], lowGainStats["signFlips"])
        self.assertTrue(highGainStats["overshoot"], f"Expected overshoot at high gain: {highGainStats}")

    def testTrajectoryHasIndependentTrainAndHeldOutTest(self) -> None:
        trajPid = runTrajectoryEpisode("per_group_pid", PerGroupPidCalibrator(), seed=42)
        trajOracle = runTrajectoryEpisode("oracle_clamped", ClampedLlmCalibrator(), seed=42)
        self.assertEqual(len(trajPid), 14)
        self.assertEqual(len(trajOracle), 14)
        oracleStats = detectOvershoot(trajOracle)
        pidStats = detectOvershoot(trajPid)
        self.assertGreater(oracleStats["endTestAcc"], pidStats["endTestAcc"])


if __name__ == "__main__":
    unittest.main()
