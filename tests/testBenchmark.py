"""Unit tests for Calibrator clamping/fuzzing and 5-seed benchmark MOEs (Unit 3)."""

import time
import unittest

from llmCalibratorSim.calibrators import clampAndValidateOutput
from llmCalibratorSim.runBenchmark import runFullBenchmark


class TestCalibratorAndBenchmark(unittest.TestCase):
    def testClampingAndMalformedJsonFuzzFallback(self) -> None:
        extreme = {
            "global_delta": {"d_theta": 99.0, "d_alpha": -50.0, "d_beta": 20.0, "d_T": -9.0},
            "group_overrides": {
                "doc_hwpx": {"d_theta": -10.0, "d_alpha": 10.0, "d_beta": 0.0, "d_T": 5.0}
            },
            "recommended_action": "trigger_replay",
            "diagnosis": "x" * 500,
        }
        cmd = clampAndValidateOutput(extreme)
        self.assertAlmostEqual(cmd.globalDelta["d_theta"], 0.15)
        self.assertAlmostEqual(cmd.globalDelta["d_alpha"], -0.20)
        self.assertAlmostEqual(cmd.globalDelta["d_beta"], 0.20)
        self.assertAlmostEqual(cmd.globalDelta["d_T"], -0.20)
        self.assertAlmostEqual(cmd.groupOverrides["doc_hwpx"]["d_theta"], -0.15)
        self.assertLessEqual(len(cmd.diagnosis), 200)

        for garbage in (
            "{not valid json",
            "```json\n{\"global_delta\": \"bad\"}\n```",
            '{"global_delta": {"d_theta": NaN}}',
            "",
        ):
            fallback = clampAndValidateOutput(garbage)
            self.assertEqual(fallback.recommendedAction, "noop")

    def testFullFiveSeedBenchmarkMeetsMoesAndMop1(self) -> None:
        t0 = time.perf_counter()
        summary = runFullBenchmark(seeds=(42, 43, 44, 45, 46))
        elapsed = time.perf_counter() - t0

        self.assertLessEqual(elapsed, 10.0, f"MOP 1 violated: {elapsed:.2f}s > 10s")

        # MOE 3: Scenario A (global_decay) parity within -3%p of PID
        aPid = summary[("global_decay", "per_group_pid")]["finalAcc"]
        aLlm = summary[("global_decay", "clamped_llm")]["finalAcc"]
        self.assertGreaterEqual(aLlm, aPid - 0.03, f"MOE 3 failed: llm={aLlm:.3f}, pid={aPid:.3f}")

        # MOE 4: Scenario C (structural_label_swap) LLM beats Per-Group PID by >= +10%p
        cPid = summary[("structural_label_swap", "per_group_pid")]["finalAcc"]
        cLlm = summary[("structural_label_swap", "clamped_llm")]["finalAcc"]
        self.assertGreaterEqual(
            cLlm - cPid,
            0.10,
            f"MOE 4 failed: llm={cLlm:.3f}, per_group_pid={cPid:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
