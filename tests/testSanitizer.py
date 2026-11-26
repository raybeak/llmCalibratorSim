"""Unit tests for Sovereignty-safe outbound telemetry sanitizer (Unit 2 / MOE 1)."""

import inspect
import math
import unittest

from llmCalibratorSim.sanitizer import (
    SovereigntyViolationError,
    sanitizeTelemetryArray,
)


def _validRecord() -> dict:
    return {
        "decision_id": 1,
        "ext_enum": "doc_hwpx",
        "pred_cat": "cat_0",
        "actual_cat": "cat_0",
        "outcome": "TP",
        "raw_goodness_G": 2.35,
        "confidence_p": 0.91,
        "timestamp_s": 1764140000,
    }


class TestSanitizer(unittest.TestCase):
    def testValidTelemetryPasses(self) -> None:
        cleaned = sanitizeTelemetryArray([_validRecord()])
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["ext_enum"], "doc_hwpx")

    def testNoBypassParameterExists(self) -> None:
        sig = inspect.signature(sanitizeTelemetryArray)
        for name in sig.parameters:
            self.assertNotIn("bypass", name.lower())

    def testRejectsExtraAndMissingKeys(self) -> None:
        extra = _validRecord()
        extra["filename"] = "/home/raybeak/secret.hwpx"
        with self.assertRaises(SovereigntyViolationError):
            sanitizeTelemetryArray([extra])

        missing = _validRecord()
        del missing["outcome"]
        with self.assertRaises(SovereigntyViolationError):
            sanitizeTelemetryArray([missing])

    def testRejectsRawPathsEmbeddingsAndInvalidEnums(self) -> None:
        badExt = _validRecord()
        badExt["ext_enum"] = "/tmp/report.hwpx"
        with self.assertRaises(SovereigntyViolationError):
            sanitizeTelemetryArray([badExt])

        badVec = _validRecord()
        badVec["raw_goodness_G"] = [0.1, 0.2, 0.3]
        with self.assertRaises(SovereigntyViolationError):
            sanitizeTelemetryArray([badVec])

        badCat = _validRecord()
        badCat["pred_cat"] = "finance_private"
        with self.assertRaises(SovereigntyViolationError):
            sanitizeTelemetryArray([badCat])

    def testRejectsBoolAsIntAndBadFloats(self) -> None:
        boolId = _validRecord()
        boolId["decision_id"] = True
        with self.assertRaises(SovereigntyViolationError):
            sanitizeTelemetryArray([boolId])

        nanG = _validRecord()
        nanG["raw_goodness_G"] = math.nan
        with self.assertRaises(SovereigntyViolationError):
            sanitizeTelemetryArray([nanG])

        badConf = _validRecord()
        badConf["confidence_p"] = 1.25
        with self.assertRaises(SovereigntyViolationError):
            sanitizeTelemetryArray([badConf])


if __name__ == "__main__":
    unittest.main()
