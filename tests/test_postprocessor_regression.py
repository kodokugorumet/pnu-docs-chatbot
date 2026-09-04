from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_postprocessor_regression",
    ROOT / "scripts" / "evaluate_postprocessor_regression.py",
)
assert SPEC is not None and SPEC.loader is not None
regression_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regression_module)


class PostprocessorRegressionTests(unittest.TestCase):
    def test_current_splitter_retains_dotted_date_better_than_legacy(self) -> None:
        cases = [
            {
                "id": "date",
                "category": "registration",
                "reference": (
                    "본등록은 2026. 8. 24.(월) ~ 8. 27.(목)입니다. "
                    "수납은행은 농협과 부산은행입니다."
                ),
            }
        ]

        result = regression_module.evaluate(cases)

        self.assertEqual(result["summary"]["improved"], 1)
        self.assertEqual(result["summary"]["regressed"], 0)

    def test_current_splitter_does_not_regress_plain_sentence(self) -> None:
        cases = [
            {
                "id": "plain",
                "category": "academic",
                "reference": "온라인 신청은 학생지원시스템에서 진행할 수 있습니다.",
            }
        ]

        result = regression_module.evaluate(cases)

        self.assertEqual(result["summary"]["regressed"], 0)


if __name__ == "__main__":
    unittest.main()
