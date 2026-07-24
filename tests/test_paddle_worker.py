import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_worker():
    path = (
        Path(__file__).resolve().parents[1]
        / "parser-workers"
        / "paddle_worker.py"
    )
    spec = importlib.util.spec_from_file_location("paddle_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load paddle worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PADDLE_WORKER = _load_worker()


class PaddleWorkerTests(unittest.TestCase):
    def test_raw_ocr_line_boundaries_replace_concatenated_markdown(self):
        result = SimpleNamespace(
            markdown={
                "markdown_texts": (
                    "부산대학교 졸업과제 챗봇"
                    "한국은행 금융경제 연구"
                    "한국인터넷진흥원 정보보호"
                )
            },
            json={
                "res": {
                    "overall_ocr_res": {
                        "rec_texts": [
                            "부산대학교 졸업과제 챗봇",
                            "한국은행 금융경제 연구",
                            "한국인터넷진흥원 정보보호",
                        ]
                    }
                }
            },
        )

        self.assertEqual(
            PADDLE_WORKER._result_text(result),
            (
                "부산대학교 졸업과제 챗봇\n"
                "한국은행 금융경제 연구\n"
                "한국인터넷진흥원 정보보호"
            ),
        )

    def test_structured_markdown_is_kept_when_it_differs_from_raw_ocr(self):
        result = SimpleNamespace(
            markdown={"markdown_texts": "# 제목\n\n본문"},
            json={
                "res": {
                    "overall_ocr_res": {
                        "rec_texts": ["제목", "본문"],
                    }
                }
            },
        )

        self.assertEqual(PADDLE_WORKER._result_text(result), "# 제목\n\n본문")


if __name__ == "__main__":
    unittest.main()
