from __future__ import annotations

import importlib.util
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parent.parent / "release_plz_release_with_retry.py"
SPEC = importlib.util.spec_from_file_location("release_plz_retry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETRY)


class ReleasePlzRetryTests(unittest.TestCase):
    def test_detects_supported_crates_io_rate_limit_messages(self) -> None:
        self.assertTrue(RETRY.is_crates_io_rate_limit("status 429 Too Many Requests"))
        self.assertTrue(RETRY.is_crates_io_rate_limit("published too many new crates"))
        self.assertFalse(RETRY.is_crates_io_rate_limit("status 500 Internal Server Error"))

    def test_retry_after_uses_cross_platform_rfc_date_parsing(self) -> None:
        output = (
            "Please try again after Wed, 14 Jan 2026 14:43:25 GMT before publishing."
        )

        retry_after = RETRY.extract_retry_after(output)

        self.assertEqual(
            retry_after,
            datetime(2026, 1, 14, 14, 43, 25, tzinfo=timezone.utc),
        )

    def test_retry_delay_includes_safety_margin_and_minimum(self) -> None:
        now = datetime(2026, 1, 14, 14, 42, 25, tzinfo=timezone.utc)
        output = "Please try again after Wed, 14 Jan 2026 14:43:25 GMT."

        self.assertEqual(RETRY.retry_delay_seconds(output, now, 30), 70)
        self.assertEqual(RETRY.retry_delay_seconds("no retry date", now, 60), 60)

    def test_positive_environment_integer_rejects_invalid_values(self) -> None:
        with patch.dict(os.environ, {"RETRY_TEST_VALUE": "0"}):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                RETRY.positive_env_int("RETRY_TEST_VALUE", 1)


if __name__ == "__main__":
    unittest.main()
