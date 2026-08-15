from __future__ import annotations

import importlib.util
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parent.parent / "release_plz_release_with_retry.py"
WORKFLOW = SCRIPT.parents[1] / ".github" / "workflows" / "release-plz.yml"
SPEC = importlib.util.spec_from_file_location("release_plz_retry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETRY)


class ReleasePlzRetryTests(unittest.TestCase):
    def test_dry_run_is_forwarded_without_a_shell_wrapper(self) -> None:
        process = MagicMock()
        process.stdout = ["dry run complete\n"]
        process.wait.return_value = 0

        with patch.object(RETRY.subprocess, "Popen", return_value=process) as popen:
            status, output = RETRY.run_release("github-token", dry_run=True)

        self.assertEqual(status, 0)
        self.assertEqual(output, "dry run complete\n")
        self.assertEqual(
            popen.call_args.args[0],
            [
                "release-plz",
                "release",
                "--dry-run",
            ],
        )
        self.assertEqual(popen.call_args.kwargs["env"]["GIT_TOKEN"], "github-token")

    def test_main_rejects_missing_github_token_before_spawning(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(RETRY.subprocess, "Popen") as popen,
        ):
            status = RETRY.main([])

        self.assertEqual(status, 2)
        popen.assert_not_called()

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

    def test_output_tail_keeps_only_the_bounded_suffix(self) -> None:
        tail = RETRY.BoundedOutputTail(maximum_bytes=8)

        tail.append("first-")
        tail.append("second")

        self.assertEqual(tail.text(), "t-second")

    def test_run_release_returns_a_bounded_diagnostic_tail(self) -> None:
        process = MagicMock()
        process.stdout = ["x" * RETRY.OUTPUT_TAIL_BYTES, "rate-limit-tail\n"]
        process.wait.return_value = 1

        with (
            patch.object(RETRY.subprocess, "Popen", return_value=process),
            patch("builtins.print"),
        ):
            status, output = RETRY.run_release("github-token")

        self.assertEqual(status, 1)
        self.assertLessEqual(len(output.encode("utf-8")), RETRY.OUTPUT_TAIL_BYTES)
        self.assertTrue(output.endswith("rate-limit-tail\n"))

    def test_manual_release_fails_closed_before_tool_installation(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        preflight = workflow.index("Verify release commit is current main")
        install = workflow.index("Install release tooling")
        release = workflow.index("Run release-plz (release)")
        self.assertLess(preflight, install)
        self.assertLess(preflight, release)
        self.assertIn('gh api "repos/${GITHUB_REPOSITORY}/commits/main"', workflow)
        self.assertIn('git/ref/tags/${expected_tag}', workflow)


if __name__ == "__main__":
    unittest.main()
