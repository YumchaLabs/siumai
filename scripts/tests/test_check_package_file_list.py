from __future__ import annotations

import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parent.parent / "check_package_file_list.py"
SPEC = importlib.util.spec_from_file_location("check_package_file_list", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class PackageFileListTests(unittest.TestCase):
    def validate(self, *paths: str) -> int:
        payload = b"\n".join(path.encode("utf-8") for path in paths) + b"\n"
        return CHECKER.validate_package_file_list(io.BytesIO(payload))

    def test_normal_package_paths_are_allowed(self) -> None:
        self.assertEqual(
            self.validate("Cargo.toml", "Cargo.toml.orig", "src/lib.rs"),
            3,
        )

    def test_private_local_and_editor_paths_are_rejected(self) -> None:
        for path in (
            ".env",
            ".env.example",
            ".env.production",
            "config.local.toml",
            "credentials.json",
            "private-key.pem",
            "release-token.p12",
            "service-account-production.json",
            ".fleet/settings.json",
            "repo-ref/provider/fixture.json",
            "siumai.code-workspace",
            "src/lib.rs.swo",
            "src/lib.rs.swp",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    CHECKER.PackageFileListError, "would be packaged"
                ):
                    self.validate(path)

    def test_live_canary_artifacts_are_rejected_without_blocking_source(self) -> None:
        for path in (
            ".canary-output.json",
            "canary-result.json",
            "live-canary-response.json",
            "sub2api_canary_payload.json",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    CHECKER.PackageFileListError, "would be packaged"
                ):
                    self.validate(path)

        self.assertEqual(
            self.validate(
                "tests/canary_contract.rs",
                "src/canary_request.rs",
                "tests/live_canary.rs",
            ),
            3,
        )

    def test_input_is_bounded_and_strict_utf8(self) -> None:
        with self.assertRaisesRegex(CHECKER.PackageFileListError, "returned no files"):
            CHECKER.validate_package_file_list(io.BytesIO())

        with patch.object(CHECKER, "MAX_PATH_BYTES", 4):
            with self.assertRaisesRegex(CHECKER.PackageFileListError, "exceeds 4 bytes"):
                CHECKER.validate_package_file_list(io.BytesIO(b"12345\n"))

        with self.assertRaisesRegex(CHECKER.PackageFileListError, "valid UTF-8"):
            CHECKER.validate_package_file_list(io.BytesIO(b"src/\xff.rs\n"))

    def test_cargo_command_keeps_dirty_inspection_explicit(self) -> None:
        self.assertEqual(
            CHECKER.cargo_package_list_command(allow_dirty=False),
            ["cargo", "package", "--workspace", "--list", "--locked"],
        )
        self.assertEqual(
            CHECKER.cargo_package_list_command(allow_dirty=True)[-1],
            "--allow-dirty",
        )

    def test_cargo_failure_and_success_are_handled_without_parsing_partial_output(self) -> None:
        failed = type(
            "CargoProcess",
            (),
            {"stdout": io.BytesIO(b"Cargo.toml\n"), "wait": lambda self: 101},
        )()
        with patch.object(CHECKER.subprocess, "Popen", return_value=failed), patch.object(
            CHECKER, "validate_package_file_list"
        ) as validate:
            with self.assertRaisesRegex(CHECKER.PackageFileListError, "exit status 101"):
                CHECKER.check_cargo_package_file_list(allow_dirty=False)

        validate.assert_not_called()

        succeeded = type(
            "CargoProcess",
            (),
            {"stdout": io.BytesIO(b"Cargo.toml\nsrc/lib.rs\n"), "wait": lambda self: 0},
        )()
        with patch.object(CHECKER.subprocess, "Popen", return_value=succeeded):
            self.assertEqual(
                CHECKER.check_cargo_package_file_list(allow_dirty=False),
                2,
            )

    def test_cargo_output_limit_is_enforced_after_success(self) -> None:
        process = type(
            "CargoProcess",
            (),
            {"stdout": io.BytesIO(b"Cargo.toml\n"), "wait": lambda self: 0},
        )()
        with patch.object(CHECKER, "MAX_CARGO_STDOUT_BYTES", 4), patch.object(
            CHECKER.subprocess, "Popen", return_value=process
        ):
            with self.assertRaisesRegex(CHECKER.PackageFileListError, "exceeds 4 bytes"):
                CHECKER.capture_cargo_package_file_list(allow_dirty=False)


if __name__ == "__main__":
    unittest.main()
