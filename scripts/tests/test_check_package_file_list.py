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
    def validate(self, *paths: str, line_ending: bytes = b"\n") -> int:
        payload = line_ending.join(path.encode("utf-8") for path in paths) + line_ending
        return CHECKER.validate_package_file_list(io.BytesIO(payload))

    def test_accepts_normal_package_paths_templates_and_source_canaries(self) -> None:
        count = self.validate(
            "Cargo.toml",
            "Cargo.toml.orig",
            "Cargo.lock",
            ".cargo_vcs_info.json",
            "src/lib.rs",
            "README.md",
            "fixtures/.env.example",
            "fixtures/.env.sample",
            "tests/canary_contract.rs",
            "examples/openai_flagship.rs",
            line_ending=b"\r\n",
        )

        self.assertEqual(count, 10)

    def test_rejects_absolute_drive_user_and_parent_paths(self) -> None:
        rejected = (
            "/Users/example/private.json",
            r"C:\Users\example\private.json",
            r"\\server\share\private.json",
            "~/private.json",
            "fixtures/../private.json",
        )

        for path in rejected:
            with self.subTest(path=path):
                with self.assertRaises(CHECKER.PackageFileListError):
                    self.validate(path)

    def test_rejects_control_characters_and_invalid_utf8(self) -> None:
        for path in ("src/bad\tname.rs", "src/hidden\u202ename.rs"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    CHECKER.PackageFileListError, "control character"
                ):
                    self.validate(path)

        with self.assertRaisesRegex(CHECKER.PackageFileListError, "valid UTF-8"):
            CHECKER.validate_package_file_list(io.BytesIO(b"src/\xff.rs\n"))

    def test_rejects_build_reference_vcs_and_local_tool_directories(self) -> None:
        rejected = (
            "target/debug/library.rlib",
            "repo-ref/ai/package.json",
            ".git/config",
            "nested/.hg/store",
            ".codex/session.json",
            ".codex-helper/config.toml",
            ".idea/workspace.xml",
            ".vscode/settings.json",
            ".fleet/settings.json",
        )

        for path in rejected:
            with self.subTest(path=path):
                with self.assertRaises(CHECKER.PackageFileListError):
                    self.validate(path)

    def test_rejects_credentials_and_private_configuration(self) -> None:
        rejected = (
            ".env",
            ".env.local",
            "config.local.toml",
            "settings.local.json",
            "credentials.json",
            "service-account-production.json",
            "secrets.yaml",
            "private-key.pem",
            "id_ed25519",
            ".netrc",
            "release-token.p12",
        )

        for path in rejected:
            with self.subTest(path=path):
                with self.assertRaises(CHECKER.PackageFileListError):
                    self.validate(path)

    def test_rejects_editor_temporary_and_live_canary_artifacts(self) -> None:
        rejected = (
            ".DS_Store",
            "notes.txt~",
            "src/lib.rs.swp",
            ".#README.md",
            "#README.md#",
            "siumai.code-workspace",
            ".canary-output.json",
            "canary.json",
            "canary-result.json",
            "live-canary-response.json",
            "sub2api_canary_payload.json",
        )

        for path in rejected:
            with self.subTest(path=path):
                with self.assertRaises(CHECKER.PackageFileListError):
                    self.validate(path)

    def test_rejects_empty_oversized_and_overlong_lists(self) -> None:
        with self.assertRaisesRegex(CHECKER.PackageFileListError, "is empty"):
            CHECKER.validate_package_file_list(io.BytesIO())

        with patch.object(CHECKER, "MAX_PATH_BYTES", 8):
            with self.assertRaisesRegex(CHECKER.PackageFileListError, "exceeds 8 bytes"):
                CHECKER.validate_package_file_list(io.BytesIO(b"123456789\n"))

        with patch.object(CHECKER, "MAX_ENTRIES", 2):
            with self.assertRaisesRegex(CHECKER.PackageFileListError, "exceeds 2 entries"):
                CHECKER.validate_package_file_list(io.BytesIO(b"one\ntwo\nthree\n"))


if __name__ == "__main__":
    unittest.main()
