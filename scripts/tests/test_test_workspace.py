from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parent.parent / "test-workspace.py"
SPEC = importlib.util.spec_from_file_location("test_workspace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
WORKSPACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKSPACE)


class WorkspaceTestRunnerTests(unittest.TestCase):
    def test_fast_suite_uses_serial_cargo_and_current_packages(self) -> None:
        args = SimpleNamespace(suite="fast")

        commands = WORKSPACE.commands_for(args, "nextest")
        cargo = commands[-1]

        self.assertEqual(cargo[:3], ["cargo", "nextest", "run"])
        self.assertIn("-j", cargo)
        self.assertEqual(cargo[cargo.index("-j") + 1], "1")
        self.assertIn("siumai-core", cargo)
        self.assertIn("siumai-runtime", cargo)
        self.assertIn("siumai-transport", cargo)
        self.assertIn("siumai-registry", cargo)
        self.assertIn("siumai", cargo)

        cargo_test = WORKSPACE.commands_for(args, "cargo-test")[-1]
        self.assertNotIn("--test-threads=1", cargo_test)

    def test_flagship_suite_uses_exact_protocol_provider_packages_with_nextest(
        self,
    ) -> None:
        args = SimpleNamespace(suite="flagship")

        commands = WORKSPACE.commands_for(args, "nextest")

        self.assertEqual(
            commands,
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "-j",
                    "1",
                    "--no-fail-fast",
                    "-p",
                    "siumai-protocol-openai",
                    "-p",
                    "siumai-provider-openai",
                    "-p",
                    "siumai-openai-compatible",
                    "-p",
                    "siumai-protocol-anthropic",
                    "-p",
                    "siumai-anthropic-compatible",
                    "-p",
                    "siumai-provider-anthropic",
                ]
            ],
        )

    def test_flagship_suite_keeps_cargo_test_single_threaded(self) -> None:
        args = SimpleNamespace(suite="flagship")

        commands = WORKSPACE.commands_for(args, "cargo-test")
        cargo = commands[0]

        self.assertEqual(cargo[:4], ["cargo", "test", "-j", "1"])
        self.assertIn("--no-fail-fast", cargo)
        self.assertEqual(cargo[-2:], ["--", "--test-threads=1"])
        self.assertEqual(
            [cargo[index + 1] for index, value in enumerate(cargo) if value == "-p"],
            [
                "siumai-protocol-openai",
                "siumai-provider-openai",
                "siumai-openai-compatible",
                "siumai-protocol-anthropic",
                "siumai-anthropic-compatible",
                "siumai-provider-anthropic",
            ],
        )

    def test_full_suite_runs_only_serial_workspace_tests(self) -> None:
        args = SimpleNamespace(suite="full")

        commands = WORKSPACE.commands_for(args, "nextest")

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[-1][commands[-1].index("-j") + 1], "1")
        self.assertEqual(
            commands[-1][commands[-1].index("--profile") + 1],
            "ci",
        )
        self.assertIn("--workspace", commands[-1])
        self.assertIn("--all-features", commands[-1])
        self.assertFalse(
            any("scripts/test-cross-feature-contracts.py" in command for command in commands)
        )


if __name__ == "__main__":
    unittest.main()
