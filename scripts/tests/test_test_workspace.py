from __future__ import annotations

import importlib.util
import sys
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
        args = SimpleNamespace(suite="fast", profile="openai")

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

    def test_smoke_suite_composes_existing_python_contract_runners(self) -> None:
        args = SimpleNamespace(suite="smoke", profile="openai-compatible")

        commands = WORKSPACE.commands_for(args, "cargo-test")
        rendered = [" ".join(command) for command in commands]

        self.assertTrue(
            any("test-provider-contracts.py openai-compat" in item for item in rendered)
        )
        self.assertTrue(any("test-provider-contracts.py groq" in item for item in rendered))
        self.assertTrue(
            any("test-cross-feature-contracts.py compatible-stack" in item for item in rendered)
        )

    def test_full_suite_runs_python_checks_before_serial_workspace_tests(self) -> None:
        args = SimpleNamespace(suite="full", profile="openai")

        commands = WORKSPACE.commands_for(args, "nextest")

        self.assertEqual(commands[0][0], sys.executable)
        self.assertIn("unittest", commands[0])
        self.assertEqual(commands[-1][commands[-1].index("-j") + 1], "1")
        self.assertEqual(
            commands[-1][commands[-1].index("--profile") + 1],
            "ci",
        )
        self.assertIn("--workspace", commands[-1])
        self.assertIn("--all-features", commands[-1])

    def test_all_provider_smoke_uses_the_all_provider_facade_profile(self) -> None:
        args = SimpleNamespace(suite="smoke", profile="all-providers")

        rendered = [" ".join(command) for command in WORKSPACE.commands_for(args, "nextest")]

        self.assertTrue(
            any(
                "test-cross-feature-contracts.py all-providers" in item
                for item in rendered
            )
        )


if __name__ == "__main__":
    unittest.main()
