from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "audit-model-catalogs.py"
SPEC = importlib.util.spec_from_file_location("audit_model_catalogs_wrapper", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ModelCatalogAuditWrapperTests(unittest.TestCase):
    def test_standard_gate_and_custom_arguments_are_forwarded(self) -> None:
        repo_root = Path("/repo")

        command = AUDIT.build_command(repo_root, ["--provider", "openai"])

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(
            command[1],
            str(
                repo_root
                / ".agents"
                / "skills"
                / "siumai-ai-sdk-maintenance"
                / "scripts"
                / "audit_model_catalogs.py"
            ),
        )
        self.assertEqual(command[-2:], ["--provider", "openai"])
        self.assertIn("deepinfra", command)


if __name__ == "__main__":
    unittest.main()
