from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "test-cross-feature-contracts.py"
SPEC = importlib.util.spec_from_file_location("test_cross_feature_contracts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONTRACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACTS)


class CrossFeatureContractTests(unittest.TestCase):
    def test_no_default_profile_does_not_pass_an_empty_features_argument(self) -> None:
        command = CONTRACTS.build_command("no-default", use_nextest=True)

        self.assertIn("--no-default-features", command)
        self.assertNotIn("--features", command)

    def test_default_profile_keeps_default_features_enabled(self) -> None:
        command = CONTRACTS.build_command("default", use_nextest=False)

        self.assertNotIn("--no-default-features", command)
        self.assertNotIn("--features", command)

    def test_one_provider_profile_does_not_enable_registry(self) -> None:
        command = CONTRACTS.build_command("one-provider", use_nextest=True)
        features = command[command.index("--features") + 1].split(",")

        self.assertEqual(features, ["openai"])
        self.assertNotIn("registry", features)

    def test_runtime_only_profile_does_not_enable_registry_or_providers(self) -> None:
        command = CONTRACTS.build_command("runtime-only", use_nextest=True)
        features = command[command.index("--features") + 1].split(",")

        self.assertEqual(features, ["runtime"])

    def test_runtime_registry_profile_keeps_composition_explicit(self) -> None:
        command = CONTRACTS.build_command("runtime-registry", use_nextest=True)
        features = command[command.index("--features") + 1].split(",")

        self.assertEqual(features, ["runtime", "registry"])

    def test_multi_provider_profile_exercises_registry_and_provider_features(self) -> None:
        command = CONTRACTS.build_command("multi-provider", use_nextest=True)
        features = command[command.index("--features") + 1]

        self.assertIn("registry", features.split(","))
        self.assertIn("runtime", features.split(","))
        self.assertIn("openai", features.split(","))
        self.assertIn("google", features.split(","))
        self.assertIn("deepgram", features.split(","))
        self.assertIn("elevenlabs", features.split(","))

    def test_all_provider_profile_compiles_registry_integration(self) -> None:
        command = CONTRACTS.build_command("all-providers", use_nextest=True)
        features = command[command.index("--features") + 1].split(",")

        self.assertEqual(features, ["runtime", "registry", "all-providers"])


if __name__ == "__main__":
    unittest.main()
