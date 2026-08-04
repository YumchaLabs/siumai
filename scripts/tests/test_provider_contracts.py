from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "test-provider-contracts.py"
SPEC = importlib.util.spec_from_file_location("provider_contracts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONTRACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACTS)


class ProviderContractProfileTests(unittest.TestCase):
    def test_recent_provider_packages_have_real_contract_profiles(self) -> None:
        self.assertEqual(
            CONTRACTS.PROFILES["gateway"],
            ("siumai-provider-gateway", "gateway"),
        )
        self.assertEqual(
            CONTRACTS.PROFILES["deepgram"],
            ("siumai-provider-deepgram", "deepgram"),
        )
        self.assertEqual(
            CONTRACTS.PROFILES["elevenlabs"],
            ("siumai-provider-elevenlabs", "elevenlabs"),
        )


if __name__ == "__main__":
    unittest.main()
