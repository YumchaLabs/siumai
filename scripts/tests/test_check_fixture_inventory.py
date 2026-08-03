from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "check_fixture_inventory.py"
SPEC = importlib.util.spec_from_file_location("check_fixture_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INVENTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INVENTORY)


def fixture_set(pattern: str) -> dict:
    return {
        "path": pattern,
        "owner": "protocol-test",
        "protocol": "example",
        "behaviors": ["stream"],
        "source_kind": "test",
        "source_date": "2026-08-04",
        "disposition": "retain-and-reverify",
    }


class FixtureInventoryTests(unittest.TestCase):
    def test_every_file_inherits_exactly_one_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "openai" / "stream").mkdir(parents=True)
            (root / "openai" / "stream" / "one.sse").write_text("data", encoding="utf-8")

            errors = INVENTORY.validate(
                root,
                {"sets": [fixture_set("openai/**")]},
            )

            self.assertEqual(errors, [])

    def test_unclassified_fixture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unknown").mkdir()
            (root / "unknown" / "one.json").write_text("{}", encoding="utf-8")

            errors = INVENTORY.validate(
                root,
                {"sets": [fixture_set("openai/**")]},
            )

            self.assertTrue(any("unclassified fixture" in error for error in errors))
            self.assertTrue(any("matches no files" in error for error in errors))

    def test_overlapping_fixture_sets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "openai" / "stream").mkdir(parents=True)
            (root / "openai" / "stream" / "one.sse").write_text("data", encoding="utf-8")

            errors = INVENTORY.validate(
                root,
                {
                    "sets": [
                        fixture_set("openai/**"),
                        fixture_set("openai/stream/**"),
                    ]
                },
            )

            self.assertTrue(any("matches multiple sets" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
