from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "check_workspace_boundaries.py"
SPEC = importlib.util.spec_from_file_location("check_workspace_boundaries", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BOUNDARIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUNDARIES)


def package(
    name: str,
    dependencies: list[str],
    rust_version: str = "1.88",
    version: str = "0.11.0-beta.9",
) -> dict:
    return {
        "id": f"path+file:///repo/{name}#{name}",
        "name": name,
        "rust_version": rust_version,
        "version": version,
        "dependencies": [{"name": dependency} for dependency in dependencies],
    }


def metadata(packages: list[dict]) -> dict:
    return {
        "packages": packages,
        "workspace_members": [item["id"] for item in packages],
    }


POLICY = {
    "msrv": "1.88",
    "workspace_version": "0.11.0-beta.9",
    "package_rules": {
        "siumai-core": {
            "current_allowed_workspace_dependencies": [],
            "target_allowed_workspace_dependencies": [],
        },
        "siumai-registry": {
            "current_allowed_workspace_dependencies": ["siumai-core"],
            "target_allowed_workspace_dependencies": ["siumai-core"],
        },
        "siumai-runtime": {
            "current_allowed_workspace_dependencies": ["siumai-core"],
            "target_allowed_workspace_dependencies": ["siumai-core"],
        },
    },
    "provider_dependency_transitions": [],
}


class WorkspaceBoundaryTests(unittest.TestCase):
    def test_target_graph_accepts_core_only_registry_and_runtime(self) -> None:
        graph = metadata(
            [
                package("siumai-core", []),
                package("siumai-registry", ["siumai-core"]),
                package("siumai-runtime", ["siumai-core"]),
                package("siumai-provider-openai", ["siumai-core"]),
            ]
        )

        self.assertEqual(BOUNDARIES.validate(graph, POLICY, target=True), [])

    def test_target_graph_rejects_runtime_registry_dependency(self) -> None:
        graph = metadata(
            [
                package("siumai-core", []),
                package("siumai-registry", ["siumai-core"]),
                package("siumai-runtime", ["siumai-core", "siumai-registry"]),
            ]
        )

        errors = BOUNDARIES.validate(graph, POLICY, target=True)

        self.assertEqual(len(errors), 1)
        self.assertIn("siumai-registry", errors[0])

    def test_target_graph_rejects_registry_provider_dependency(self) -> None:
        graph = metadata(
            [
                package("siumai-core", []),
                package(
                    "siumai-registry",
                    ["siumai-core", "siumai-provider-openai"],
                ),
                package("siumai-runtime", ["siumai-core"]),
                package("siumai-provider-openai", ["siumai-core"]),
            ]
        )

        errors = BOUNDARIES.validate(graph, POLICY, target=True)

        self.assertEqual(len(errors), 1)
        self.assertIn("siumai-provider-openai", errors[0])

    def test_migration_policy_rejects_unnamed_registry_dependency(self) -> None:
        graph = metadata(
            [
                package("siumai-core", []),
                package("siumai-registry", ["siumai-core"]),
                package("siumai-provider-openai", ["siumai-core"]),
                package("siumai-runtime", ["siumai-core"]),
            ]
        )

        self.assertEqual(BOUNDARIES.validate(graph, POLICY, target=False), [])

        graph["packages"][1]["dependencies"].append({"name": "siumai-provider-openai"})

        errors = BOUNDARIES.validate(graph, POLICY, target=False)
        self.assertEqual(len(errors), 1)
        self.assertIn("siumai-provider-openai", errors[0])

    def test_provider_to_provider_dependency_is_rejected(self) -> None:
        graph = metadata(
            [
                package("siumai-core", []),
                package("siumai-registry", ["siumai-core"]),
                package("siumai-runtime", ["siumai-core"]),
                package("siumai-provider-openai", ["siumai-core"]),
                package(
                    "siumai-provider-anthropic",
                    ["siumai-core", "siumai-provider-openai"],
                ),
            ]
        )

        errors = BOUNDARIES.validate(graph, POLICY, target=False)

        self.assertEqual(len(errors), 1)
        self.assertIn("provider-to-provider", errors[0])

    def test_declared_msrv_is_required_for_every_package(self) -> None:
        graph = metadata(
            [
                package("siumai-core", [], rust_version="1.89"),
                package("siumai-registry", ["siumai-core"]),
                package("siumai-runtime", ["siumai-core"]),
            ]
        )

        errors = BOUNDARIES.validate(graph, POLICY, target=True)

        self.assertEqual(len(errors), 1)
        self.assertIn("rust_version='1.89'", errors[0])

    def test_workspace_version_is_required_for_every_package(self) -> None:
        graph = metadata(
            [
                package("siumai-core", []),
                package(
                    "siumai-registry",
                    ["siumai-core"],
                    version="0.11.0-beta.10",
                ),
                package("siumai-runtime", ["siumai-core"]),
            ]
        )

        errors = BOUNDARIES.validate(graph, POLICY, target=True)

        self.assertEqual(len(errors), 1)
        self.assertIn("siumai-registry: version='0.11.0-beta.10'", errors[0])
        self.assertIn("expected '0.11.0-beta.9'", errors[0])


if __name__ == "__main__":
    unittest.main()
