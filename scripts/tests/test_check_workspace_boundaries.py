from __future__ import annotations

import contextlib
import io
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "check_workspace_boundaries.py"
SPEC = importlib.util.spec_from_file_location("check_workspace_boundaries", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BOUNDARIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUNDARIES)


def package(name: str, dependencies: list[str]) -> dict:
    return {
        "id": f"path+file:///repo/{name}#{name}",
        "name": name,
        "dependencies": [{"name": dependency} for dependency in dependencies],
    }


def metadata(packages: list[dict]) -> dict:
    return {
        "packages": packages,
        "workspace_members": [item["id"] for item in packages],
    }


class WorkspaceBoundaryTests(unittest.TestCase):
    def test_legacy_arguments_are_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                BOUNDARIES.parse_args(["--target"])

        self.assertEqual(error.exception.code, 2)

    def test_current_architecture_is_accepted(self) -> None:
        graph = metadata(
            [
                package("siumai-core", []),
                package("siumai-transport", ["siumai-core"]),
                package("siumai-registry", ["siumai-core"]),
                package("siumai-runtime", ["siumai-core"]),
                package("siumai-protocol-openai", ["siumai-core"]),
                package(
                    "siumai-openai-compatible",
                    ["siumai-core", "siumai-protocol-openai", "siumai-transport"],
                ),
                package(
                    "siumai-provider-openai",
                    ["siumai-core", "siumai-protocol-openai", "siumai-transport"],
                ),
                package("siumai-server", ["siumai-core", "siumai-runtime"]),
                package("siumai-mcp", ["siumai-core", "siumai-runtime"]),
                package(
                    "siumai",
                    [
                        "siumai-core",
                        "siumai-provider-openai",
                        "siumai-registry",
                        "siumai-runtime",
                    ],
                ),
            ]
        )

        self.assertEqual(BOUNDARIES.validate(graph), [])

    def test_versions_and_msrv_are_not_architecture_inputs(self) -> None:
        core = package("siumai-core", [])
        core.update({"version": "0.11.0-beta.10", "rust_version": "1.95"})
        runtime = package("siumai-runtime", ["siumai-core"])
        runtime.update({"version": "99.0.0", "rust_version": "1.0"})

        self.assertEqual(BOUNDARIES.validate(metadata([core, runtime])), [])

    def test_core_cannot_depend_on_another_workspace_package(self) -> None:
        graph = metadata(
            [
                package("siumai-core", ["siumai-transport"]),
                package("siumai-transport", ["siumai-core"]),
            ]
        )

        errors = BOUNDARIES.validate(graph)

        self.assertEqual(len(errors), 1)
        self.assertIn("siumai-core", errors[0])
        self.assertIn("siumai-transport", errors[0])

    def test_foundation_packages_reject_host_or_provider_dependencies(self) -> None:
        for name in ("siumai-transport", "siumai-registry", "siumai-runtime"):
            with self.subTest(package=name):
                graph = metadata(
                    [
                        package("siumai-core", []),
                        package(name, ["siumai-core", "siumai-provider-openai"]),
                        package("siumai-provider-openai", ["siumai-core"]),
                    ]
                )

                errors = BOUNDARIES.validate(graph)

                self.assertEqual(len(errors), 1)
                self.assertIn(name, errors[0])
                self.assertIn("siumai-provider-openai", errors[0])

    def test_branded_provider_cannot_depend_on_another_branded_provider(self) -> None:
        graph = metadata(
            [
                package("siumai-core", []),
                package("siumai-provider-openai", ["siumai-core"]),
                package(
                    "siumai-provider-anthropic",
                    ["siumai-core", "siumai-provider-openai"],
                ),
            ]
        )

        errors = BOUNDARIES.validate(graph)

        self.assertEqual(len(errors), 1)
        self.assertIn("provider-to-provider", errors[0])
        self.assertIn("siumai-provider-openai", errors[0])

    def test_branded_provider_cannot_depend_on_host_layers(self) -> None:
        graph = metadata(
            [
                package("siumai-core", []),
                package("siumai-registry", ["siumai-core"]),
                package("siumai-runtime", ["siumai-core"]),
                package(
                    "siumai-provider-openai",
                    ["siumai-core", "siumai-registry", "siumai-runtime"],
                ),
            ]
        )

        errors = BOUNDARIES.validate(graph)

        self.assertEqual(len(errors), 1)
        self.assertIn("host-layer", errors[0])
        self.assertIn("siumai-registry", errors[0])
        self.assertIn("siumai-runtime", errors[0])

    def test_wire_and_compatibility_layers_cannot_depend_on_host_layers(self) -> None:
        for name, dependency in (
            ("siumai-protocol-openai", "siumai-runtime"),
            ("siumai-openai-compatible", "siumai-registry"),
        ):
            with self.subTest(package=name, dependency=dependency):
                graph = metadata(
                    [
                        package("siumai-core", []),
                        package(dependency, ["siumai-core"]),
                        package(name, ["siumai-core", dependency]),
                    ]
                )

                errors = BOUNDARIES.validate(graph)

                self.assertEqual(len(errors), 1)
                self.assertIn("wire or compatibility", errors[0])
                self.assertIn(dependency, errors[0])

    def test_workspace_packages_cannot_depend_on_the_facade(self) -> None:
        graph = metadata(
            [
                package("siumai-core", []),
                package("siumai", ["siumai-core"]),
                package("siumai-protocol-openai", ["siumai-core", "siumai"]),
            ]
        )

        errors = BOUNDARIES.validate(graph)

        self.assertEqual(len(errors), 1)
        self.assertIn("facade back-edge", errors[0])
        self.assertIn("siumai-protocol-openai", errors[0])


if __name__ == "__main__":
    unittest.main()
