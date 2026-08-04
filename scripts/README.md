# Repository Scripts

All maintained local-development entry points are Python 3 scripts so the same command works on
Windows, macOS, and Linux. Rust test commands run serially with `-j 1`; when available,
`cargo-nextest` is preferred automatically.

The examples below use `python3`; on Windows, use `py -3` if Python is installed through the
standard Python launcher.

## Local test suites

`test-workspace.py` is the main local entry point:

```text
# Fast core, transport, registry, and facade loop
python3 scripts/test-workspace.py fast

# Provider-aware smoke suite
python3 scripts/test-workspace.py smoke
python3 scripts/test-workspace.py smoke --profile openai-compatible
python3 scripts/test-workspace.py smoke --profile all-providers

# Full workspace/all-features suite
python3 scripts/test-workspace.py full
```

Useful options:

- `--runner auto|nextest|cargo-test` selects the Rust test runner.
- `--dry-run` prints the exact commands without executing them.
- `--profile` applies only to the smoke suite.

The old platform-specific fast/smoke/full wrappers were merged into this script. The old M1,
interactive integration, environment smoke, and Ollama wrappers were removed because they invoked
legacy facade tests that are not part of the current Cargo test surface. Credentialed provider
smoke tests should be added back as provider-owned test targets instead of reviving those facade
entry points.

## Provider contract suites

`test-provider-contracts.py` runs no-network tests for one provider package or the complete provider
matrix:

```text
python3 scripts/test-provider-contracts.py openai-native
python3 scripts/test-provider-contracts.py google-vertex
python3 scripts/test-provider-contracts.py
```

`test-cross-feature-contracts.py` checks the facade's supported feature shapes:

```text
python3 scripts/test-cross-feature-contracts.py no-default
python3 scripts/test-cross-feature-contracts.py default
python3 scripts/test-cross-feature-contracts.py one-provider
python3 scripts/test-cross-feature-contracts.py registry-openai
python3 scripts/test-cross-feature-contracts.py multi-provider
python3 scripts/test-cross-feature-contracts.py all-providers
python3 scripts/test-cross-feature-contracts.py openai-realtime
python3 scripts/test-cross-feature-contracts.py
```

The complete matrix covers no-default, default, one provider without Registry, Registry plus one
provider, a representative multi-provider build, the OpenAI-compatible stack, and the experimental
OpenAI Realtime feature.
The all-provider profile also compiles every facade provider feature together with Registry.

## Architecture and fixture checks

```text
python3 -B scripts/check_workspace_boundaries.py
python3 -B scripts/check_workspace_boundaries.py --target
python3 -B scripts/check_fixture_inventory.py
python3 -B -m unittest discover -s scripts/tests -p "test_*.py"
```

These scripts validate declared package boundaries and fixture ownership. They intentionally do not
parse Rust source or attempt to infer compiler semantics.

## Model catalog audit

`audit-model-catalogs.py` runs the repository-standard AI SDK catalog drift gate:

```text
python3 scripts/audit-model-catalogs.py
```

It requires a local Vercel AI SDK checkout discoverable as `repo-ref/ai`, `AI_SDK_REPO`, or
`VERCEL_AI_REPO`. Additional arguments are forwarded to the underlying audit tool. The standard
wrapper always enables green/skipped output and defers DeepInfra pending a separate catalog policy
decision.

## Release retry

`release_plz_release_with_retry.py` is the release-only retry wrapper used by GitHub Actions. It
streams `release-plz` output, recognizes crates.io rate limiting, parses the retry timestamp with
the Python standard library, and applies bounded retries without depending on Bash or GNU `date`.
