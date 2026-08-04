# Facade Tests

The maintained `siumai` facade test target is `facade_contract.rs`, declared explicitly in
`siumai/Cargo.toml`. It verifies direct/facade/Registry behavior and feature-gated provider exports.

Run the maintained local suites through the cross-platform Python entry points:

```text
python3 scripts/test-workspace.py fast
python3 scripts/test-workspace.py smoke
python3 scripts/test-cross-feature-contracts.py
```

Provider protocol, transport, policy, and wire-contract tests belong in their owning provider or
protocol crate and are covered by:

```text
python3 scripts/test-provider-contracts.py
```

The other files under this directory are legacy facade tests and are not auto-discovered because
the facade sets `autotests = false`. Do not add new tests to those suites or use them as evidence of
current behavior. Retained scenarios should be migrated to the owning crate or the explicit facade
contract before the legacy files are deleted.

Credentialed smoke tests are intentionally not launched from a generic facade script. A new live
test must be an explicit, opt-in target in the provider package whose endpoint, credentials, and
model policy it exercises.
