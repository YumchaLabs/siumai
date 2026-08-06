# Facade Tests

The maintained `siumai` facade test target is `facade_contract.rs`, declared explicitly in
`siumai/Cargo.toml`. It verifies direct/facade/Registry behavior and feature-gated provider exports.

Run the maintained local suites through the cross-platform Python entry points:

```text
python3 scripts/test-workspace.py fast
python3 scripts/test-workspace.py full
```

Provider protocol, transport, policy, and wire-contract tests belong in their owning provider or
protocol crate. Run that package's focused `cargo nextest` target while iterating; the full
workspace suite covers every retained package before release.

The facade sets `autotests = false`; do not add an unregistered Rust test file here. Add portable
facade/Registry behavior to `facade_contract.rs`, and place provider protocol, transport, policy,
or wire behavior in the crate that owns it.

External wire fixtures live beside the protocol or provider test that executes them. Do not add a
central snapshot archive or a separate fixture inventory: an unreferenced fixture is stale data,
not test coverage.

Credentialed smoke tests are intentionally not launched from a generic facade script. A new live
test must be an explicit, opt-in target in the provider package whose endpoint, credentials, and
model policy it exercises.
