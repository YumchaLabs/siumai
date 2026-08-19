# Facade Tests

The maintained `siumai` facade test targets are declared explicitly in `siumai/Cargo.toml` because
the crate sets `autotests = false`:

- `facade_contract.rs` verifies the provider-neutral family modules, Registry behavior, and
  feature-gated provider exports.
- `facade_migration_contract.rs` verifies the current migration-facing call and error shape.
- `siumai_builder_contract.rs` is the focused compile-oriented contract for the typed
  `Siumai::builder()` journey, provider construction stages, reusable hubs, family clients, and
  concrete provider/model access.

Run the maintained local suites through the cross-platform Python entry points:

```text
python3 scripts/test-workspace.py fast
python3 scripts/test-workspace.py full
```

Provider protocol, transport, policy, and wire-contract tests belong in their owning provider or
protocol crate. Run that package's focused `cargo nextest` target while iterating; the full
workspace suite covers every retained package before release.

Run the typed builder contract without default facade features and then with only the provider
feature under test:

```text
cargo nextest run -p siumai --no-default-features --test siumai_builder_contract --test-threads 1
cargo nextest run -p siumai --no-default-features --features openai --test siumai_builder_contract --test-threads 1
```

Do not add an unregistered Rust test file here. Add provider-neutral family and Registry behavior
to `facade_contract.rs`, typed facade construction and binding contracts to
`siumai_builder_contract.rs`, and provider protocol, transport, policy, or wire behavior to the
crate that owns it.

External wire fixtures live beside the protocol or provider test that executes them. Do not add a
central snapshot archive or a separate fixture inventory: an unreferenced fixture is stale data,
not test coverage.

Credentialed smoke tests are intentionally not launched from a generic facade script. A new live
test must be an explicit, opt-in target in the provider package whose endpoint, credentials, and
model policy it exercises.
