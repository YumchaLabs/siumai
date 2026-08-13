# Contributing to Siumai

Thank you for improving Siumai. This workspace is a Rust-first provider integration library, so a
change should deepen one clear owner instead of adding another compatibility layer.

## Start with the owning contract

Read [`AGENTS.md`](AGENTS.md) and the relevant document under [`docs/architecture`](docs/architecture)
before changing a public type or package dependency. The main ownership rules are:

- `siumai-core` owns provider-neutral model-family contracts and shared data types;
- `siumai-transport` owns network mechanics, endpoint policy, replay safety, and resource bounds;
- `siumai-protocol-*` owns wire schemas, codecs, and stream state machines;
- `siumai-provider-*` owns provider construction, credentials, technical endpoints, typed options,
  request policy, support evidence, and native resources;
- `siumai-registry` owns deterministic local route lookup only;
- `siumai-runtime` owns provider-neutral multi-step execution;
- `siumai` owns curated re-exports and feature aggregation.

Inspect the affected manifests and run `scripts/check_workspace_boundaries.py` before adding a
workspace dependency. Provider crates must not depend on other branded provider crates, Registry,
runtime, or the facade. Cargo remains authoritative for workspace membership, versions, MSRV, and
dependency resolution.

## Provider change checklist

A named provider, platform, model family, API mode, resource, session, or job is ready to claim only
when the change includes the applicable items below:

- an official documentation source and verification date;
- exact provider, technical platform, family, protocol, and API-mode identity, or a provider-native
  surface identity;
- explicit fidelity (`native` or `verified-compatible`) and public stability;
- a long-lived configured provider with synchronous, network-free model construction;
- open model IDs, with dated constants and lifecycle rows treated as advisory hints;
- typed provider options or annotations with provider-owned validation and merge rules;
- focused offline fixtures for distinct request, response, stream, error, and policy branches;
- sanitized errors and bounded response, stream, queue, and resource handling;
- facade features, rustdoc, support matrix, changelog, and migration updates when public scope changes.

Custom endpoints remain generic unless separate evidence proves a named profile. Do not encode
account entitlement, commercial region availability, pricing, quota, health, or fallback policy in
provider support declarations.

Prefer a native protocol when semantics differ materially, a verified compatibility profile when
the shared dialect is accurate, and an explicit custom-compatible configuration as the fallback.

## Tests and local checks

Run Cargo commands serially and reuse the workspace target directory. Start with the smallest
meaningful package lane:

```text
cargo fmt --all -- --check
cargo nextest run -p <package> --all-features -j 1
cargo clippy -p <package> --all-targets --all-features -j 1 -- -D warnings
```

For repository-level checks:

```text
python3 -B -m unittest discover -s scripts/tests -p "test_*.py"
python3 -B scripts/check_workspace_boundaries.py
python3 -B scripts/test-workspace.py full --runner nextest
```

Default tests must be deterministic, offline, secret-free, and non-billable. Do not add repeated
smoke wrappers or source-scanning tests when a focused Rust contract test, `cargo metadata`, Clippy,
rustdoc, or nextest already proves the behavior.

## Documentation and changes

Write source code, comments, rustdoc, technical documentation, changelog entries, and commit messages
in English. Keep examples small and declare their required features. Record durable decisions in
`docs/adr/`, current boundaries in `docs/architecture/`, dated provider evidence in
`docs/providers/`, and user-facing breaking changes in `docs/migration/`.

Use English Conventional Commit messages. Keep commits reviewable, stage only files that belong to
the change, and never commit credentials, local paths, build output, or files copied from `repo-ref/`.
Publishing crates, creating releases, pushing branches, and opening pull requests require explicit
maintainer authorization.
