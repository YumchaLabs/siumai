# AI SDK Provider Interface Convergence - Evidence And Gates

Status: Active
Last updated: 2026-05-18

## Smallest Current Repro

The active implementation slice is stream semantics convergence after the core and registry guard
slices completed.

```bash
cargo nextest run -p siumai-core --no-fail-fast
cargo nextest run -p siumai-registry --no-fail-fast
cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast
```

## Gate Set

### Program Document Gate

```bash
git status --short
```

Proves the program docs and index changes are the only changes in the opening commit.

### Core Seam Gate

```bash
cargo nextest run -p siumai-core --no-fail-fast
```

Proves provider-agnostic runtime guards and core tests still pass after seam changes.

### Registry Seam Gate

```bash
cargo nextest run -p siumai-registry --no-fail-fast
```

Proves registry family handles, factories, caches, aliases, and source guards still pass after
family-first construction changes.

### Stream Convergence Gate

```bash
cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast
cargo nextest run -p siumai-bridge --no-fail-fast
```

Proves protocol-owned stream/parser/serializer changes preserve no-network fixture and bridge
behavior.

### Provider Parity Gate

```bash
cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast
cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast
```

Proves public package exports and promoted OpenAI-compatible vendor behavior do not drift while
package parity slices land.

### Broad Closeout Gate

```bash
cargo nextest run --profile ci --all-features --workspace
```

Proves the full workspace still passes after this program closes. On local Windows, a package-scoped
format check may be used when `cargo fmt --all -- --check` hits command-line length limits; CI
should still run the full formatting gate.

## Evidence Anchors

- `docs/workstreams/ai-sdk-provider-interface-convergence/DESIGN.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/TODO.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/MILESTONES.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/PARITY_INVENTORY.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/HANDOFF.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/WORKSTREAM.json`
- `docs/adr/0001-vercel-aligned-modular-split.md`
- `docs/adr/0006-family-model-first-trait-policy.md`
- `docs/adr/0007-llmclient-demotion-policy.md`
- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`

## Command Log

| Date | Command | Result | Notes |
| --- | --- | --- | --- |
| 2026-05-18 | `git status --short --branch` | Passed | Opening worktree was clean on `main`. |
| 2026-05-18 | `cargo fmt -p siumai-core --check` | Passed | Formatting gate for the AIPC-030 core guard slice. |
| 2026-05-18 | `cargo nextest run -p siumai-core core_standards_module_does_not_reintroduce_provider_protocol_islands --no-fail-fast` | Passed | Proves `siumai-core::standards` cannot regain provider/protocol island directories or modules. |
| 2026-05-18 | `cargo nextest run -p siumai-registry stable_registry_handles_do_not_use_compat_client_paths_for_primary_family_execution --no-fail-fast` | Passed | Reconfirmed the existing registry stable-family handle guard. |
| 2026-05-18 | `cargo nextest run -p siumai-registry --no-fail-fast` | Passed | 96 tests passed; existing warnings are unrelated dead-code warnings in test support. |
| 2026-05-18 | `cargo nextest run -p siumai-core --no-fail-fast` | Passed | 426 tests passed after moving the new guard into the existing integration boundary test suite. |
| 2026-05-18 | `cargo fmt -p siumai-registry --check` | Passed | Formatting gate for the AIPC-040 registry handle boundary slice. |
| 2026-05-18 | `cargo nextest run -p siumai-registry remaining_registry_handle_compat_paths_are_extension_only --no-fail-fast` | Passed | Proves remaining registry compat client paths are isolated to extension-only image/audio helpers. |
| 2026-05-18 | `cargo nextest run -p siumai-registry --no-fail-fast` | Passed | 97 tests passed; existing warnings are unrelated dead-code warnings in test support. |
| 2026-05-18 | `cargo fmt -p siumai-protocol-openai --check` | Passed | Formatting gate for the OpenAI Responses AIPC-050 stream-part slice. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses --test responses_sse_feature_surface_test --no-fail-fast` | Passed | Proves the public OpenAI Responses feature surface roundtrips stable `ChatStreamEvent::Part` tool call/result parts. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-openai responses_feature_surface_uses_stable_parts_for_tool_stream_parts --no-fail-fast` | Passed | Proves the public OpenAI Responses feature surface does not regress to provider custom events for stable tool stream parts. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast` | Passed | 448 tests passed for the OpenAI protocol package after the first AIPC-050 slice. |
