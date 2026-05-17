# AI SDK Provider Interface Convergence - Milestones

Status: Active
Last updated: 2026-05-18

## M0 - Program Scope And Inventory

Exit criteria:

- The program lane has machine-readable status.
- ADR and workstream authority is explicit.
- Target seams are named.
- Initial AI SDK provider-interface parity inventory exists.
- First executable code slice is selected.

Primary evidence:

- `docs/workstreams/ai-sdk-provider-interface-convergence/DESIGN.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/TODO.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/PARITY_INVENTORY.md`

Status: completed

## M1 - Core And Registry Guard Slices

Exit criteria:

- `siumai-core` has guard coverage for provider-specific residue returning to provider-agnostic
  runtime space.
- `siumai-registry` has guard coverage for stable family handles avoiding primary execution through
  generic-client compatibility paths where native family paths exist.
- Any real drift found by guards is either fixed or split into a bounded child task.

Primary gates:

```bash
cargo nextest run -p siumai-core --no-fail-fast
cargo nextest run -p siumai-registry --no-fail-fast
```

Status: completed

## M2 - Stream Semantics Convergence

Exit criteria:

- Provider stream parsers and serializers prefer `ChatStreamEvent::Part(ChatStreamPart)` for
  AI SDK-stable semantics.
- Protocol-only replay hints remain outside generic provider metadata.
- Gateway and bridge tests assert stable stream parts where that is the intended downstream
  contract.

Primary gates:

```bash
cargo nextest run -p siumai-core --no-fail-fast
cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast
cargo nextest run -p siumai-bridge --no-fail-fast
```

Status: completed

Progress note: AIPC-050 and AIPC-060 are complete. Provider stream parsers/serializers, bridge
paths, and extras gateway helpers now have stable-part-first regression coverage for the highest-risk
tool stream paths audited in this milestone.

## M3 - Provider Package Parity

Exit criteria:

- Every built-in provider row in `PARITY_INVENTORY.md` has one of:
  - green parity,
  - an intentional Rust-specific decision,
  - a child workstream,
  - or an explicit deferred reason.
- OpenAI-compatible promoted vendors do not inherit unsupported package capabilities by accident.
- Provider-specific typed options and metadata helpers stay provider-owned.

Primary gates:

```bash
cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast
cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast
cargo nextest run --profile ci --all-features --workspace
```

Status: in progress

## M4 - Workstream Hygiene And Closeout

Exit criteria:

- Legacy unknown lanes touched by this program are normalized, superseded, or left with explicit
  handoff notes.
- `WORKSTREAM.json` records the final state.
- `EVIDENCE_AND_GATES.md` records the final gate set and command outcomes.
- Any remaining work is split into narrower follow-ons.

Primary gates:

```bash
cargo fmt --all -- --check
cargo nextest run --profile ci --all-features --workspace
```

Status: not started
