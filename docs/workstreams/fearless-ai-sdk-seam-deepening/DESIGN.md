# Fearless AI SDK Seam Deepening

Status: Closed
Last updated: 2026-05-27

## Why This Lane Exists

`fearless-ai-sdk-contract-hardening` made the AI SDK-facing stream, diagnostics, tool, usage, and
capability contracts explicit. The follow-up architecture review found that several of those rules
are still shallow: the interfaces expose nearly as much caller knowledge as the implementation.

This lane turns the remaining caller knowledge into deep modules with named seams and stronger
locality.

## Contract References

- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-stream-part.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-stream-result.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-usage.ts`
- `repo-ref/ai/packages/provider/src/shared/v4/shared-v4-warning.ts`

## Architecture References

- `docs/adr/0001-vercel-aligned-modular-split.md`
- `docs/adr/0006-family-model-first-trait-policy.md`
- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- `docs/adr/0009-ai-sdk-contract-seam-deepening.md`
- `docs/workstreams/fearless-ai-sdk-contract-hardening/`

## Target State

- OpenAI Responses stream state is internally decomposed into deep modules for stream state, replay,
  terminal buffering, reasoning lifecycle, provider tool ownership, and serializer state.
- Public provider metadata versus private diagnostics crosses an executable projection seam before
  user-visible output.
- Provider-executed tool ownership is owned by one contract module and reused by prompt validation,
  runtime tooling, UI conversion, and protocol adapters.
- Unsupported capability gates are centralized so family executors do not repeat guard choreography.
- Stream usage snapshot replacement is represented by a named module and kept distinct from
  `Usage::merge()` aggregation.
- Stream final response assembly receives a compact accumulated record instead of reaching across
  broad processor internals.
- Obsolete pass-through helpers and duplicated compatibility code are deleted once parity tests
  prove the deepened module owns the behavior.

## Non-Goals

- Do not change Hajimi adapters in this lane.
- Do not publish a breaking public `ContentPart` move; ADR-0008 still controls that later slice.
- Do not move OpenAI wire-specific behavior into `siumai-spec`.
- Do not split crates unless a task proves a crate seam is required.
- Do not make provider behavior less lossless to simplify the architecture.

## Candidate Map

| Candidate | Recommendation | Main modules |
| --- | --- | --- |
| OpenAI Responses stream state | Strong | `siumai-protocol-openai/src/standards/openai/responses_sse/converter/*` |
| Diagnostics projection | Strong | `siumai-spec/src/types/{provider_metadata,streaming,http}.rs`, protocol adapters |
| Provider-executed tool ownership | Strong | `siumai-spec/src/types/tools`, `siumai-core/src/tooling`, protocol adapters |
| Unsupported capability gates | Worth exploring | `siumai-core/src/execution/executors/*`, `traits/capabilities.rs` |
| Usage snapshot ledger | Worth exploring | `siumai-spec/src/types/usage.rs`, `siumai-core/src/streaming` |
| Stream final response assembly | Speculative | `siumai-core/src/streaming/processor*` |

## Risk Controls

- Keep public behavior stable unless a task explicitly states a breaking cleanup.
- Prefer targeted no-network fixtures for protocol and core behavior.
- Use crate-scoped `cargo fmt --check` when full workspace formatting hits Windows path-length
  error 206.
- Every implementation task must update root and touched crate changelogs when behavior or public
  contracts change.

## Closeout Result

Closed on 2026-05-27. The target state is met: OpenAI Responses stream state, diagnostics
projection, tool ownership, capability gates, usage snapshots, and final stream assembly now each
cross named crate-owned seams with focused regressions. No Siumai follow-up is split from this
lane; Hajimi adapter changes remain out of scope.
