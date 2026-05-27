# Fearless Residual Architecture Deepening

Status: Active
Last updated: 2026-05-27

## Why This Lane Exists

The AI SDK contract hardening and seam-deepening lanes closed the stream, diagnostics, tool
ownership, usage, and capability ambiguities. A follow-up architecture review found five remaining
shallow Modules whose Interfaces still expose too much implementation knowledge:

- provider registry metadata and built-in factory wiring,
- OpenAI-compatible message dialect conversion,
- bridge request normalization across protocol wire formats,
- provider contract and public path parity test harnesses,
- the ADR-0008-gated legacy `ContentPart` root compatibility move.

This lane turns those residual seams into deeper Modules, deleting pass-through code when parity
tests prove the new seam owns the behavior.

## Architecture References

- `docs/adr/0001-vercel-aligned-modular-split.md`
- `docs/adr/0002-provider-crates-by-provider.md`
- `docs/adr/0006-family-model-first-trait-policy.md`
- `docs/adr/0007-llmclient-demotion-policy.md`
- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- `docs/adr/0009-ai-sdk-contract-seam-deepening.md`
- `docs/workstreams/fearless-ai-sdk-seam-deepening/`
- `repo-ref/ai/packages/provider/`
- `repo-ref/ai/packages/provider-utils/`
- `repo-ref/ai/packages/openai-compatible/`

## Target State

- Registry provider facts cross one descriptor seam before catalog views, built-in registration,
  default-model lookup, and factory tests consume them.
- OpenAI-compatible message dialect conversion is no longer a broad utilities bag; dialect-specific
  conversion and common OpenAI protocol helpers live behind smaller deep Modules.
- Bridge request normalization keeps its public helpers stable but routes each wire format through
  a codec Module, with shared content/tool helpers internal to that seam.
- Provider factory and public path parity tests use scenario/harness seams instead of large manual
  matrices where one test Interface is nearly as large as the implementation.
- `ContentPart` root compatibility is either completed under ADR-0008 gates or blocked by a
  concrete gate record that future work can satisfy without rediscovery.

## Non-Goals

- Do not change Hajimi adapters in this lane.
- Do not reopen the already closed AI SDK stream/usage/tool seam work unless a new slice proves a
  direct dependency.
- Do not split published crates unless a task proves a crate seam is necessary.
- Do not break public root `ContentPart` paths until ADR-0008 parity gates are proven in the same
  task.
- Do not replace provider-specific typed options with a generic map-only design.

## Candidate Map

| Candidate | Recommendation | Main Modules |
| --- | --- | --- |
| Registry provider descriptor seam | Strong | `siumai-registry/src/{provider_catalog.rs,native_provider_metadata.rs,provider/catalog_ids.rs,registry/*}` |
| OpenAI-compatible message dialect conversion | Strong | `siumai-protocol-openai/src/standards/openai/{utils.rs,compat,transformers}` |
| Bridge request codecs | Worth exploring | `siumai-bridge/src/request/normalize.rs`, protocol request parsers |
| Provider contract harness | Worth exploring | `siumai-registry/src/registry/factories/contract_tests.rs`, facade public path parity tests |
| `ContentPart` root compatibility move | Speculative / ADR-gated | `siumai-spec`, `siumai-core`, `siumai`, protocol parsers |

## Risk Controls

- Keep public behavior stable unless a task explicitly states a breaking cleanup.
- Use source guards only as secondary proof; the primary test surface should be the deepened Module
  Interface.
- Every implementation task updates root and touched crate changelogs when behavior or public
  contracts change.
- Use crate-scoped formatting and nextest gates for touched crates; record any Windows path-length
  limitations explicitly.
- `ContentPart` work must first prove ADR-0008 root-move gates or stop with a concrete blocker.
