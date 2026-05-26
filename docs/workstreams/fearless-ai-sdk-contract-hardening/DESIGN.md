# Fearless AI SDK Contract Hardening

Status: Closed
Last updated: 2026-05-26

## Why This Lane Exists

Siumai exposes AI SDK-aligned stream, tool, usage, metadata, cancellation, and error surfaces, but
some cross-crate contracts remain implicit. Downstream adapters can mis-handle those implicit
contracts by appending final replay content twice, dropping final text that only appears at stream
end, leaking raw provider metadata, or treating unsafe provider errors as public messages.

This lane exists to harden those boundaries from first principles instead of adding adapter-side
patches around ambiguous behavior.

## Relevant Authority

- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
  - `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- Existing docs:
  - `docs/architecture/public-surface.md`
  - `CHANGELOG.md`
- AI SDK reference:
  - `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-stream-part.ts`
  - `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-stream-result.ts`
  - `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-generate-result.ts`
  - `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-usage.ts`
- Related workstreams:
  - `docs/workstreams/stream-delta-lossless-boundary`
  - `docs/workstreams/stream-metadata-parity-hardening`
  - `docs/workstreams/openai-compatible-reasoning-policy-alignment`
  - `docs/workstreams/openai-compatible-usage-policy-alignment`
  - `docs/workstreams/protocol-response-generated-output-boundary`

## Problem

The current public and internal contracts are flexible enough to preserve provider data, but not
explicit enough to tell consumers which fields are final replay, delta, safe public data, private
diagnostics, provider-executed tool ownership, or cumulative usage.

The audit found twelve contract gaps:

1. `ChatStreamEvent::StreamEnd { response }` does not state whether `response.content` is full final
   replay, supplemental content, or provider-optional replay.
2. Reasoning deltas and final visible text can be separated by provider behavior, but the exact
   fallback contract lacks a fixture.
3. Cancelable text streams have first-class handles, but docs should make `stream_with_cancel` the
   recommended cancelable path and distinguish local cancellation from provider remote abort.
4. `ContentPart::ToolCall.provider_metadata` may contain provider-specific raw data without a
   public/private classification.
5. `ChatStreamEvent::Custom` and `ChatStreamPart::Raw` can carry raw provider event data, but the
   diagnostics ownership is not explicit.
6. `Tool::function(name, ...)` does not validate a generic tool-name contract.
7. `provider_executed` needs an execution-owner definition that covers hosted/provider-executed
   tools and legacy provider-defined constructors.
8. `ToolInputStart` preserves `dynamic` and `title`, while provider-specific index data only exists
   as replay/provider metadata; this needs consumer-facing wording and tests.
9. `Usage` in stream finish parts and final responses lacks a single-call cumulative snapshot
   contract; repeated usage merging can over-count if providers emit cumulative snapshots.
10. `ResponseMetadata.headers` and `ResponseMetadata.body` can include raw/private provider data.
11. `LlmErrorExt::user_message()` is not guaranteed safe because fallback paths expose provider
    detail through `Display`.
12. Unsupported provider capability behavior is mixed across warnings, validation errors, and
    provider fallback paths.

## Target State

When this lane closes:

- Stream final replay semantics are documented and tested: `StreamEnd.response.content` is final
  response replay/fallback, not an append-only delta.
- Reasoning/text separation has a contract fixture that proves final visible text is not lost when
  it only materializes in `StreamEnd.response.content`.
- Cancelable stream docs and tests steer callers to `stream_with_cancel`, and remote abort support is
  described as provider-specific unless a provider proves it.
- Provider metadata, raw stream events, response headers/body, and raw error details have an
  explicit private diagnostics contract.
- Public error messages are separate from raw diagnostic detail.
- Tool name validation and provider-executed ownership semantics are explicit and guarded.
- `ToolInputStart` projection preserves meaningful stable fields and explains provider-specific
  replay index handling.
- Stream usage semantics are defined as a single provider-call final snapshot unless a future
  provider-specific contract explicitly says otherwise.
- Unsupported capability behavior is documented as reject, warn, or provider fallback with tests for
  the shared cases.
- Root and crate changelogs track the user-visible contract changes.

## In Scope

- `siumai-spec` stream, content, tool, usage, metadata, warning, and error-facing type contracts.
- `siumai-core` stream processor, cancellation wrappers, error policy, and capability validation
  contracts.
- `siumai-protocol-openai` fixtures where OpenAI Responses replay and raw event behavior prove the
  contract.
- Provider-specific tests only when needed to prove a shared contract with real provider behavior.
- Documentation, changelog, and workstream evidence updates.

## Out Of Scope

- Hajimi adapter changes.
- Provider model catalog refreshes.
- Live-network provider validation.
- Reopening closed workstreams for unrelated cleanup.
- Broad public API removals unless required to make the contract safe.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| AI SDK V4 is the contract reference for stream parts, provider metadata, tools, usage, and warnings. | High | `repo-ref/ai/packages/provider/src/language-model/v4/*` | If upstream changed, refresh the reference before coding. |
| `StreamEnd.response.content` is best modeled as final replay/fallback, not delta. | High | OpenAI completion emits deltas and then a full final response. | If a provider uses it as delta, provider code must be normalized or documented as a bug. |
| Raw metadata and response bodies should be private by default. | High | `ResponseMetadata.body` is documented as raw response body. | Public projections must whitelist safe fields instead of passing through all metadata. |
| Cancellation can be unified at the Siumai handle level while remote provider abort remains provider-owned. | High | Core default wraps streams; OpenAI Responses has remote cancel coverage. | Docs must avoid promising remote cancel for providers that only support local cancellation. |

## Architecture Direction

The spec crate owns portable contract language and serde-visible fields. Core owns runtime behavior
that combines stream parts into final responses, handles cancellation, classifies public errors, and
normalizes capability failures. Protocol crates own provider wire replay and raw event handling.
Provider crates prove real provider behavior through fixtures without widening the stable unified
facade. The facade should expose safe defaults and documentation, not provider raw internals.

Prefer contract tests over adapter-specific patches. When a field can carry raw provider data, keep
the raw value available for diagnostics but make the safe public projection explicit.

## Unsupported Capability Policy

Siumai follows AI SDK's warning surface for non-fatal unsupported behavior: unsupported settings and
compatibility fallbacks are returned as call warnings, including stream-start warnings. The Siumai
contract adds an explicit policy step before warning/error projection so request projection cannot
silently ignore caller settings that change semantics:

- `reject`: fail before provider execution when continuing would call the wrong model family,
  endpoint, transport, or tool execution owner.
- `warn`: continue only when Siumai can name the unsupported feature and surface an AI SDK-style
  `Warning::Unsupported { feature, details }`.
- `provider-fallback`: continue when support is intentionally delegated to the provider; surface a
  compatibility warning when Siumai reports that delegation, and treat the provider response/error
  as authoritative.

Provider-specific request projection can still decide which behavior applies to each option, but it
must make that decision explicitly. Broad provider-by-provider rewrites are follow-on work if a
single adapter has many unsupported option cases.

## Closeout Condition

This lane can close when:

- all AICH tasks in `TODO.md` are done, deferred with rationale, or split into narrower follow-ons,
- fresh gates in `EVIDENCE_AND_GATES.md` pass,
- root and touched crate changelogs describe the contract changes,
- workstream docs and `WORKSTREAM.json` agree on final status,
- and Hajimi can safely consume Siumai without relying on undocumented replay or raw-data behavior.

Result: closed on 2026-05-26. All AICH tasks are complete, fresh spec/core/protocol gates passed,
and any remaining Hajimi adapter work is explicitly out of scope.
