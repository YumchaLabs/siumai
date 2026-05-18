# Google Interactions Runtime Alignment

Status: Active
Last updated: 2026-05-18

## Why This Lane Exists

`@ai-sdk/google` exposes `google.interactions(...)` as a first-class language-model surface backed
by `POST /v1beta/interactions`, background interaction polling, resumable SSE streaming,
interaction cancellation, and provider metadata round-trips. Siumai already exposes the Rust package
surface and fail-fast handle so imports compile, but it intentionally does not execute that runtime
path yet.

This is too large to finish honestly inside the provider-package inventory lane. It needs its own
durable implementation and verification track.

## Relevant Authority

- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
  - `docs/adr/0007-llmclient-demotion-policy.md`
  - `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- Existing docs:
  - `docs/workstreams/ai-sdk-provider-interface-convergence/PARITY_INVENTORY.md`
  - `docs/workstreams/ai-sdk-provider-interface-convergence/JOURNAL/2026-05-18-aipc-080-google-vertex-xai.md`
- AI SDK reference:
  - `repo-ref/ai/packages/google/src/interactions/google-interactions-language-model.ts`
  - `repo-ref/ai/packages/google/src/interactions/convert-to-google-interactions-input.ts`
  - `repo-ref/ai/packages/google/src/interactions/parse-google-interactions-outputs.ts`
  - `repo-ref/ai/packages/google/src/interactions/stream-google-interactions.ts`
  - `repo-ref/ai/packages/google/src/interactions/poll-google-interactions.ts`
  - `repo-ref/ai/packages/google/src/interactions/cancel-google-interaction.ts`

## Problem

The visible Rust package surface is aligned enough for imports and construction, but
`GoogleInteractionsLanguageModel` currently returns `UnsupportedOperation` for every chat path. That
is the right boundary for the provider-interface convergence program, but not the final product
state for users who expect `google.interactions(...)` to execute.

The runtime differs from ordinary Gemini `:generateContent` in several ways:

- request body shape is `input` steps, optional `model` or `agent`, and split `generation_config` /
  `agent_config`;
- agent calls can become background interactions that require polling;
- streaming is `GET /interactions/{id}?stream=true` and can reconnect with `last_event_id`;
- abort/cancel semantics require best-effort `POST /interactions/{id}/cancel`;
- output parsing must preserve `provider_metadata.google.interactionId` and per-step `signature`;
- prior-turn compaction depends on `previousInteractionId`, `store`, and metadata round-trip.

## Target State

When this lane closes:

- `GoogleInteractionsLanguageModel` executes non-stream and stream chat requests through
  `/v1beta/interactions` rather than ordinary Gemini `:generateContent`.
- model and agent modes are both represented honestly, including warnings for unsupported
  agent-only/body combinations.
- request conversion supports system messages, user text/file parts, assistant text/reasoning/file
  parts, tool calls, tool results, provider references, signatures, and history compaction.
- response parsing emits stable `ChatStreamPart` / `ContentPart` values where possible and keeps
  provider-native replay metadata under `provider_metadata.google`.
- streaming handles reconnect and cancellation semantics without coupling the ordinary Gemini
  stream converter to Interactions.
- focused fixture tests prove behavior without network access.

## In Scope

- `siumai-provider-gemini/src/providers/gemini/interactions.rs`
- `siumai-provider-gemini/src/providers/gemini/ext/request_options.rs`
- Google Interactions request/response/stream conversion modules under `siumai-provider-gemini`
- focused fixtures under `siumai/tests/fixtures/google/interactions` if needed
- public-path tests under `siumai/tests/provider_public_path_parity_test.rs`
- provider-local tests in `siumai-provider-gemini`

## Out Of Scope

- Replacing ordinary Gemini `:generateContent` behavior.
- Moving Interactions into `siumai-core`.
- Implementing browser-only AI SDK workflow serialization hooks.
- Treating Interactions as a generic gateway provider.
- Removing the fail-fast boundary before the first runtime slice has equivalent test coverage.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Interactions must stay provider-owned, not core-owned. | High | AI SDK keeps it under `packages/google/src/interactions`; Siumai ADRs keep provider semantics in provider crates. | Core/provider coupling would grow again and undo AIPC guard work. |
| Existing `ChatCapability` can host the runtime once the converter exists. | Medium | Current handle already implements `ChatCapability` and fails fast. | A narrow trait extension or handle wrapper may be needed. |
| Fixture-driven no-network tests are enough for initial parity. | High | Existing provider parity lanes use fixture/capture transports successfully. | If the wire shape is underspecified, add reference fixtures before implementation. |
| Streaming reconnect and cancellation should be explicit follow-on slices. | High | AI SDK implements dedicated helpers for both. | A one-shot stream implementation would regress real agent runs. |

## Architecture Direction

Treat Interactions as a provider-owned runtime adjacent to, but separate from, Gemini chat.
Conversion code should have small public seams:

- request conversion from `ChatRequest` + Google provider options into Interactions request body;
- response conversion from Interactions response steps into stable response parts;
- stream conversion from Interactions SSE events into stable stream parts plus provider metadata;
- polling/cancellation helpers owned by the provider runtime.

Do not route Interactions through the OpenAI-compatible adapter or ordinary Gemini response
transformer. They do not share a wire contract even when output concepts overlap.

## Closeout Condition

This lane can close when:

- model-mode non-stream, model-mode stream, and agent polling paths are implemented or explicitly
  deferred with evidence;
- request and response converters are covered by no-network tests using AI SDK reference fixtures or
  equivalent local fixtures;
- abort/cancel and stream reconnect behavior is covered at least at helper level;
- `google.interactions(...)` no longer fail-fasts for the implemented paths;
- AIPC docs point to this lane as the runtime follow-on.
