# Protocol Response Generated-Output Boundary — Handoff

Status: Active
Last updated: 2026-05-19

## Current State

This workstream was opened after `content-part-compat-namespace-break` closed and split protocol /
parser generated-output migration as future work.

Already shipped before this lane:

- legacy `ContentPart` has an explicit public compatibility namespace;
- `siumai::prelude::unified::*` no longer exports legacy `ContentPart`;
- spec-owned `response_compat_projection.rs` owns legacy `ContentPart` ->
  `GenerateTextContentPart` projection;
- bridge request normalization already centralizes request-side legacy content construction.

This lane starts from the remaining gap: protocol/provider response parsers still construct legacy
`ContentPart` inline, even when directionality guards prevent obvious provider-map leakage.

PRG-020 is complete:

- `siumai-protocol-openai/src/standards/openai/transformers/response/responses/response_content.rs`
  now owns OpenAI Responses response-side legacy content compatibility constructors.
- The broad OpenAI Responses response transformer delegates legacy content construction to
  `response_content::*`.
- Source guards verify both sides of the boundary:
  - the transformer no longer directly constructs legacy content variants;
  - the adapter only initializes empty legacy request provider options and does not read request
    provider option maps.
- Existing OpenAI Responses transformer behavior tests pass.

PRG-030 is complete:

- OpenAI Responses output shapes are classified in `EVIDENCE_AND_GATES.md` by generated-output
  projection lossiness.
- A source guard verifies production OpenAI Responses parsing does not call
  `GenerateTextContentPart` projection helpers directly.
- Behavior tests prove a lossless text/reasoning/function-call subset can project, while MCP
  approval requests and output-only hosted tool results are rejected rather than forced through a
  lossy projection.

PRG-040 is complete:

- `siumai-protocol-anthropic/src/standards/anthropic/utils/parse/response_content.rs` now owns
  Anthropic response-side legacy content compatibility constructors.
- `parse.rs` delegates response content construction to the local adapter while preserving
  Anthropic-specific citation/source/tool-use metadata behavior.
- Source guards cover legacy-construction delegation, empty request-provider-option defaults, and
  no direct generated-output projection.

PRG-050 is complete:

- `siumai-protocol-gemini/src/standards/gemini/transformers/response/response_content.rs` now owns
  Gemini response-side legacy content compatibility constructors.
- Gemini was not a no-op candidate; direct response `ContentPart` construction moved behind the
  parser-local adapter.
- Source guards cover legacy-construction delegation, empty request-provider-option defaults, and
  no direct generated-output projection.
- Grounding, URL context, safety, logprobs, sources, usage, service tier, and finish-message
  metadata remain on response-side provider metadata.

PRG-060 is complete:

- Bridge response/stream paths were audited and intentionally kept primitive-only:
  `ChatResponse` / `ChatStreamEvent` in, target JSON/SSE bytes or values out, plus `BridgeReport`
  loss accounting.
- Bridge delegates wire response and stream encoding to protocol-owned JSON/SSE converters through
  `target_dispatch.rs`.
- Bridge does not use parser-local protocol `response_content` adapters and does not call
  generated-output projection helpers directly.
- `OpenAiResponsesStreamPartsBridge` remains a narrow stream replay shim for cross-protocol
  gateway/proxy use-cases, not the canonical owner of OpenAI Responses response semantics.

PRG-070 is complete:

- `siumai-provider-gemini/src/providers/gemini/interactions/response/response_content.rs` now owns
  Google Interactions response-side legacy compatibility constructors.
- `siumai-provider-gemini/src/providers/gemini/interactions/response.rs` delegates response
  content construction to the local adapter while preserving interaction ids, signatures,
  built-in tool calls/results, image outputs, source citations, usage, service tier, and provider
  metadata.
- `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/response_content.rs` now owns Bedrock
  response-side legacy compatibility constructors.
- Bedrock non-stream response parsing and stream final-response aggregation both use the local
  adapter, while request conversion remains untouched.
- Source guards verify provider-owned parsers do not use generated-output projection helpers
  directly and only initialize request-side `provider_options` as empty compatibility defaults.

## Active Decision

Do not force protocol response parsers through `GenerateTextContentPart` until lossiness is proven.
Instead, first extract parser-local response compatibility adapters that:

- emit legacy `ContentPart` only as a compatibility payload;
- initialize request-side `provider_options` with empty defaults;
- preserve provider response metadata exactly;
- document which shapes can later project into generated-output parts.

PRG-080 is complete:

- `docs/architecture/public-surface.md` now explains response parser adapters as internal
  compatibility seams for legacy `ChatResponse` / `MessageContent` payloads.
- `docs/migration/migration-0.11.0-beta.7.md` now separates parser-local response compatibility
  adapters from spec-owned generated-output projection helpers.
- The migration guide keeps `ContentPart` compatibility-only and warns that hosted tool results,
  approval requests, files, images, audio, and provider-specific metadata should not be forced
  through generated-output projection without proven losslessness.
- No new public response model was proposed; an ADR is still required before adding one.

## Last Completed Task

PRG-080:

- Status: DONE.
- Scope:
  `docs/architecture,docs/migration,docs/workstreams`
- Result:
  Updated public architecture and beta.7 migration docs to distinguish parser-local response
  compatibility adapters, legacy compatibility payloads, and fallible generated-output projection.

## Next Executable Task

PRG-090:

- Scope:
  `docs/workstreams/protocol-response-generated-output-boundary`
- Goal:
  Close this lane or split remaining parser-wide generated-output migration into narrower
  follow-ons.
- Important constraint:
  Final status must name retained compatibility paths and their removal/narrowing criteria.

## Blockers

None known.

## Risks

- OpenAI Responses parser is large and feature-rich; extraction should be incremental.
- Generated-output projection is intentionally fallible; using it too early can drop files, images,
  audio, tool-approval context, or provider metadata.
- Bedrock and some gateway/proxy files mix request and response responsibilities; avoid broad edits
  there until parser-local proofs exist.
- Bridge stream replay can become a semantic sink if expanded casually; split a follow-on before
  adding richer provider-specific response JSON/SSE reconstruction.
- Closeout can accidentally promise removal of compatibility payloads too early. PRG-090 should
  preserve the current compatibility contract unless it opens an ADR-backed public output model
  follow-on.

## Next Recommended Action

Run PRG-090 with `close-workstream` or `verify-rust-workstream`. Decide whether this lane is ready
to close after PRG-080's documentation boundary, or split follow-ons for broader parser-wide
generated-output migration.
