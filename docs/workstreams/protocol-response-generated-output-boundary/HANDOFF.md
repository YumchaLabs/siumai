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

## Active Decision

Do not force protocol response parsers through `GenerateTextContentPart` until lossiness is proven.
Instead, first extract parser-local response compatibility adapters that:

- emit legacy `ContentPart` only as a compatibility payload;
- initialize request-side `provider_options` with empty defaults;
- preserve provider response metadata exactly;
- document which shapes can later project into generated-output parts.

## Last Completed Task

PRG-060:

- Status: DONE.
- Scope:
  `siumai-bridge/src/response,siumai-bridge/src/stream`
- Result:
  Recorded the bridge ownership decision, added source guards, and passed the required bridge
  response gate.

## Next Executable Task

PRG-070:

- Scope:
  `siumai-provider-gemini,siumai-provider-amazon-bedrock`
- Goal:
  Audit provider-owned response parsers that are not pure protocol modules and decide whether they
  should adopt local response adapters or remain provider-owned exceptions.
- Important constraint:
  Bedrock is a mixed request/response file; split before broad edits.

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

## Next Recommended Action

Run PRG-070 with `run-workstream-task`. Start with `siumai-provider-gemini` because it is narrower,
then audit Bedrock separately before touching its mixed request/response chat file.
