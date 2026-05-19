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

## Active Decision

Do not force protocol response parsers through `GenerateTextContentPart` until lossiness is proven.
Instead, first extract parser-local response compatibility adapters that:

- emit legacy `ContentPart` only as a compatibility payload;
- initialize request-side `provider_options` with empty defaults;
- preserve provider response metadata exactly;
- document which shapes can later project into generated-output parts.

## First Executable Task

PRG-020:

- Scope:
  `siumai-protocol-openai/src/standards/openai/transformers/response`
- Goal:
  Extract OpenAI Responses response-owned legacy `ContentPart` construction behind a named response
  compatibility adapter module without changing serialized output.
- Guard:
  `responses_response_transformer_source_does_not_emit_request_provider_options`
- Important constraint:
  If the whole OpenAI Responses transformer is too large, split PRG-021 for text/reasoning first.

## Blockers

None known.

## Risks

- OpenAI Responses parser is large and feature-rich; extraction should be incremental.
- Generated-output projection is intentionally fallible; using it too early can drop files, images,
  audio, tool-approval context, or provider metadata.
- Bedrock and some gateway/proxy files mix request and response responsibilities; avoid broad edits
  there until parser-local proofs exist.

## Next Recommended Action

Run PRG-020 with `run-workstream-task` after committing or explicitly accepting this planning
workstream. If code edits start, keep file scope narrow and validate with the OpenAI Responses
source guard before expanding.
