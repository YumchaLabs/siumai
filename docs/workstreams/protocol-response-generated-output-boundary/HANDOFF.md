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

## Active Decision

Do not force protocol response parsers through `GenerateTextContentPart` until lossiness is proven.
Instead, first extract parser-local response compatibility adapters that:

- emit legacy `ContentPart` only as a compatibility payload;
- initialize request-side `provider_options` with empty defaults;
- preserve provider response metadata exactly;
- document which shapes can later project into generated-output parts.

## Last Completed Task

PRG-030:

- Status: DONE.
- Scope:
  `siumai-protocol-openai/src/standards/openai/transformers/response`
- Result:
  Classified OpenAI Responses generated-output lossiness and added projection-boundary guards.

## Next Executable Task

PRG-040:

- Scope:
  `siumai-protocol-anthropic/src/standards/anthropic/utils/parse.rs`
- Goal:
  Apply the response-adapter pattern to Anthropic response parsing or document why Anthropic
  citations/tool-use require a different adapter shape.
- Important constraint:
  Keep citation/source metadata provider-owned and response-side; avoid creating a generic
  catch-all adapter if Anthropic needs a provider-specific shape.

## Blockers

None known.

## Risks

- OpenAI Responses parser is large and feature-rich; extraction should be incremental.
- Generated-output projection is intentionally fallible; using it too early can drop files, images,
  audio, tool-approval context, or provider metadata.
- Bedrock and some gateway/proxy files mix request and response responsibilities; avoid broad edits
  there until parser-local proofs exist.

## Next Recommended Action

Run PRG-040 with `run-workstream-task`. Start with a narrow Anthropic parser scan and either extract
a parser-local response compatibility adapter or record a documented provider-specific exception.
Do not start Gemini or bridge migration until the Anthropic proof is recorded or intentionally
deferred.
