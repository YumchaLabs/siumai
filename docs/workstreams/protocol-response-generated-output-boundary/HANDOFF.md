# Protocol Response Generated-Output Boundary — Handoff

Status: Closed
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

PRG-090:

- Status: DONE.
- Scope:
  `docs/workstreams/protocol-response-generated-output-boundary`
- Result:
  Closed the workstream after fresh docs-only verification; metadata, milestones, TODO, and
  evidence now all agree the lane is closed.

## Final State

- Workstream status: closed.
- `WORKSTREAM.json`: `status=closed`, `active_task=null`, `next_task=null`.
- Final gates:
  - `python -c "import json, pathlib; data=json.loads(pathlib.Path('docs/workstreams/protocol-response-generated-output-boundary/WORKSTREAM.json').read_text(encoding='utf-8')); assert data['status']=='closed' and data['active_task'] is None and data['next_task'] is None and data['continue_policy']['default_action']=='closed'"`
  - `python -c "from pathlib import Path; files=['docs/workstreams/protocol-response-generated-output-boundary/DESIGN.md','docs/workstreams/protocol-response-generated-output-boundary/MILESTONES.md','docs/workstreams/protocol-response-generated-output-boundary/TODO.md','docs/workstreams/protocol-response-generated-output-boundary/EVIDENCE_AND_GATES.md','docs/workstreams/protocol-response-generated-output-boundary/HANDOFF.md']; assert all('Status: Closed' in Path(p).read_text(encoding='utf-8') for p in files); todo=Path('docs/workstreams/protocol-response-generated-output-boundary/TODO.md').read_text(encoding='utf-8'); assert '[x] PRG-090' in todo and '[ ] PRG-090' not in todo"`
  - `git diff --check -- docs/workstreams/protocol-response-generated-output-boundary`
- Target state met:
  - docs distinguish parser-local response compatibility adapters from spec-owned generated-output
    projection;
  - legacy `ContentPart` remains compatibility-only;
  - no new public response model was introduced.

## Blockers

None known.

## Risks

- No active blockers. The remaining risk is future scope drift if parser-wide generated-output
  migration is reopened without a separate ADR-backed lane.

## Next Recommended Action

If future parser-wide generated-output migration becomes necessary, open a new workstream with a
fresh ADR-backed scope. Otherwise no further action is required in this lane.
