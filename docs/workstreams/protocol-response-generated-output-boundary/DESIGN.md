# Protocol Response Generated-Output Boundary — Design

Status: Active
Last updated: 2026-05-19

## Why This Lane Exists

`content-part-compat-namespace-break` made the public namespace honest: legacy `ContentPart` is now
an explicit compatibility carrier, and response-side projection from legacy chat content into
`GenerateTextContentPart` lives in `siumai-spec/src/types/ai_sdk/response_compat_projection.rs`.

That work deliberately deferred protocol/provider parser migration. The reason is architectural:
OpenAI Responses, Anthropic, Gemini, Bedrock, bridge response encoders, and stream replay still
produce provider-native `ChatResponse` / `ContentPart` payloads directly. Some of those payloads can
project into generated-output parts without loss, but several cannot:

- tool approval requests/responses need surrounding tool-call context;
- URL-backed generated files and media cannot become `GeneratedFile` without data loss;
- image/audio response payloads are ambiguous in the high-level generated-text projection;
- provider-owned source, citation, reasoning, and hosted-tool metadata must keep exact namespaces.

This lane exists to make the protocol response boundary explicit before attempting broad rewrites.
The first architectural win is not “convert everything to `GenerateTextContentPart` immediately”.
The first win is to move parser-owned legacy response construction behind named response adapters,
with lossiness criteria and tests that prevent request-side `providerOptions` from leaking into
response payloads.

## Relevant Authority

- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- Existing docs:
  - `docs/architecture/public-surface.md`
  - `docs/migration/migration-0.11.0-beta.7.md`
  - `docs/workstreams/fearless-spec-core-boundary-convergence/content-part-construction-audit.md`
- Related workstreams:
  - `docs/workstreams/content-part-compat-namespace-break/`
  - `docs/workstreams/fearless-module-deepening/`
  - `docs/workstreams/fearless-spec-core-boundary-convergence/`
  - `docs/workstreams/generate-text-output-alignment/`
  - `docs/workstreams/stream-delta-lossless-boundary/`

## Problem

Protocol and provider response parsers still construct legacy `ContentPart` values inline. The
current source guards enforce directionality, but the code shape still makes the legacy enum look
like the parser's canonical output model.

The main risks are:

1. New provider response features may continue landing as ad-hoc `ContentPart::...` construction.
2. Response parsers may accidentally initialize or preserve request-side `provider_options`.
3. A future generated-output model could be blocked by scattered parser-specific construction.
4. Generated-output projection could become lossy if forced before source/tool/file semantics are
   classified.
5. Large parser files remain hard for humans and agents to navigate.

## Current Hotspots

The initial planning scan counted direct legacy construction and provider-map field initializers in
high-value response/parser areas, excluding test files where possible.

| Path | Direct constructor hits | `MessageContent::MultiModal` hits | `provider_metadata:` hits | `provider_options:` hits | Initial classification |
| --- | ---: | ---: | ---: | ---: | --- |
| `siumai-protocol-openai/src/standards/openai/transformers/response/responses.rs` | 50 | 3 | 28 | 22 | Best first proof; response-owned parser with hosted tools, sources, reasoning, files, and tool results. |
| `siumai-protocol-anthropic/src/standards/anthropic/utils/parse.rs` | 30 | 11 | 9 | 8 | Good second proof; citations/sources and reasoning/tool-use parser. |
| `siumai-protocol-gemini/src/standards/gemini/transformers/response.rs` | 15 | 1 | 1 | 13 | Good third proof; already has strict source guard for empty request options. |
| `siumai-provider-gemini/src/providers/gemini/interactions/response.rs` | 16 | 4 | 8 | 6 | Provider-owned response parser; useful after protocol proofs. |
| `siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs` | 45 | 2 | 1 | 8 | Mixed request/response file; should be split or touched only with very narrow scopes. |
| `siumai-bridge/src/response/inspect.rs` | 42 | 2 | 1 | 0 | Bridge inspection/serialization boundary; likely adapter extraction rather than generated-output conversion. |
| `siumai-protocol-openai/src/standards/openai/json_response.rs` | 42 | 10 | 18 | 17 | Gateway/proxy response encoder; classify after parser proof. |
| `siumai-protocol-anthropic/src/standards/anthropic/json_response.rs` | 41 | 12 | 19 | 19 | Gateway/proxy response encoder; classify after parser proof. |

This table is an audit starting point, not a mandate to rewrite every file in one patch.

## Target State

When this workstream closes:

- high-value protocol/provider response parsers no longer scatter direct legacy `ContentPart`
  construction in broad parser bodies;
- response-owned legacy construction is centralized behind named response compatibility adapters
  that always default request-side `provider_options` to empty values;
- each migrated parser states whether it can project into `GenerateTextContentPart` losslessly or
  must remain a `ChatResponse` compatibility payload;
- source guards prove request-side `providerOptions` are not emitted by response adapters except
  empty legacy defaults;
- docs explain the difference between:
  - protocol-native response parsing,
  - legacy `ChatResponse` compatibility payloads,
  - and high-level generated-output projection;
- any remaining direct construction sites are either explicitly deferred as lossy or split into a
  narrower follow-on.

## In Scope

- Auditing response/parser construction sites in OpenAI Responses, Anthropic, Gemini, bridge
  response/stream paths, and selected provider-owned response parsers.
- Extracting named response compatibility adapters in narrow modules.
- Adding or strengthening source guards around response construction.
- Preserving fixture parity and public `ChatResponse` / `MessageContent` serialized shape.
- Documenting lossiness criteria for generated-output projection.
- Updating workstream/audit docs as parser boundaries become explicit.

## Out Of Scope

- Removing the legacy `ContentPart` enum.
- Changing `ChatResponse.content` or `ChatMessage.content` serialized shape.
- Forcing every protocol response into `GenerateTextContentPart` before a lossless carrier exists.
- Rewriting request serializers; request-side bridge normalization has its own adapter lane.
- Broad formatting or whole-file cleanup unrelated to the touched parser seam.
- Creating a new public response model without a separate ADR.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Protocol response parsers still need to produce `ChatResponse` for compatibility. | High | Existing public API and ADR-0008 keep legacy payload shape stable. | If `ChatResponse` changes soon, this lane should split into an ADR-backed public model migration. |
| Direct conversion of all response parser output into `GenerateTextContentPart` is lossy today. | High | `response_compat_projection.rs` rejects images, audio, approval parts, URL-backed files, and tool results without input. | If a parser path is lossless, migrate that path earlier and record it as a proof. |
| OpenAI Responses is the best first proof. | Medium | It has the richest response parser and the largest response-owned construction hotspot. | If scope is too large, choose Anthropic parse as the first smaller adapter proof. |
| Adapter extraction can be behavior-preserving. | High | Request-side bridge and response projection extractions already landed without serialized-shape changes. | If extraction changes fixture output, revert the design of that slice before touching more providers. |
| A new public output model would require ADR-level approval. | Medium | ADR-0008 only authorizes compatibility adapters and directional existing surfaces. | If implementation proves a missing model is required, pause and write an ADR before broad changes. |

## Architecture Direction

Use a two-step boundary:

1. **Parser-local response compatibility adapters.** A provider/protocol parser may still emit
   legacy `ContentPart`, but only through a named response adapter module such as
   `legacy_response_content`, `response_content`, or provider-specific equivalent. These adapters
   own empty request-option defaults and response metadata namespacing.
2. **Lossless generated-output projection.** Only response subsets that can preserve all provider
   data should be projected into `GenerateTextContentPart` or related generated-output carriers.
   Lossy subsets remain compatibility payloads until a better response output model exists.

This mirrors the Vercel-aligned split without pretending the current stable `ChatResponse` shape is
already the final generated-output model.

## Closeout Condition

This lane can close when:

- at least one high-value response parser has a named response compatibility adapter with no
  behavior change;
- at least one second provider/protocol path proves the pattern is not OpenAI-specific or is
  explicitly deferred with evidence;
- source guards cover the migrated response adapters and reject non-empty request option emission;
- final gates pass for the touched protocol/provider crates;
- remaining parser-wide generated-output work is either split into precise follow-ons or documented
  as intentionally deferred due to lossiness.
