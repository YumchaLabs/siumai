# FMD-080 — Legacy ContentPart Directional Boundary Decision

Date: 2026-05-19
Status: Accepted for this workstream
Task: FMD-080

## Decision

Do **not** open the breaking `ContentPart` namespace move yet.

Continue this lane with one non-breaking directional adapter proof slice in FMD-090: extract the
request-side legacy `ContentPart` adapter helpers used by bridge request normalization into a narrow
`siumai-bridge/src/request/legacy_content.rs` module, then route the selected JSON -> `ChatRequest`
normalization paths through that module and guard against direct request-side legacy construction
returning to `request/normalize.rs`.

A later child workstream should own the breaking move of legacy `ContentPart` under an explicit
compatibility namespace. Opening that child now would be premature because protocol response parsers
and public serde payloads still legitimately depend on the stable legacy carrier.

## Inputs Reviewed

- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- `docs/workstreams/fearless-content-part-boundary-split/`
- current `siumai-spec` prompt and generated-output projection code
- current `siumai-bridge/src/request/normalize.rs` request-side legacy adapter helpers
- current `siumai-core/src/streaming/processor.rs` response-side `response_text_part(...)` adapter
- current `siumai/src/text.rs` generate-text projection delegation to `siumai-spec`

## Why Not A Breaking Child Workstream Now

ADR-0008 classifies legacy `ContentPart` as compatibility-only, but explicitly defers moving it
under `compat` until directional request and response adapters cover the high-value paths and
migration docs can preserve serde/provider fixture parity.

The closed `fearless-content-part-boundary-split` workstream already made this same call after two
proof migrations:

- a request-side low-risk migration via `request_text_part(...)`, and
- a response-side low-risk migration via `response_text_part(...)`.

Current evidence still shows broad legitimate use of `ContentPart` in:

- protocol response transformers that populate stable `ChatResponse` content,
- bridge inspectors and serializers that preserve provider metadata/options through compatibility
  payloads,
- facade text projection fallback paths for legacy tool-result details, and
- public serde-facing `ChatMessage` / `ChatResponse` shapes.

Moving the enum now would convert an internal module-deepening lane into a broad public API break
before the replacement paths are deep enough.

## Target State For The Next Slice

### Request direction

Primary request code should prefer typed prompt/model-message parts:

- `UserContentPart`
- `AssistantContentPart`
- `ToolContentPart`
- AI SDK V4 prompt parts

When bridge request normalization must create the legacy `ContentPart` compatibility carrier, it
should do so through a named request adapter module. That adapter may populate request-side
`provider_options`, but must keep response-side `provider_metadata` empty.

FMD-090 should therefore extract request-side bridge helpers such as:

- `request_text_part(...)`
- `request_reasoning_part(...)`
- `request_image_part(...)`
- `request_audio_part(...)`
- `request_file_part(...)`
- `request_tool_call_part(...)`
- `request_tool_result_part(...)`

into `siumai-bridge/src/request/legacy_content.rs` and route selected OpenAI/Anthropic/Gemini
request normalization paths through that module.

### Response direction

Primary response projection should prefer generated-output/content types:

- `GenerateTextContentPart`
- output part carriers
- AI SDK V4 generated content parts
- stream output parts

Response parsers that still need to emit legacy `ContentPart` should use response-side adapter
helpers that may populate response-side `provider_metadata`, but must not emit non-empty request
`provider_options` except for explicitly documented legacy compatibility payloads.

FMD-090 should not attempt a broad response-parser rewrite. It should leave a follow-on note for a
future response-side adapter module once one concrete response parser seam is selected.

### Compatibility direction

Legacy `ContentPart` remains available at existing public paths during this lane. Direct production
construction remains acceptable only inside audited compatibility adapters, bridge/protocol legacy
payload seams, or tests. New public docs should continue to teach request/response-specific imports
instead of presenting `ContentPart` as the canonical content model.

## FMD-090 Proof Slice

Implement the request-side bridge adapter extraction first because it is:

- non-breaking,
- already supported by local helper functions,
- aligned with ADR-0008's adapter-first migration rule,
- directly testable with bridge request fixture gates, and
- a real module deepening improvement: deleting the adapter would force request provider-map safety
  rules back into several protocol JSON parsers.

Recommended FMD-090 gates:

```powershell
cargo fmt --check -p siumai-bridge
cargo nextest run -p siumai-bridge --features openai,anthropic,google request --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google `
  --test request_direct_bridge_fixtures_alignment_test --no-fail-fast
```

If `siumai-spec` projection helpers are touched, also run:

```powershell
cargo nextest run -p siumai-spec --no-default-features prompt --no-fail-fast
```

## Future Child Workstream Trigger

Open a child breaking-slice workstream only after:

1. bridge request normalization uses request-side adapters for all high-value legacy construction;
2. at least one protocol response transformer has a response-side adapter seam;
3. facade docs and migration notes consistently recommend prompt/generated-output parts;
4. source guards classify any remaining direct production construction; and
5. fixture parity proves serde, bridge, and provider protocol payloads are unchanged.

That child should decide whether to move `ContentPart` to `compat::*` directly or keep a deprecated
old-path re-export for one release cycle.
