# Protocol Response Generated-Output Boundary — Evidence And Gates

Status: Active
Last updated: 2026-05-19

## Evidence Log

### 2026-05-19 — PRG-010 planning scan

Commands used:

```text
rg -n "ContentPart::|MessageContent::MultiModal|provider_metadata:|provider_options:" \
  siumai-bridge/src/response siumai-bridge/src/stream \
  siumai-protocol-openai/src/standards/openai \
  siumai-protocol-anthropic/src/standards/anthropic \
  siumai-protocol-gemini/src/standards/gemini \
  siumai-provider-gemini/src/providers/gemini/interactions \
  siumai-provider-amazon-bedrock/src/standards/bedrock -g "*.rs"
```

The scan found that response/parser construction remains concentrated in a few large files rather
than uniformly across the repo. Highest-value follow-up targets:

| Target | Why it matters | First action |
| --- | --- | --- |
| OpenAI Responses response transformer | Richest response parser; constructs hosted tools, tool approval, reasoning, sources, files, custom payloads, and metadata. | Extract named response compatibility adapter first. |
| Anthropic parse utilities | Citations/sources and tool-use response content are good second-provider proof. | Apply adapter pattern or document Anthropic-specific adapter shape. |
| Gemini response transformer | Already source-guarded; useful to verify pattern does not overfit OpenAI. | Audit before editing; no-op classification is acceptable. |
| Bridge response/stream paths | They serialize/inspect provider-native response payloads and should not own provider semantics. | Decide ownership after parser proof. |
| Bedrock chat parser | Mixed request/response implementation; high constructor count but risky broad scope. | Split narrowly before touching. |

### 2026-05-19 — Existing shipped seam

`siumai-spec/src/types/ai_sdk/response_compat_projection.rs` is the existing response-side
compatibility seam for projecting legacy `ContentPart` to `GenerateTextContentPart`.

Important behavior:

- preserves response `provider_metadata`;
- ignores request `provider_options`;
- rejects ambiguous or lossy legacy carriers:
  - image/audio content,
  - URL-backed files that cannot become `GeneratedFile`,
  - tool approval parts without surrounding tool-call context,
  - tool results without original input.

This means protocol parsers should not be forced through that projection until their output subset
is proven lossless.

### 2026-05-19 — PRG-020 OpenAI Responses response adapter extraction

Changed files:

- `siumai-protocol-openai/src/standards/openai/transformers/response/responses.rs`
- `siumai-protocol-openai/src/standards/openai/transformers/response/responses/response_content.rs`

Implementation evidence:

- Added parser-local `response_content` adapter module.
- Moved OpenAI Responses response-owned legacy `ContentPart` construction behind
  `response_content::*` helpers.
- Kept `ChatResponse.content` shape behavior-preserving; existing OpenAI Responses behavior tests
  still pass.
- Added source guards:
  - broad transformer delegates direct legacy content construction to `response_content`;
  - `response_content` owns empty `ProviderOptionsMap::default()` initialization and does not read
    request provider option maps.

Fresh verification:

```text
cargo fmt --check -p siumai-protocol-openai
```

Result: PASS. Formatting is clean for the touched package.

```text
cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses responses_response_transformer_source_does_not_emit_request_provider_options --no-fail-fast
```

Result: PASS. 1 test passed. This verifies the required PRG-020 directionality guard.

```text
cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses standards::openai::transformers::response::responses::tests --no-fail-fast
```

Result: PASS. 19 tests passed. This covers the touched OpenAI Responses transformer behavior tests,
including hosted tools, custom tool calls, text metadata, reasoning metadata, source metadata, xAI
usage, and the new adapter source guards.

```text
cargo check -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses
```

Result: PASS. Touched package compiles under the PRG-020 feature set.

Broader gates not run:

- `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast`:
  skipped for PRG-020 because no spec projection code changed. Keep this gate for PRG-030.
- Full `siumai-protocol-openai` package test matrix: skipped because PRG-020 changed only the
  OpenAI Responses response transformer; targeted transformer tests were run.

### 2026-05-19 — PRG-030 OpenAI Responses generated-output lossiness classification

Classification rule:

- lossless means the current OpenAI Responses `ChatResponse.content` compatibility payload can be
  projected through spec-owned `GenerateTextContentPart` helpers without dropping required response
  semantics;
- legacy-only means the shape must remain a response compatibility payload unless a future,
  shape-specific adapter adds the missing context.

Lossiness matrix:

| OpenAI Responses output shape | Current compatibility payload | Generated-output projection status | Reason |
| --- | --- | --- | --- |
| `message.content[].output_text` / text annotations | text plus optional source compatibility parts | Lossless subset | text and source metadata are response-side and preserved by spec projection. |
| `reasoning` summaries / encrypted reasoning metadata | reasoning compatibility part | Lossless subset | reasoning text and provider metadata project without request options. |
| `compaction` | custom compatibility part | Lossless subset | custom kind plus provider metadata project as generated custom output. |
| `function_call` | tool-call compatibility part | Lossless subset | tool-call id, name, arguments, and provider metadata are complete. |
| provider-executed calls with paired input/result, including `custom_tool_call`, MCP call results, file/web/code/computer/image-generation call paths with parser-owned input | tool-call plus tool-result compatibility parts | Lossless subset for content projection | tool-result carries original input, so spec projection can build generated tool output. |
| `mcp_approval_request` | tool-call plus tool-approval-request compatibility parts | Legacy-only | approval output requires surrounding original tool-call/approval workflow context and is intentionally rejected by spec projection. |
| output-only hosted tool results: `tool_search_output`, `local_shell_call_output`, `shell_call_output`, `apply_patch_call_output` when no original input is present | tool-result compatibility part with `input: None` | Legacy-only | generated tool-result output requires original input; projection rejects these instead of inventing it. |
| direct image/audio or URL-backed generated files if a future Responses parser emits them as legacy content parts | image/audio/file compatibility payloads | Legacy-only unless binary/base64 file data is available | spec projection rejects ambiguous image/audio and URL-only files to avoid lossy generated-output claims. |
| response-level sources, raw response body, and provider metadata outside individual content parts | `ChatResponse` metadata/raw payload | Keep response payload | content-part projection is not a replacement for full response semantics. |

Implementation evidence:

- Added `responses_response_transformer_does_not_force_generated_output_projection`, a source guard
  proving production OpenAI Responses parsing and its response-content adapter do not call
  `GenerateTextContentPart` projection helpers directly.
- Added a positive projection fixture for a lossless text + reasoning + function-call subset.
- Added negative projection fixtures for `mcp_approval_request` and output-only hosted tool result
  payloads.
- Left `siumai-spec` projection behavior unchanged; its boundary tests remain the authority for
  rejecting ambiguous legacy carriers.

Fresh verification:

```text
cargo fmt --check -p siumai-protocol-openai
```

Result: PASS. Formatting is clean for the touched OpenAI package.

```text
cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast
```

Result: PASS. 8 tests passed. This verifies the spec projection boundary still preserves response
metadata, ignores request provider options, and rejects ambiguous legacy carriers.

```text
cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses responses_generated_output_projection --no-fail-fast
```

Result: PASS. 3 tests passed. This verifies representative OpenAI Responses lossless and lossy
projection paths.

```text
cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses standards::openai::transformers::response::responses::tests --no-fail-fast
```

Result: PASS. 23 tests passed. This verifies the full OpenAI Responses transformer test module,
including PRG-020 adapter guards and PRG-030 projection-boundary guards.

Broader gates not run:

- Full `siumai-protocol-openai` package matrix: skipped because PRG-030 changed only OpenAI
  Responses transformer tests/docs and did not change production projection logic.
- Anthropic/Gemini/bridge gates: intentionally deferred to PRG-040, PRG-050, and PRG-060.

### 2026-05-19 — PRG-040 Anthropic response adapter proof

Changed files:

- `siumai-protocol-anthropic/src/standards/anthropic/utils/parse.rs`
- `siumai-protocol-anthropic/src/standards/anthropic/utils/parse/response_content.rs`

Implementation evidence:

- Added parser-local `response_content` adapter module for Anthropic response compatibility
  payloads.
- Moved Anthropic response-owned legacy `ContentPart` and `ToolResultContentPart` construction
  behind named helpers:
  - text and thinking/reasoning parts;
  - URL/document source parts for citations and web-search result attribution;
  - user tool, server tool, MCP tool call parts;
  - provider-executed tool result parts.
- Kept Anthropic-specific metadata response-side:
  - text block citations remain under `provider_metadata["anthropic"]["citations"]`;
  - citation/source metadata stays on source parts and the separate `AnthropicSource` list;
  - server tool names, callers, and MCP server names remain Anthropic provider metadata;
  - request-side `provider_options` are initialized only as empty compatibility defaults.
- Added source guards:
  - parser code delegates legacy content construction to `response_content`;
  - parser and adapter do not emit/read request provider options beyond empty defaults;
  - parser and adapter do not call generated-output projection helpers directly.

Fresh verification:

```text
cargo fmt --check -p siumai-protocol-anthropic
```

Result: PASS. Formatting is clean for the touched Anthropic package.

```text
cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_parse_response_content_source_does_not_emit_request_provider_options --no-fail-fast
```

Result: PASS. 1 test passed. This verifies the PRG-040 directionality guard.

```text
cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_parse_response_content --no-fail-fast
```

Result: PASS. 3 source-guard tests passed. This verifies delegation to the adapter, request option
hygiene, and no forced generated-output projection.

```text
cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard standards::anthropic::utils::parse::tests --no-fail-fast
```

Result: PASS. 16 tests passed. This covers the touched Anthropic response parser behavior tests,
including citations, document sources, web-search source attribution, server tools, MCP metadata,
tool-result normalization, dynamic code execution marking, and usage parsing.

```text
cargo check -p siumai-protocol-anthropic --no-default-features --features anthropic-standard
```

Result: PASS. Touched Anthropic package compiles under the PRG-040 feature set.

Broader gates not run:

- Full `siumai-protocol-anthropic` package matrix: skipped because PRG-040 changed only the
  Anthropic parse utility response-content construction seam; targeted parse tests and package
  check were run.
- Gemini and bridge gates: intentionally deferred to PRG-050 and PRG-060.

### 2026-05-19 — PRG-050 Gemini response adapter proof

Changed files:

- `siumai-protocol-gemini/src/standards/gemini/transformers/mod.rs`
- `siumai-protocol-gemini/src/standards/gemini/transformers/response.rs`
- `siumai-protocol-gemini/src/standards/gemini/transformers/response/response_content.rs`

Implementation evidence:

- Audited Gemini response parsing and found it was not a no-op candidate: `transform_chat_response`
  directly constructed text, reasoning, reasoning-file, image, audio, file, tool-call, and
  tool-result legacy compatibility parts.
- Added parser-local `response_content` adapter module for Gemini response compatibility payloads.
- Moved Gemini response-owned legacy `ContentPart` construction behind named helpers while keeping
  response semantics unchanged:
  - text and thought reasoning parts;
  - base64/URL reasoning files, images, audio, and files;
  - function-call and provider-executed code-execution tool parts;
  - text-vs-multimodal final `MessageContent` selection.
- Kept Gemini-specific metadata response-side:
  - thought signatures stay in provider metadata for individual response parts;
  - grounding metadata, URL context metadata, safety ratings, usage metadata, logprobs, sources,
    service tier, and finish message remain on `ChatResponse.provider_metadata`;
  - request-side `provider_options` are initialized only as empty compatibility defaults.
- Added/strengthened source guards:
  - parser code delegates legacy content construction to `response_content`;
  - parser and adapter do not emit/read request provider options beyond empty defaults;
  - parser and adapter do not call generated-output projection helpers directly.

Fresh verification:

```text
cargo fmt --check -p siumai-protocol-gemini
```

Result: PASS. Formatting is clean for the touched Gemini package.

```text
cargo nextest run -p siumai-protocol-gemini --no-default-features --features google gemini_response_content_source_does_not_emit_request_provider_options --no-fail-fast
```

Result: PASS. 1 test passed. This verifies the PRG-050 directionality guard.

```text
cargo nextest run -p siumai-protocol-gemini --no-default-features --features google gemini_response_content --no-fail-fast
```

Result: PASS. 3 source-guard tests passed. This verifies delegation to the adapter, request option
hygiene, and no forced generated-output projection.

```text
cargo nextest run -p siumai-protocol-gemini --no-default-features --features google standards::gemini::transformers::response --no-fail-fast
```

Result: PASS. 15 tests passed. This covers the touched Gemini response transformer behavior tests,
including grounding/url context metadata, safety metadata, raw finish reasons, logprobs, usage,
custom generated IDs, function-call finish semantics, and thought reasoning files.

```text
cargo check -p siumai-protocol-gemini --no-default-features --features google
```

Result: PASS. Touched Gemini package compiles under the PRG-050 feature set.

Broader gates not run:

- Full `siumai-protocol-gemini` package matrix: skipped because PRG-050 changed only the Gemini
  response transformer seam; targeted response tests and package check were run.
- Bridge gate: intentionally deferred to PRG-060.

### 2026-05-19 — PRG-060 Bridge response/stream ownership decision

Changed files:

- `siumai-bridge/src/response/tests.rs`

Bridge audit:

| Area | Current responsibility | Ownership decision |
| --- | --- | --- |
| `siumai-bridge/src/response/serialize.rs` | Clones/remaps a normalized `ChatResponse`, runs bridge hooks, inspects target lossiness, and dispatches JSON bytes/value encoding. | Keep as orchestration only; it must not own provider response parsing semantics. |
| `siumai-bridge/src/response/inspect.rs` and `target_caps.rs` | Compares response content, usage, finish reasons, and provider metadata against target capabilities, recording carried/lossy/dropped fields. | Keep target-loss accounting in bridge; provider metadata meaning remains provider/protocol-owned. |
| `siumai-bridge/src/stream/serialize.rs`, `inspect.rs`, and `profile.rs` | Normalizes terminal stream events, applies bridge hooks/remappers, marks cross-protocol streams lossy, and chooses target SSE converters. | Keep primitive stream-event serialization; do not introduce response-parser adapters here. |
| `siumai-bridge/src/stream/openai_responses_parts_bridge.rs` | Upgrades legacy/custom stream-part carriers into stable `ChatStreamPart` events and attaches OpenAI Responses replay raw items for provider-executed tool events. | Keep as a narrow stream replay adapter for gateway/proxy use-cases; split a follow-on before broadening provider semantics. |
| `siumai-bridge/src/target_dispatch.rs` | Calls protocol-owned request transformers, JSON response converters, and SSE event converters. | This is the correct bridge/protocol boundary: bridge delegates wire shapes to protocol crates. |

Decision:

- Do **not** wire bridge response/stream paths to parser-local protocol `response_content` adapters.
  Those adapters are parser-owned compatibility constructors for inbound provider responses, not a
  public bridge serialization API.
- Do **not** call `GenerateTextContentPart` or spec projection helpers from bridge response/stream
  serialization. Generated-output projection remains behind the spec-owned lossiness boundary.
- Keep bridge response/stream paths primitive-only at their public boundary:
  `ChatResponse` / `ChatStreamEvent` in, target JSON/SSE bytes or values out, with explicit
  `BridgeReport` loss accounting.
- Treat `OpenAiResponsesStreamPartsBridge` as a narrow replay shim for cross-protocol streaming,
  not as the canonical owner of OpenAI Responses provider semantics. If additional gateway/proxy
  JSON encoders need richer provider semantics, split a dedicated follow-on instead of expanding
  PRG-060.

Implementation evidence:

- Reused the existing source guard that proves response/stream bridge code does not emit
  request-side `provider_options` / `providerOptions`.
- Added source guards proving response/stream/dispatch bridge code:
  - does not call generated-output projection helpers directly;
  - does not depend on parser-local `response_content` adapters.

Fresh verification:

```text
cargo fmt --check -p siumai-bridge
```

Result: PASS. Formatting is clean for the touched bridge package.

```text
cargo nextest run -p siumai-bridge --features openai,anthropic,google response_and_stream_bridge_sources --no-fail-fast
```

Result: PASS. 3 tests passed. This verifies the bridge ownership source guards: no request
provider-options emission in response/stream scope, no direct generated-output projection, and no
parser-local response adapter coupling.

```text
cargo nextest run -p siumai-bridge --features openai,anthropic,google response --no-fail-fast
```

Result: PASS. 55 tests passed. This is the required PRG-060 bridge gate and covers response bridge
behavior plus stream replay tests selected by the `response` filter.

Broader gates not run:

- Full `siumai-bridge` package matrix: skipped because PRG-060 changed only bridge ownership source
  guards and workstream evidence; the required response gate plus targeted guard gate passed.
- Protocol parser gates: skipped because PRG-060 did not change protocol parser code.

### 2026-05-19 — PRG-070 Provider-owned response parser adapters

Changed files:

- `siumai-provider-gemini/src/providers/gemini/interactions/response.rs`
- `siumai-provider-gemini/src/providers/gemini/interactions/response/response_content.rs`
- `siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs`
- `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/response_content.rs`
- `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/streaming.rs`
- `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/tests.rs`

Provider-owned parser audit:

| Target | Classification | Decision |
| --- | --- | --- |
| `siumai-provider-gemini/src/providers/gemini/interactions/response.rs` | Provider-owned Interactions response parser, separate from the pure `siumai-protocol-gemini` GenerateContent parser. | Adopt a local `response_content` adapter. This file owns Interactions-specific step parsing, IDs, signatures, built-in tool calls/results, and citation/source compatibility payloads. |
| `siumai-provider-gemini/src/providers/gemini/interactions/stream.rs` | Streaming Interactions converter that emits stable runtime stream parts and reuses response source helpers for source events/final metadata. | No response `ContentPart` adapter extraction needed in stream production code; it already emits `ChatStreamPart` primitives and only consumes source compatibility parts from the response parser helpers. |
| `siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs` response transformer | Mixed request/response file, but response transformer is a narrow section after request conversion. | Adopt a local `response_content` adapter for response-side text/reasoning/tool-call/final content construction while leaving request conversion untouched. |
| `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/streaming.rs` final response aggregation | Stream converter owns final `ChatResponse` reconstruction from accumulated Bedrock stream blocks. | Reuse the Bedrock local `response_content` adapter only for final-response compatibility construction; stream event emission remains stable `ChatStreamPart` primitives. |

Implementation evidence:

- Added provider-local response compatibility adapter modules:
  - Google Interactions `response/response_content.rs`;
  - Bedrock Chat `chat/response_content.rs`.
- Moved response-side legacy `ContentPart` constructors behind those adapters for:
  - Google Interactions text, image-as-file, reasoning, function/tool calls, provider-executed
    tool results, and sources;
  - Bedrock text, reasoning, tool calls, and final `MessageContent` assembly in both non-stream and
    stream final-response paths.
- Added/strengthened source guards:
  - Google Interactions parser delegates legacy construction to `response_content`;
  - Google Interactions parser/adapter do not read or emit request provider options beyond empty
    compatibility defaults;
  - Google Interactions parser/adapter do not call generated-output projection helpers directly;
  - Bedrock response/stream source delegates response construction to `response_content`;
  - Bedrock response/stream/adapter do not read request provider option maps or force
    generated-output projection.

Fresh verification:

```text
cargo fmt --check -p siumai-provider-gemini -p siumai-provider-amazon-bedrock
```

Result: PASS. Formatting is clean for the touched provider packages.

```text
cargo nextest run -p siumai-provider-gemini --features google google_interactions_response --no-fail-fast
```

Result: PASS. 6 tests passed. This covers Google Interactions response behavior plus adapter
source guards.

```text
cargo nextest run -p siumai-provider-gemini --features google --no-fail-fast
```

Result: PASS. 105 tests passed. This verifies the full Gemini provider package under the `google`
feature after the provider-owned response adapter extraction.

```text
cargo nextest run -p siumai-provider-amazon-bedrock --features bedrock response_and_stream_source --no-fail-fast
```

Result: PASS. 3 tests passed. This verifies Bedrock response/stream source guards for adapter
delegation, request-provider-options hygiene, and no direct generated-output projection.

```text
cargo nextest run -p siumai-provider-amazon-bedrock --features bedrock --no-fail-fast
```

Result: PASS. 76 tests passed. This verifies the full Bedrock provider package under the `bedrock`
feature after the response/stream adapter extraction.

Broader gates not run:

- Workspace-wide nextest: skipped because PRG-070 touched only two provider packages, and both full
  feature gates passed.
- Protocol parser gates: skipped because protocol parser code was not changed by PRG-070.

### 2026-05-19 — PRG-080 Public architecture and migration docs

Changed files:

- `docs/architecture/public-surface.md`
- `docs/migration/migration-0.11.0-beta.7.md`
- `docs/workstreams/protocol-response-generated-output-boundary/TODO.md`
- `docs/workstreams/protocol-response-generated-output-boundary/WORKSTREAM.json`
- `docs/workstreams/protocol-response-generated-output-boundary/HANDOFF.md`
- `docs/workstreams/protocol-response-generated-output-boundary/JOURNAL/2026-05-19-prg-080.md`

Documentation boundary updates:

- Public surface docs now state that response parsers may still emit legacy `ChatResponse` /
  `MessageContent` payloads for compatibility, but that this does not make `ContentPart` the
  canonical response model.
- Migration docs now distinguish parser-local `response_content` modules from spec-owned
  generated-output projection helpers.
- Migration docs explicitly keep generated-output projection fallible and keep hosted tool results,
  approval requests, files, images, audio, and provider-specific metadata as compatibility payloads
  unless lossless projection is proven.
- No new public generated-output response model was proposed; future model work still requires an
  ADR-backed follow-on.

Fresh verification:

```text
python -c "import json, pathlib; json.loads(pathlib.Path('docs/workstreams/protocol-response-generated-output-boundary/WORKSTREAM.json').read_text(encoding='utf-8'))"
```

Result: PASS. Verifies workstream metadata remains valid JSON after PRG-080 updates.

```text
python -c "from pathlib import Path; arch=Path('docs/architecture/public-surface.md').read_text(encoding='utf-8'); mig=Path('docs/migration/migration-0.11.0-beta.7.md').read_text(encoding='utf-8'); assert 'Response parsing, compatibility payloads, and generated output' in arch; assert 'Parser-local response compatibility adapters' in mig; assert 'Spec-owned generated-output projection helpers' in mig; assert 'The named response-side adapter is' not in mig"
```

Result: PASS. Verifies the public docs contain the new boundary language and no longer call the
generated-output projection helper the named response-side adapter.

```text
git diff --check -- docs/architecture/public-surface.md docs/migration/migration-0.11.0-beta.7.md docs/workstreams/protocol-response-generated-output-boundary
```

Result: PASS. Verifies touched documentation has no whitespace errors. Git emitted local line-ending
conversion warnings for these existing text files on Windows, but reported no whitespace errors.

Broader gates not run:

- Rust crate tests: skipped because PRG-080 changed only documentation and workstream metadata.
- Workspace-wide nextest/fmt: skipped for the same reason; no Rust source files changed.

## Planned Gates

### PRG-010 — Workstream planning

Required:

```text
python -c "import json, pathlib; json.loads(pathlib.Path('docs/workstreams/protocol-response-generated-output-boundary/WORKSTREAM.json').read_text())"
git diff --check -- docs/workstreams/protocol-response-generated-output-boundary
```

### PRG-020 / PRG-030 — OpenAI Responses proof

Required:

```text
cargo fmt --check -p siumai-protocol-openai
cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses responses_response_transformer_source_does_not_emit_request_provider_options --no-fail-fast
cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast
```

Add targeted OpenAI Responses transformer fixture tests if the extraction touches behavior.

### PRG-040 — Anthropic proof

Required:

```text
cargo fmt --check -p siumai-protocol-anthropic
cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_parse_response_content_source_does_not_emit_request_provider_options --no-fail-fast
```

### PRG-050 — Gemini audit/proof

Required:

```text
cargo fmt --check -p siumai-protocol-gemini
cargo nextest run -p siumai-protocol-gemini --no-default-features --features google gemini_response_content_source_does_not_emit_request_provider_options --no-fail-fast
```

### PRG-060 — Bridge response/stream decision

Required:

```text
cargo nextest run -p siumai-bridge --features openai,anthropic,google response --no-fail-fast
```

### Final closeout

Required:

```text
cargo fmt --check -p siumai-protocol-openai -p siumai-protocol-anthropic -p siumai-protocol-gemini -p siumai-bridge
cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast
cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses --no-fail-fast
cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard --no-fail-fast
cargo nextest run -p siumai-protocol-gemini --no-default-features --features google --no-fail-fast
cargo nextest run -p siumai-bridge --features openai,anthropic,google response --no-fail-fast
```

Use narrower final gates if the lane intentionally closes after a smaller proof and splits the rest.
