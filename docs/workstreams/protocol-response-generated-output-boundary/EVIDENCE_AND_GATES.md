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
