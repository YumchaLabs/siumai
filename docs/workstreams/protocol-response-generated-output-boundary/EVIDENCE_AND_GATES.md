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
