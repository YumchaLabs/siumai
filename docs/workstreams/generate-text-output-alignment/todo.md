# Generate Text Output Alignment - TODO

Last updated: 2026-04-24

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Reference Audit

- [x] Audit `repo-ref/ai/packages/ai/src/generate-text/output.ts`.
- [x] Audit provider mapping for schema-less JSON response formats in OpenAI, OpenAI-compatible,
  Gemini, Anthropic, and Bedrock packages.
- [x] Confirm `choice()` is the generateText output name while `enum` remains the generateObject
  output strategy name.

## Track B - Core Response Format

- [x] Add `ResponseFormat::json_object()` for AI SDK `{ type: "json" }` without schema.
- [x] Preserve existing `ResponseFormat::json_schema(...)` serialization and deserialization.
- [x] Add public surface coverage for schema-less JSON response format.
- [x] Add AI SDK `parsePartialJson` / `fixJson` parity as Rust
  `fix_partial_json(...)`, `parse_partial_json(...)`, `PartialJsonParseState`, and
  `PartialJsonParseResult`.

## Track C - Provider Mapping

- [x] Map schema-less JSON to OpenAI Chat Completions `{ type: "json_object" }`.
- [x] Map schema-less JSON to OpenAI Responses `text.format = { type: "json_object" }`.
- [x] Map schema-less JSON to Gemini `responseMimeType = "application/json"` without schema.
- [x] Map schema-less JSON to Ollama `format = "json"`.
- [-] Anthropic schema-less JSON enforcement remains unsupported upstream and is not forced.
- [-] Bedrock schema-less JSON enforcement remains unsupported upstream and is not forced.

## Track D - Helper Surface

- [x] Add `structured_output::generate_json(...)`.
- [x] Add `structured_output::generate_choice(...)`.
- [x] Add passive AI SDK output-part shapes for basic `text`, `custom`, and generated `file`
  content parts without reusing prompt-side request carriers.
- [x] Add `GenerateTextContentPart` as the passive output-side union matching AI SDK
  `generate-text/content-part.ts`.
- [x] Add passive AI SDK result envelope structures for `ResponseMessage`, `StepResult`, and
  `GenerateTextResult` without changing the current runtime return type, including the split
  between step-side `providerOptions` reasoning and final result-side `providerMetadata` reasoning.
- [x] Add AI SDK export aliases for `StepResult`, `DefaultGeneratedFile`,
  `Experimental_GeneratedImage`, and static/dynamic/typed tool call/result/error views over the
  existing passive carriers.
- [x] Add passive AI SDK `TextStreamPart` output structures without replacing runtime
  `ChatStreamPart` provider V4 semantics.
- [x] Add passive AI SDK `LanguageModelStreamPart` model-call structures for
  `stream-language-model-call.ts`, including `model-call-start`, `model-call-response-metadata`,
  and `model-call-end`, without claiming the experimental runtime helper is implemented.
- [x] Add passive AI SDK callback/event payload structures for `GenerateTextStartEvent`,
  `GenerateTextStepStartEvent`, `GenerateTextEndEvent`, `StreamTextChunkEvent`,
  `ToolExecutionStartEvent`, `ToolExecutionEndEvent`, and `ToolOutput`.
- [x] Add passive AI SDK step-control and policy payloads for `StopCondition`, `filterActiveTools`, prepare-step
  options/results, tool approval status/configuration/context, and tool-call repair context/result.
- [x] Add Rust-named helper aliases for upstream `experimental_filterActiveTools` and deprecated
  `stepCountIs`.
- [x] Add `prune_messages(...)` over shared `ModelMessage` values for upstream `pruneMessages`
  reasoning/tool/empty-message pruning behavior.
- [x] Add passive AI SDK output-part shapes for `GeneratedFile`, `ReasoningOutput`, and
  `ReasoningFileOutput`.
- [x] Add passive AI SDK output-part shapes for `ToolError` and `ToolOutputDenied`.
- [x] Add passive AI SDK output-part shapes for `ToolApprovalRequestOutput` and
  `ToolApprovalResponseOutput`.
- [x] Re-export the new helpers from the root facade and `prelude::unified::*`.
- [x] Keep `generate_enum(...)` strict for generateObject enum parity.

## Track E - Deferred Streaming Output

- [x] Land the partial JSON parser foundation needed by future streaming output transforms.
- [x] Add a narrow `partial_json_value_stream(...)` projection over existing `ChatStream`.
- [-] Do not expose the full AI SDK `StreamTextResult` multi-lane result object until tee/backpressure
  semantics are designed.
- [-] Do not add an `Output` trait that claims streaming parity before Track E exists.
- [-] Do not expose `smoothStream`, `StreamTextTransform`, `UIMessageStreamOptions`, callback
  function aliases, or type-level infer helpers until there is real Rust runtime/type behavior
  behind them.
