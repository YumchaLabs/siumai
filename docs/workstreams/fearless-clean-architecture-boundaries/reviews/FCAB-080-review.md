# FCAB-080 Review — ContentPart Adapter Boundaries

Date: 2026-05-21

## Workstream Compliance

- Blocking: none.
- Important: none.
- Minor: none.

FCAB-080 satisfies the task contract by moving high-value production direct `ContentPart`
construction behind named response adapters where response parsers still need the legacy carrier:

- `siumai-protocol-openai/src/standards/openai/compat/response_content.rs`
- `siumai-protocol-anthropic/src/standards/anthropic/streaming/response_content.rs`
- `siumai-provider-cohere/src/standards/cohere/response_content.rs`
- `siumai-provider-ollama/src/standards/ollama/response_content.rs`

The task stayed inside the directional-content seam. It did not mix in provider-utils movement,
bridge-target ownership, or facade finalization.

## Code Quality

- Blocking: none.
- Important: none.
- Minor: none.

The new modules follow the existing response-adapter pattern already used by OpenAI Responses,
Anthropic parsing, Gemini response conversion, Google Interactions, and Bedrock. They keep the
unavoidable legacy defaults local and make response metadata preservation explicit.

`siumai-provider-cohere::standards::cohere::shared` no longer carries an unused generic
`message_content_from_parts` helper after Cohere response construction moved to the response adapter.

## Missing Gates

None for the FCAB-080 claim. Fresh evidence is recorded for:

- touched package compile checks;
- adapter source guards;
- behavior tests for OpenAI-compatible tool calls, annotations, and Perplexity citations;
- Cohere citation metadata;
- Ollama tool-call finish reason;
- Anthropic streaming source guards;
- formatting and whitespace checks.

## Residual Risk

- Request serializers still pattern-match legacy prompt carriers by necessity. They remain
  classified as request-side adapters and should be revisited only when a lossless prompt-directional
  replacement exists.
- `siumai-core::streaming::processor` still constructs final legacy content while aggregating stream
  parts. That belongs to the provider-utils/core isolation work in FCAB-090/FCAB-100.
- `siumai-bridge` still owns target response/stream inspection and protocol target adapters.
  FCAB-110 owns that movement after provider/protocol content adapters are stable.

## Verdict

FCAB-080 is ready to mark complete. Continue with FCAB-090 and keep the next slice focused on
provider-utils ownership.
