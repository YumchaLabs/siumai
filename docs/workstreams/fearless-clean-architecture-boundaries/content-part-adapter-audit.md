# FCAB-080 ContentPart Adapter Audit

Date: 2026-05-21

## Scope

FCAB-080 audited production direct `ContentPart` construction in:

- `siumai-protocol-*`
- `siumai-provider-*`
- `siumai-bridge/src/response`
- `siumai-bridge/src/stream`

The goal was not to delete the legacy carrier outright. ADR-0008 still keeps `ContentPart` as the
serde-facing compatibility carrier. The FCAB-080 rule is narrower: production response parsing that
must create legacy content should do so through a named request/response adapter, and request-side
serializers should not read response metadata unless they are an explicitly documented replay seam.

## Moved Behind Named Response Adapters

| Area | New adapter | Previous direct construction moved |
| --- | --- | --- |
| OpenAI-compatible chat responses | `siumai-protocol-openai/src/standards/openai/compat/response_content.rs` | Text parts from array content, tool calls, annotation/perplexity source parts, and reasoning parts in `compat/transformers.rs`. |
| Anthropic streaming final response aggregation | `siumai-protocol-anthropic/src/standards/anthropic/streaming/response_content.rs` | Final stream `Text` and `Reasoning` content parts created from accumulated text/thinking blocks. |
| Cohere non-stream chat responses | `siumai-provider-cohere/src/standards/cohere/response_content.rs` | Text, thinking, citation source, and tool-call parts in the non-stream response transformer. |
| Ollama response message conversion | `siumai-provider-ollama/src/standards/ollama/response_content.rs` | Text, image, tool-call, and thinking parts in `convert_from_ollama_message`. |

The new adapters own the unavoidable compatibility defaults:

- `provider_options: ProviderOptionsMap::default()` for legacy content parts that still carry a
  request-side option bag in their public shape;
- response-side `provider_metadata` preservation where the provider response supplies it;
- final `MessageContent` text-vs-multimodal collapse rules.

## Existing Adapters Kept

The scan found existing response adapters that already satisfy the FCAB-080 boundary and were left
in place:

- `siumai-protocol-anthropic/src/standards/anthropic/utils/parse/response_content.rs`
- `siumai-protocol-gemini/src/standards/gemini/transformers/response/response_content.rs`
- `siumai-provider-gemini/src/providers/gemini/interactions/response/response_content.rs`
- `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/response_content.rs`

## Audited Non-Constructor Or Explicit Request Paths

- `siumai-bridge/src/response/inspect.rs` pattern-matches existing response content for lossy
  bridge inspection and capability reports. It does not construct production `ContentPart` values.
- `siumai-bridge/src/stream/*` currently constructs stream parts or uses test fixtures; it does not
  own production response-content `ContentPart` construction in the audited slice.
- `siumai-protocol-openai/src/standards/openai/utils.rs`,
  `siumai-protocol-openai/src/standards/openai/compat/completion.rs`,
  `siumai-protocol-gemini/src/standards/gemini/convert.rs`,
  `siumai-protocol-gemini/src/standards/gemini/request_bridge.rs`,
  `siumai-provider-cohere/src/standards/cohere/chat.rs` request conversion sections, and
  `siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs` request conversion sections are
  request serializers. They pattern-match prompt-side legacy carriers while mapping user/model
  messages to provider wire shapes.
- `siumai-protocol-openai/src/standards/openai/transformers/response/responses/response_content.rs`
  is the named OpenAI Responses legacy response-content adapter. It centralizes unavoidable
  response-side `ContentPart` construction and default empty request-option bags for Responses API
  items.
- `siumai-provider-gemini/src/providers/gemini/interactions/request.rs` remains the documented
  Google Interactions replay exception: it needs explicit signature/id metadata to replay provider
  interaction state.
- Provider metadata extension modules pattern-match existing parts to expose typed metadata views;
  those modules do not create response content.

## Source Guards Added Or Refreshed

- `openai_compatible_chat_response_delegates_legacy_construction_to_adapter`
- `openai_compatible_chat_response_source_does_not_emit_request_provider_options`
- `anthropic_streaming_response_content_delegates_legacy_construction_to_adapter`
- `anthropic_streaming_response_content_adapter_keeps_request_options_defaulted`
- `cohere_response_content_delegates_legacy_construction_to_adapter`
- `cohere_response_content_adapter_keeps_request_options_defaulted`
- `ollama_response_content_delegates_legacy_construction_to_adapter`
- `ollama_response_content_adapter_keeps_request_options_defaulted`

These guards prove the moved production response sections call the named `response_content`
adapters and do not own legacy request-option defaults directly.

## Residual Work

The remaining direct `ContentPart` appearances are intentionally outside this FCAB-080 slice:

- request serializer pattern matches and synthetic prompt fallback construction;
- typed provider metadata views;
- test fixtures and public-surface compatibility assertions;
- core streaming aggregation, which belongs to the provider-utils/core isolation decision in
  FCAB-090/FCAB-100 rather than provider/protocol response parsing;
- bridge target wire parsing ownership, which belongs to FCAB-110 after content adapters are stable.

No separate follow-on workstream is required solely for FCAB-080. The residual work is already
covered by FCAB-090, FCAB-100, FCAB-110, and FCAB-120.
