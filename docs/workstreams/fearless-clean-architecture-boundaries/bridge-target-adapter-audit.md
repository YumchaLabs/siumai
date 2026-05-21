# FCAB-110 Bridge Target Adapter Audit

Date: 2026-05-21

## Scope

FCAB-110 audits target-specific request/response/stream adapter ownership in `siumai-bridge`.
The rule is:

- `siumai-bridge` keeps bridge reports, loss policy, lifecycle, hooks, customization, planning, and
  target dispatch;
- `siumai-protocol-*` owns target wire parsing/serialization when the adapter can depend only on
  protocol/core data shapes and not on bridge report policy;
- retained bridge shims must be thin and documented.

## Moved This Slice

| Target | Direction | New owner | Bridge role |
| --- | --- | --- | --- |
| Gemini GenerateContent | JSON request -> `ChatRequest` normalization | `siumai-protocol-gemini::standards::gemini::request_bridge` | `siumai-bridge/src/request/normalize/gemini_generate_content.rs` is now a compatibility shim that delegates to the protocol adapter. |

This was the cleanest first FCAB-110 move because the existing adapter already had a narrow file and
depended mostly on Gemini protocol types. Moving it to the protocol crate removes a clear
target-wire parser from the bridge without changing public bridge wrapper APIs.

## Retained For Later Or By Design

| Area | Current owner | Reason |
| --- | --- | --- |
| Generic request serialization dispatch | `siumai-bridge::target_dispatch` plus protocol transformers | The bridge chooses a target and applies bridge hooks/loss policy. The actual request transformers already live in protocol crates. |
| Response JSON serialization dispatch | `siumai-bridge::target_dispatch` plus protocol converters | Bridge inspection/reporting remains bridge-owned; protocol converters already own wire JSON shape. |
| Stream SSE serialization dispatch | `siumai-bridge::target_dispatch` plus protocol converters | Bridge decides target route and policy; protocol crates own event converters. |
| OpenAI Responses <-> Anthropic Messages direct request pairs | `siumai-bridge::request::pairs` | These are cross-protocol bridge policies, not a single protocol's wire parser. They intentionally emit loss warnings and apply bridge-specific approximations. |
| OpenAI/Anthropic request JSON normalization | `siumai-bridge::request::normalize` | They are still intertwined with bridge direct-pair replay semantics and bridge warnings. Move them only after extracting protocol-local parsers that do not depend on bridge policy. |
| OpenAI Responses stream parts bridge | `siumai-bridge::stream::openai_responses_parts_bridge` | This maps older Siumai stream parts into bridge-target runtime parts and owns bridge compatibility policy rather than a pure protocol parser. |

## Guards

FCAB-110 refreshed the existing source guards:

- `gemini_generate_content_request_normalization_is_protocol_adapter_backed`
- `gemini_request_normalization_source_uses_provider_options_for_thought_signature`

They now prove that:

- `normalize.rs` keeps only wrapper delegation for Gemini GenerateContent request normalization;
- the bridge shim delegates to `siumai-protocol-gemini`;
- the protocol adapter owns `GeminiGenerateContentRequest`, `parse_gemini_content`, and
  `parse_gemini_tools`;
- request-side Gemini `thoughtSignature` remains request `providerOptions`, not response
  `providerMetadata`.

## Follow-up

Further FCAB-110 work should target one adapter at a time. Candidate next moves are protocol-local
OpenAI Responses request normalization or Anthropic Messages request normalization, but only if
their bridge replay/loss-warning responsibilities can stay out of the protocol crate.
