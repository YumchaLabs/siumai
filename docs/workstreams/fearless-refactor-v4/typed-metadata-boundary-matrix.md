# Typed Metadata Boundary Matrix

This document records the current V4 boundary for provider-owned typed metadata.

The goal is to keep the public surface Rust-first and provider-owned while avoiding premature typing
for unstable event payloads or protocol-specific escape hatches that are not yet stable enough.

Cross-provider reasoning / thinking extraction semantics are tracked separately in
`reasoning-alignment.md` so this matrix stays focused on typed metadata boundaries.

## Rules

- Promote fields to typed metadata only when they are already a stable response-side contract.
- Keep event-level SSE payload assertions raw when the contract being validated is event shape,
  provider-key rewriting, or namespace-specific streaming behavior.
- Reuse implementation details across providers when helpful, but keep provider-owned naming on the
  public API.
- Do not invent source/content-part typed metadata for providers whose underlying protocol surface
  has not stabilized that shape yet.

## Matrix

| Provider | `ChatResponse` / `StreamEnd` typed | `ContentPart` typed | `Source` typed | Raw event boundary kept | Notes |
| --- | --- | --- | --- | --- | --- |
| Anthropic | Yes | Yes (`ToolCall.caller`) | Yes | Yes | `AnthropicMetadata.sources` already exposes typed `AnthropicSource` values directly, and top-level builder/provider/config-first guards now also verify `AnthropicMetadata.container` on both 200-response and `StreamEnd`; no extra `SourceExt` helper is needed because the source shape itself is already first-class. |
| OpenAI | Yes | Yes (`itemId`, `reasoningEncryptedContent`) | Yes (`fileId`, `containerId`, `index`) | Yes | SSE finish/reasoning/text-start payloads remain raw; top-level builder/provider/config-first guards now also verify Responses 200-response / `StreamEnd` source extraction through `OpenAiSourceExt` and Chat Completions 200-response / `StreamEnd` `logprobs` through `OpenAiChatResponseExt`, while event payload contracts stay raw. |
| Azure | No provider-owned response metadata promotion beyond OpenAI response transformer boundaries | No | No | Yes | Streaming checks stay raw because they validate namespace-key rewriting rather than a stable Azure metadata surface; top-level builder/provider/config-first Responses stream parity now explicitly locks `text-start` / `finish` custom payloads and final `provider_metadata["azure"]` roots on that raw boundary. |
| Google / Gemini | Yes | Yes (`thoughtSignature`) | No | Yes | Google/Gemini own the typed response/content-part story, and top-level builder/provider/config-first guards now verify typed `GeminiContentPartExt` / `GeminiChatResponseExt` extraction on 200-response and `StreamEnd` while keeping streaming `reasoning-start` `providerMetadata.google.thoughtSignature` payloads plus normalized `provider_metadata["google"]` roots on the raw event boundary. |
| Google Vertex | Yes | Yes (`thoughtSignature`) | Yes | Yes | Plain Vertex now exposes a provider-owned typed metadata facade under the `vertex` namespace only: `VertexMetadata` / `VertexChatResponseExt` cover response-level usage, safety, grounding, URL-context, logprobs, and typed `VertexSource` values, while `VertexContentPartExt` exposes part-level `thoughtSignature`; focused no-network public-path and registry guards now verify those typed reads on 200-response roots, while streaming `reasoning-start` / `reasoning-delta` payloads plus final namespace-root assertions remain raw. |
| xAI | Yes | No | Yes | N/A | `XaiSourceExt` keeps `xai` naming while accepting compatible envelopes underneath, and top-level builder/provider/config-first guards now also verify `XaiMetadata` on both 200-response and `StreamEnd`. |
| Groq | Yes | No | Yes | N/A | `GroqSourceExt` keeps `groq` naming while accepting compatible envelopes underneath, and top-level builder/provider/config-first guards now also verify `GroqMetadata` on both 200-response and `StreamEnd`. |
| DeepSeek | Yes | No | Yes | N/A | `DeepSeekSourceExt` keeps `deepseek` naming while accepting compatible envelopes underneath, and top-level builder/provider/config-first guards now also verify `DeepSeekMetadata.logprobs` on both 200-response and `StreamEnd`. |
| Perplexity | Yes (`citations`, `images`, `usage`) | No | No | N/A | Hosted-search `citations` are now promoted; top-level builder/provider/config-first guards also verify both 200-response and `StreamEnd` extraction with normalized `provider_metadata["perplexity"]` roots, while unknown vendor fields stay in `extra`. |
| OpenRouter | Yes (`logprobs`, `sources`) | Yes (`itemId`, `reasoningEncryptedContent`) | Yes (`fileId`, `containerId`, `index`) | N/A | OpenRouter now exposes an alias-based vendor-owned metadata view over the shared OpenAI-shaped compat payload: top-level builder/provider/config/registry guards verify both 200-response and `StreamEnd` extraction through `OpenRouterChatResponseExt`, `OpenRouterSourceExt`, and `OpenRouterContentPartExt` while keeping the public naming vendor-owned. |
| MiniMaxi | Yes | Yes (`ToolCall.caller`) | Yes | N/A | Reuses the typed Anthropic-shaped source struct under `MinimaxiMetadata.sources`, keeps `minimaxi` naming on the public surface, and top-level builder/provider/config-first guards now verify both typed metadata plus normalized `provider_metadata["minimaxi"]` keys on 200-response and `StreamEnd`; no separate `SourceExt` helper is needed. |
| Bedrock | Yes | No | No | N/A | Response-side metadata helper exists, and top-level builder/provider/config-first guards now verify `BedrockMetadata` on both 200-response and JSON-stream `StreamEnd`; raw response-map assertions are reduced rather than expanded. |
| Ollama | Yes | No | No | N/A | Response-side timing metadata is typed; request-side tool omission remains a separate transport-boundary concern. |

## Current Intentional Gaps

- Google Vertex custom streaming payloads remain raw and namespace-focused even though final response/content-part metadata is now typed under `provider_ext::google_vertex::metadata`.
- Azure custom streaming payloads remain raw and namespace-focused.
- Anthropic and MiniMaxi already expose typed source entries through
  `AnthropicMetadata.sources` / `MinimaxiMetadata.sources`; keep the boundary there unless a
  future provider-specific source payload grows beyond the current shared struct.

## Usage Guidance

- Prefer provider-owned typed helpers first (`provider_ext::<provider>::metadata::*`).
- Fall back to raw provider metadata only for:
  - event-level streaming contracts,
  - namespace rewrite assertions,
  - fields not yet promoted into a stable typed surface.
- When adding a new typed helper, update this matrix in the same change.
