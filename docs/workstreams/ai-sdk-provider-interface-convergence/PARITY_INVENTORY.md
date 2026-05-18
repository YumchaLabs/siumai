# AI SDK Provider Interface Convergence - Parity Inventory

Status: Active
Last updated: 2026-05-18

This inventory records the first program-level comparison between AI SDK package seams and Siumai
crate seams. It is intentionally a control surface, not a claim that every row is complete.

Status legend:

- `Green`: aligned enough for the current stable baseline.
- `Amber`: structurally close, but follow-up review or guard work remains.
- `Red`: materially misaligned; should drive a child workstream.
- `Deferred`: intentionally outside the current Rust runtime or package scope.

## Seam Inventory

| Seam | AI SDK reference | Siumai anchor | Status | Next action |
| --- | --- | --- | --- | --- |
| Provider interface | `packages/provider/src/*/v4` | `siumai-spec`, `siumai-core::{text,embedding,image,speech,transcription,rerank,completion,video}` | Amber | Audit whether stable family traits cover current provider package needs without routing through `LlmClient`. |
| Provider utilities | `packages/provider-utils/src` | `siumai-core::{utils,tooling,execution,streaming}`, selected facade helpers | Amber | Keep provider-agnostic helpers in core; reject provider-specific URL/model/default logic in core. |
| Root high-level helpers | `packages/ai/src` | `siumai` facade modules and `siumai-extras` orchestration | Amber | Keep runtime-heavy helpers honest; defer frontend/browser-only hooks. |
| Provider options | `shared/v4/shared-v4-provider-options.ts` | `ProviderOptionsMap`, provider-owned typed option modules | Green | Continue enforcing request-time-only semantics on new paths. |
| Provider metadata | `packages/ai/src/types/provider-metadata.ts` | `ProviderMetadataMap`, provider-owned typed metadata helpers | Green | Continue using provider-rooted response metadata. |
| Stream part union | `language-model-v4-stream-part.ts` | `ChatStreamEvent::Part`, `ChatStreamPart`, `LanguageModelV4StreamPart`, replay carriers | Green | AIPC-050/AIPC-060 closed the highest-risk stable stream-part test and bridge/gateway assertion gaps; keep custom events to compatibility/provider-native boundaries. |
| Registry/provider factory | `packages/ai/src/registry`, provider callable factories | `siumai-registry::ProviderFactory`, registry handles | Green | Keep AIPC-030/AIPC-040 guards green as new families and extension helpers land. |
| OpenAI-compatible adapter | `packages/openai-compatible` | `siumai-provider-openai-compatible`, `siumai-protocol-openai` | Green | Completion family exposure is explicit metadata, not inherited from chat transport; continue typed option/metadata parity review. |
| Protocol bridge/gateway | `packages/gateway`, protocol adapters | `siumai-bridge`, `siumai-extras::server`, protocol crates | Green | Bridge and extras helper tests cover stable provider-tool stream parts into OpenAI Responses output items; extras runtime gateway code imports bridge-owned stream adapters through `siumai_bridge::stream`. |
| Legacy content compatibility | V3/V4 prompt/content split | `ContentPart` compatibility boundary plus V4 prompt/content projections | Amber | Do not move `ContentPart` until ADR-0008 future-breaking conditions are met. |

## Provider Package Inventory

| AI SDK package | Siumai crate/surface | Current package boundary | Status | First follow-up |
| --- | --- | --- | --- | --- |
| `openai` | `siumai-provider-openai`, `siumai-protocol-openai`, `provider_ext::openai` | Native OpenAI plus Responses/Chat/Completion/Embedding/Files/Skills typed surfaces | Amber | Check remaining Responses stream/custom-tool replay and typed option surface drift. |
| `azure` | `siumai-provider-azure`, OpenAI protocol reuse | Azure OpenAI provider with OpenAI-family semantics | Amber | Keep Azure-specific metadata/options provider-owned and avoid generic OpenAI leakage. |
| `openai-compatible` | `siumai-provider-openai-compatible` | Shared OpenAI-compatible runtime plus vendor presets | Green | Keep generic custom-provider configs broad while promoted presets advertise only documented package families. |
| `anthropic` | `siumai-provider-anthropic`, `siumai-protocol-anthropic` | AI SDK-aligned native Anthropic Messages/files/skills/tools package surface with provider-owned metadata and replay helpers | Green | Registry metadata and `AnthropicClient` capability discovery now expose files and skills while keeping embedding/image/rerank/audio families unsupported; stream custom/raw hints remain confined to explicit provider-native replay and compatibility tests. |
| `google` | `siumai-provider-gemini`, `siumai-protocol-gemini` | AI SDK-aligned Google/Gemini chat/embedding/image/video/files/tools package surface plus explicit deferred Interactions boundary | Amber | `google.interactions(...)` package surface now has typed Rust handles/options/metadata/constants, but real `/interactions` execution, polling, cancellation, signatures, and stream transforms remain a dedicated runtime lane; continue stream and reasoning-file/source coverage review. |
| `google-vertex` | `siumai-provider-google-vertex`, Gemini/Anthropic/OpenAI-compatible integrations | Vertex-native, Anthropic-on-Vertex, Vertex MaaS, and missing Vertex XAI sub-entry boundary | Amber | Vertex-native root aliases now expose `google_vertex()` / `create_google_vertex()` while keeping `vertex()` / `create_vertex()` compatibility; next audit the upstream `@ai-sdk/google-vertex/xai` sub-entry instead of folding it into native xAI or generic Vertex MaaS by accident. |
| `amazon-bedrock` | `siumai-provider-amazon-bedrock` | AI SDK-aligned Bedrock chat/embedding/image/rerank/tools package surface with provider-owned runtime semantics | Green | Registry, facade, and public-path gates lock chat/embedding/image/rerank plus Anthropic tool re-exports; SigV4 credential-provider hooks and upstream test-only `generateId` stay deferred, and public image options are not exported because the upstream package index does not export them. |
| `cohere` | `siumai-provider-cohere` | AI SDK-aligned native Cohere v2 chat/embedding/rerank plus Rust package settings/version helpers | Green | Unified factory/catalog/public-surface guards lock chat/embedding/rerank and explicitly keep completion/image/audio/files unsupported; only non-index internals such as `CohereErrorData` and upstream `generateId` stay deferred. |
| `deepseek` | `siumai-provider-deepseek` | AI SDK-aligned chat/language-model package surface with provider-owned reasoning, metadata, and include-usage stream behavior | Green | Public settings/error/options exports and registry chat-only guards are locked; completion/embedding/image/audio/rerank remain unsupported despite the shared OpenAI-compatible runtime. |
| `groq` | `siumai-provider-groq`, OpenAI-compatible runtime | AI SDK-aligned chat/transcription plus provider-owned Rust speech extension | Green | OpenAI-compatible preset no longer inherits completion; Groq speech/TTS remains a documented provider-owned Rust extension outside the audited AI SDK chat/transcription catalog. |
| `xai` | `siumai-provider-xai` | AI SDK-aligned chat/responses/image/video/files/tools plus Siumai provider-owned Rust speech extension | Green | Registry metadata now declares `file_management`; speech/TTS remains documented outside the audited `@ai-sdk/xai` package surface while completion/embedding/rerank/transcription stay unsupported. |
| `togetherai` | `siumai-provider-togetherai`, OpenAI-compatible runtime | AI SDK-aligned text/image/rerank plus Siumai OpenAI-compatible audio extension | Green | Provider-owned image/rerank stay off generic compat image/rerank execution; speech/transcription remain documented Siumai audio extensions outside the audited `@ai-sdk/togetherai` package surface. |
| `mistral` | OpenAI-compatible promoted wrapper | Chat + embedding boundary | Green | Completion remains unsupported unless upstream package policy changes. |
| `perplexity` | OpenAI-compatible promoted wrapper | Chat/search provider boundary | Green | Completion and non-text families remain unsupported; keep typed web-search options and metadata shape aligned. |
| `fireworks` | OpenAI-compatible promoted wrapper plus provider-owned image | Hybrid text plus provider-owned image | Green | Completion/embedding/transcription are explicit documented compat capabilities; speech remains unsupported. |
| `deepinfra` | OpenAI-compatible promoted wrapper plus provider-owned image | Hybrid text/embedding plus image | Green | Completion/embedding are explicit compat capabilities; image remains provider-owned outside generic compat inference. |
| `moonshotai` | OpenAI-compatible promoted wrapper | Chat-only package boundary | Green | Completion remains unsupported and historical `moonshot` alias stays hidden as migration-only. |
| `ollama` | `siumai-provider-ollama` | Siumai-native provider not mirrored as AI SDK official package | Deferred | Maintain Rust package boundary; do not force AI SDK package parity. |
| `minimaxi` | `siumai-provider-minimaxi` | Siumai provider with text/audio/video/music capabilities | Deferred | Use provider-owned typed options; no official AI SDK package parity target. |
| `gateway` | `siumai-bridge`, `siumai-extras` | Bridge/gateway runtime helpers, not a Vercel Gateway provider clone | Deferred | Only add real gateway provider behavior if a future workstream owns it. |

## Compatibility Path Inventory

| Compatibility path | Allowed use | Risk | Follow-up |
| --- | --- | --- | --- |
| `LlmClient` | Backward compatibility, pooling, extension adapters, migration bridges | New features may land only on generic-client downcasts | AIPC-030/AIPC-040 source guard and handle audit. |
| `compat_*_client_with_ctx` | Historical factory methods and extension-only surfaces | Stable family handles may silently route through legacy clients | AIPC-040 locks remaining handle usages to extension-only image/audio helpers. |
| `as_*_capability()` | Compatibility client capability discovery | Broad optional interface hides stable family ownership | Keep out of stable family primary paths unless documented. |
| Legacy `ContentPart` | Serde and compatibility payload carrier | Request/response provider channels can mix | Follow ADR-0008; use directional adapters for new code. |
| Provider-scoped `Custom` stream events | Provider-native escape hatch and protocol replay only | Major AI SDK stream semantics can become raw JSON again | Prefer `ChatStreamEvent::Part` for stable semantics. |

## First Child-Slice Candidates

1. **Core/provider residue guard refresh**
   - Scope: `siumai-core`, tests.
   - Goal: Guard against provider-specific defaults, URL normalization, protocol transformers, and
     hosted-tool factories re-entering core.

2. **Registry stable-family compat audit**
   - Scope: `siumai-registry/src/registry/entry`.
   - Goal: Ensure stable language/embedding/completion/image/speech/transcription/rerank/video
     handles prefer native family paths when available.

3. **OpenAI-compatible promoted vendor capability audit**
   - Scope: `siumai-provider-openai-compatible`, facade/provider_ext tests.
   - Goal: Prevent Mistral, Perplexity, Fireworks, Groq, DeepInfra, MoonshotAI, and related
     packages from inheriting unsupported families through the shared runtime.

4. **Stable stream-part assertion migration**
   - Scope: protocol crates, bridge, extras gateway tests.
   - Goal: Move tests and helpers toward stable `Part` semantics where current assertions still
     depend on legacy/custom shadows.

## Inventory Update Rule

Every implementation slice must update this file when it changes:

- a seam status,
- a provider status,
- an intentional deviation from AI SDK,
- a child workstream decision,
- or a compatibility path rule.
