# AI SDK Provider Interface Convergence - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The program workstream is open. The target seams, initial parity inventory, milestones, gates, and
task ledger are recorded. AIPC-030, AIPC-040, AIPC-050, AIPC-060, and AIPC-070 are complete.
AIPC-080 is also complete: provider package rows now either have green evidence, intentional Rust
boundaries, deferred non-official package status, or a child workstream.
AIPC-050 closed after three stream-part slices: OpenAI Responses public feature-surface tests now exercise stable
`ChatStreamEvent::Part` tool call/result inputs instead of provider custom event inputs; extras
gateway smoke tests now require stable downstream tool stream parts for the Anthropic-to-OpenAI
Responses route; Anthropic and Gemini serializer tests now have custom-input source guards so stable
serializer behavior cannot be covered only through custom event inputs. Converter-level custom-event
compatibility tests remain in place where they explicitly prove backward compatibility or
provider-native replay behavior.

AIPC-060 closed after bridge and extras gateway/helper coverage proved that OpenAI Responses stream
bridging emits output item frames from stable provider-tool stream parts, not only V3/custom
compatibility events. Extras Axum SSE code imports the OpenAI Responses parts adapter directly from
`siumai_bridge::stream` instead of through the facade experimental streaming re-export, and a
boundary test locks that seam. M2 is complete.

AIPC-070 closed after re-auditing promoted OpenAI-compatible vendor completion inheritance. The
shared compat adapter no longer infers completion from chat transport; completion is exposed only
when provider metadata explicitly contains `completion`. DeepSeek, Groq, xAI, OpenRouter,
SiliconFlow, Alibaba/Qwen, Mistral, Perplexity, and MoonshotAI are locked as non-completion compat
presets. TogetherAI, DeepInfra, Fireworks, generic custom OpenAI-compatible configs, and Vertex MaaS
retain explicit completion support where Siumai has documented family coverage.

This lane is intentionally a coordination and execution program. It should keep spawning bounded
vertical slices instead of becoming one cross-provider mega patch.

## Active Task

- Task ID: AIPC-080
- Owner: codex
- Files:
  - `siumai-provider-*`
  - `siumai-protocol-*`
  - `docs/workstreams/ai-sdk-provider-interface-convergence/*`
- Validation:
  - package-specific no-network tests
  - focused public import tests for touched providers

## Decisions Since Last Update

- Opened a new program workstream instead of reopening `fearless-refactor-v4`.
- Kept `ai-sdk-structural-alignment` as historical evidence rather than the active machine-readable
  lane.
- Chose AIPC-030 as the first executable slice because source guards are safer before broader
  provider or registry rewrites.
- AIPC-030 found a stale empty local `siumai-core/src/standards/openai` directory; it was not tracked
  by Git and was removed so the new guard reflects the intended tracked source tree.
- AIPC-040 found that remaining registry handle compatibility clients are already extension-only:
  image edit/variation helpers and audio streaming/listing/translation helpers. A new boundary test
  locks that shape so primary stable-family execution does not regress.
- AIPC-050 started with OpenAI Responses because its public feature surface still modeled stable
  tool stream parts through `Custom("openai:*")` inputs even though production serialization already
  accepts `ChatStreamEvent::Part`.
- AIPC-050 then tightened the extras gateway smoke helper from typed-or-custom to stable-part-only
  so gateway tests cannot mask a regression from stable tool parts back to provider custom payloads.
- AIPC-050 added protocol serializer source guards for Anthropic and Gemini: `Custom` serializer
  inputs are allowed only in explicitly named V3 compatibility, provider-native, or compatibility
  tests.
- AIPC-050 intentionally did not remove loose custom-event parsers in extras object/tool-loop/server
  helpers. Those are compatibility-boundary consumers rather than stable protocol feature-surface
  tests and should be audited under AIPC-060 only if bridge/gateway evidence shows they can hide a
  stable-part regression.
- AIPC-060 treats `OpenAiResponsesStreamPartsBridge` as bridge-owned. The facade may re-export it
  for advanced users, but extras runtime gateway code should import it directly from
  `siumai_bridge::stream`.
- AIPC-060 added direct extras SSE helper coverage for stable provider-tool stream parts, so the
  bridge/gateway stream milestone can close without removing compatibility-boundary custom-event
  readers in object/tool-loop helpers.
- AIPC-070 makes OpenAI-compatible completion capability metadata explicit. The old denylist model
  was too brittle because every chat-capable promoted vendor inherited completion unless listed as
  an exception.
- Vertex MaaS remains completion-capable, but it is a dedicated Google Vertex MaaS factory/package
  decision rather than evidence that ordinary OpenAI-compatible promoted presets should inherit
  completion.
- AIPC-080 started with Groq. The upstream `@ai-sdk/groq` package surface is chat/transcription
  plus tools; Siumai keeps Groq speech/TTS as a provider-owned Rust extension. New registry catalog
  and factory tests lock that boundary so speech models stay out of the AI SDK-aligned
  chat/transcription catalog while remaining listed on the Rust provider surface.
- AIPC-080 then re-audited TogetherAI terminology. The upstream `@ai-sdk/togetherai` package
  surface is chat/completion/embedding/image/reranking. Siumai intentionally keeps
  speech/transcription as OpenAI-compatible audio extensions, so the docs and metadata now name
  those audio paths as Siumai extensions instead of folding them into AI SDK package parity.
- AIPC-080 then re-audited xAI. The upstream `@ai-sdk/xai` package surface is
  chat/responses/image/video/files/tools and explicitly rejects embedding. Siumai already had a
  provider-owned files lane, but registry metadata did not advertise `file_management`; that is now
  fixed. Siumai's xAI speech/TTS path remains a documented provider-owned Rust extension outside
  the audited AI SDK package surface, while completion/embedding/rerank/transcription stay
  unsupported.
- AIPC-080 then re-audited Cohere. Existing code/tests already align the native Cohere package
  surface to chat/embedding/rerank on `/v2`, with public `cohere()`, `create_cohere()`,
  `CohereProviderSettings`, `VERSION`, typed options, and curated model modules. The stale Cohere
  child-workstream notes that still treated settings/version as deferred were corrected, and the
  AIPC inventory row is now Green. `CohereErrorData` remains deferred because it is not exported
  from the audited upstream package index; upstream `generateId` remains deferred because Siumai has
  no comparable provider-level stable ID hook.
- AIPC-080 then re-audited DeepSeek. A new lightweight child record documents that the upstream
  `@ai-sdk/deepseek` package is chat/language-model only, with settings, error data, typed options,
  and `VERSION` exported. Siumai already exposes the matching Rust package helpers and guards the
  registry/provider-owned boundary so completion/embedding/image/audio/rerank do not leak in from
  the shared OpenAI-compatible runtime. The AIPC inventory row is now Green.
- AIPC-080 then re-audited Amazon Bedrock. The upstream `@ai-sdk/amazon-bedrock` package surface is
  chat/language-model, embedding/text-embedding aliases, image, rerank, Anthropic tools, settings,
  and `VERSION`. Siumai already exposes the matching Rust package helpers through
  `provider_ext::bedrock`, uses provider-owned Bedrock clients for registry family paths, and has
  public-path parity coverage for settings plus chat/embedding/image/rerank request construction.
  SigV4 credential-provider hooks and upstream test-only `generateId` stay deferred, and Siumai does
  not export a public image-options type because the upstream package index does not export one.
  The AIPC inventory row is now Green.
- AIPC-080 then re-audited Anthropic. The upstream `@ai-sdk/anthropic` package surface is
  language/chat/messages, files, skills, provider tools, settings/version, typed options/metadata,
  and container-id forwarding. Siumai already had the provider-owned files and skills resources, but
  registry metadata and `AnthropicClient` capability discovery did not advertise files consistently.
  The provider metadata/client capability drift is now fixed and guarded while embedding, image,
  rerank, speech, transcription, and audio remain unsupported family paths. The AIPC inventory row
  is now Green.
- AIPC-080 then re-audited Google/Gemini against the current `@ai-sdk/google` package surface. The
  upstream package now exposes `google.interactions(...)` targeting `POST /v1beta/interactions`,
  plus Interactions model ids, agent names, provider options, and metadata. Siumai now mirrors that
  package-visible boundary with typed Rust handles/options/metadata/constants and public facade
  guards, but the handle intentionally fails fast for chat execution. Real `/interactions` request
  conversion, polling, cancellation, signatures, interaction-id compaction, and stream transforms
  must land in a dedicated runtime lane rather than being routed through ordinary Gemini
  `:generateContent`.
- AIPC-080 then re-audited the Google Vertex package root. The current upstream root uses
  `googleVertex` / `createGoogleVertex` as the primary names and keeps `vertex` / `createVertex` as
  deprecated aliases. Siumai now exposes the Rust snake_case primary names
  `provider_ext::google_vertex::{google_vertex, create_google_vertex}` and
  `Provider::google_vertex()` while keeping the existing `vertex` / `create_vertex` compatibility
  aliases. The same re-audit found a larger remaining package boundary:
  `@ai-sdk/google-vertex/xai` exists upstream and is now modeled as its own AIPC-080 slice with
  dedicated `google_vertex_xai` / `vertex_xai` aliases, registry metadata, capability guards, and
  request-body normalization instead of being hidden under native `xai` or generic Vertex MaaS.
- AIPC-080 then re-audited Azure. The upstream `@ai-sdk/azure` package exports OpenAI option aliases
  plus Azure-owned Responses metadata envelope names rooted at `providerMetadata.azure`. Siumai now
  exports the matching Azure Responses metadata envelope types through `siumai-provider-azure` and
  `provider_ext::azure`, while the existing registry/public-path tests continue to prove default
  Azure runtime metadata uses `azure` and only switches to `openai` when the caller explicitly
  overrides the provider metadata key.
- AIPC-080 then re-audited OpenAI's package index. The upstream `@ai-sdk/openai` package exports
  five `OpenaiResponses*ProviderMetadata` envelope names for response/reasoning/text/compaction/
  source-document metadata. Siumai now exports matching Rust structs through the OpenAI protocol,
  provider, and facade metadata modules, while existing Responses stream/replay behavior stays on
  the protocol/bridge evidence already closed by AIPC-050 and AIPC-060.
- AIPC-080 closed the remaining Google/Gemini package row by separating package-surface parity from
  runtime execution. Ordinary Gemini reasoning/source/provider-metadata public paths already have
  focused coverage. `google.interactions(...)` remains package-visible and fail-fast by design until
  the new `docs/workstreams/google-interactions-runtime-alignment` lane implements the dedicated
  `/interactions` runtime.

## Blockers

- None currently.

## Next Recommended Action

Execute AIPC-090 next. Normalize or explicitly defer legacy unknown lanes that this program
supersedes, starting from `docs/workstreams/INDEX.md` and only updating lanes whose status can be
inferred from their own docs or this program's explicit follow-on split.
