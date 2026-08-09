# Provider Support Policy

This document defines what Siumai means when it claims support for a provider. It
does not list model IDs; provider-owned catalogs carry dated lifecycle advice and
unknown IDs remain valid input.

## Claim scope

Every model-family claim is scoped by:

```text
provider + technical_platform + family + protocol + api_mode + fidelity + stability + evidence
```

Provider-native resources, sessions, and jobs use the same identity and evidence fields but name a
native surface instead of pretending to be a model family. Every public named scope also carries an
official source and verification date through a provider-owned profile or support manifest.

Profiles and manifests are evidence surfaces, not executable capability authorities. Registry uses
the selected registration and request policy instead. A custom endpoint may expose generic claims or
an explicitly identified empty manifest when Siumai makes no named support assertion.

A broad provider name alone is not a support claim. For example, a native language
implementation does not imply native image or realtime support, and a cloud-hosted
deployment may differ from the provider's first-party platform.

Custom endpoints and generic compatibility configurations never inherit official fidelity or
named-model support merely because they use a branded provider builder. A custom service is generic
until the caller supplies a separately verified profile.

Account entitlement, commercial region availability, quotas, pricing, compliance, health, default
models, and fallback policy are host control-plane facts and are not part of a Siumai support claim.
When a remote API requires a caller-selected region, project, workspace, or deployment to address
or sign a request, Siumai models that value only as technical addressing context. It does not infer
that a model is commercially available there or publish an exhaustive region/deployment inventory.

### Upstream lifecycle evidence

`ApiStability` describes the stability of Siumai's public surface. It is independent from the
optional upstream evidence fields carried by `VerificationEvidence` and
`NativeVerificationEvidence`:

| Field | Meaning |
|---|---|
| `upstream.maturity` | Normalized provider maturity such as `stable`, `preview`, `beta`, or `experimental`. |
| `upstream.support_status` | Normalized provider status such as `active`, `legacy`, `deprecated`, or `retired`. |
| `upstream.official_label` | The provider's wording when normalization would lose useful meaning. |

Each field is independently optional. A missing value means that the cited official source does
not make that assertion; it is not a claim that the surface is active or stable. Siumai never
infers upstream state from model names, recommendation prose, or the absence of a deprecation
notice, and these fields never act as runtime capability or routing authority.

## Current facade surface

The `0.11.0-beta.9` facade intentionally exposes narrow provider slices. This table is an inventory
of compiled public scope, not a promise that every account can use every model or endpoint.

| Facade feature | Public scope | Deliberately not claimed |
|---|---|---|
| `openai` | Native Chat Completions and Responses; portable text embedding, image generation, buffered speech, and final-result transcription; provider-owned Responses, Conversations, Files, Vector Stores, and Skills slices | Image edits/streaming, realtime transcription, vector search/batches, zip skill upload, or a universal resource client |
| `openai-realtime` | Experimental native Realtime bootstrap and session transport | A stable provider-neutral realtime family |
| `anthropic` | Native Messages plus Anthropic-owned files, message batches, token counting, and skills | OpenAI-shaped language modes |
| `google` | Stable-v1 Interactions language/image, explicit stable-v1 Legacy GenerateContent, text embedding, buffered speech, Files metadata, and Veo submit/status | Stored/background Interactions, Live, File upload/register, broad Veo workflows, and broad Vertex support |
| `google-vertex-anthropic` | Verified Anthropic Messages execution on the caller-selected Vertex project and location | Vertex Gemini/media APIs or an SDK-maintained region/model catalog |
| `alibaba` | Verified Chat Completions, Responses, and Anthropic-compatible Messages modes, native embeddings, and experimental Wan video jobs | Complete Anthropic parity, a separate DashScope provider identity, or business-region routing |
| `moonshotai` | Verified Moonshot AI Kimi Chat Completions dialect with typed Kimi options, Partial Mode, and Files lifecycle | Kimi Batch, token-estimate, Formula, or other resources not implemented by the branded provider |
| `volcengine` | Verified Volcengine ARK Chat Completions/Responses dialects, portable Image, Remote MCP, and typed Video task lifecycle | Account-specific ARK deployments and media features outside the implemented Image/Video slices |
| `openai-compatible` | Explicit generic-compatible configuration for caller-owned endpoints | Named-provider fidelity, model advice, or native-provider resources |
| `groq` | Verified Chat Completions and Responses dialects plus final-result transcription | A universal OpenAI clone or unrelated Groq products |
| `xai` | Verified Responses and Chat Completions language modes with typed xAI tools/options | Files, image, speech, video, or a generic native-resource client |
| `minimax` | Verified Messages, Chat Completions, and bounded Responses modes; portable image and buffered speech; native files, media, input-token counting, and voice lifecycle resources | Cross-provider media/voice abstractions or hidden polling workflows |
| `deepseek` | Verified Chat Completions, explicit beta Chat strict/prefix, Responses, and Anthropic-compatible Messages language modes | Unverified non-language products |
| `cohere` | Native v2 embedding, including Embed v4, and rerank through Rerank v4/v3 | Cohere chat |
| `deepgram` | Native final-result prerecorded transcription with current Nova-3/Nova-2 hints | Flux/live transcription or legacy-model lifecycle claims |
| `elevenlabs` | Native speech synthesis | Transcription and broad resource clients |

`all-providers` activates the retained branded provider slices but intentionally does not enable
the generic `openai-compatible` escape hatch or experimental `openai-realtime` transport.

## Exact portable claim matrix

The rows below mirror the provider-owned profiles compiled on 2026-08-08. `Protocol / API mode`
names the exact execution surface; it is not provider-wide identity.

| Facade feature / provider | Provider / platform | Family | Protocol / API mode | Fidelity | Stability | Official source | Verified |
|---|---|---|---|---|---|---|---|
| `openai` / OpenAI | `openai` / `openai-api` | Language | `openai.responses` / `responses` | `native` | `stable` | https://developers.openai.com/api/docs/guides/latest-model | 2026-08-04 |
| `openai` / OpenAI | `openai` / `openai-api` | Language | `openai` / `chat-completions` | `native` | `stable` | https://developers.openai.com/api/docs/guides/latest-model | 2026-08-04 |
| `openai` / OpenAI | `openai` / `openai-api` | Embedding | `openai.embeddings` / `embeddings` | `native` | `stable` | https://developers.openai.com/api/docs/guides/embeddings | 2026-08-08 |
| `openai` / OpenAI | `openai` / `openai-api` | Image | `openai.images` / `image-generations` | `native` | `stable` | https://developers.openai.com/api/docs/guides/image-generation | 2026-08-08 |
| `openai` / OpenAI | `openai` / `openai-api` | Speech | `openai.audio` / `audio-speech` | `native` | `stable` | https://developers.openai.com/api/docs/guides/text-to-speech | 2026-08-08 |
| `openai` / OpenAI | `openai` / `openai-api` | Transcription | `openai.audio` / `audio-transcriptions` | `native` | `stable` | https://developers.openai.com/api/docs/guides/speech-to-text | 2026-08-08 |
| `anthropic` / Anthropic | `anthropic` / `anthropic-api` | Language | `anthropic-messages` / `messages` | `native` | `stable` | https://platform.claude.com/docs/en/api/messages | 2026-08-06 |
| `google` / Google Gemini | `google` / `gemini-api` | Language | `gemini-interactions` / `interactions` | `native` | `stable` | https://ai.google.dev/api/interactions-api | 2026-08-08 |
| `google` / Google Gemini | `google` / `gemini-api` | Image | `gemini-interactions` / `interactions` | `native` | `stable` | https://ai.google.dev/gemini-api/docs/image-generation | 2026-08-08 |
| `google` / Google Gemini | `google` / `gemini-api` | Language | `gemini-generate-content` / `generate-content` | `native` | `stable` | https://ai.google.dev/gemini-api/docs/generate-content/text-generation | 2026-08-08 |
| `google` / Google Gemini | `google` / `gemini-api` | Embedding | `gemini-embed-content` / `embed-content-v1` | `native` | `stable` | https://ai.google.dev/gemini-api/docs/embeddings | 2026-08-08 |
| `google` / Google Gemini | `google` / `gemini-api` | Speech | `gemini-interactions` / `interactions-speech` | `native` | `experimental` | https://ai.google.dev/gemini-api/docs/speech-generation | 2026-08-08 |
| `google-vertex-anthropic` / Claude on Vertex AI | `google` / `vertex-ai` | Language | `anthropic-messages` / `messages` | `verified-compatible` | `stable` | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude | 2026-08-06 |
| `alibaba` / Alibaba | `alibaba` / `alibaba-model-studio` | Language | `openai` / `chat-completions` | `verified-compatible` | `stable` | https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions | 2026-08-05 |
| `alibaba` / Alibaba | `alibaba` / `alibaba-model-studio` | Language | `openai.responses` / `responses` | `verified-compatible` | `stable` | https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-responses | 2026-08-05 |
| `alibaba` / Alibaba | `alibaba` / `alibaba-model-studio` | Language | `anthropic-messages` / `messages` | `verified-compatible` | `stable` | https://www.alibabacloud.com/help/en/model-studio/anthropic-api-messages | 2026-08-08 |
| `alibaba` / Alibaba | `alibaba` / `alibaba-model-studio` | Embedding | `alibaba-native` / `text-embedding` | `native` | `stable` | https://www.alibabacloud.com/help/en/model-studio/text-embedding-synchronous-api | 2026-08-06 |
| `moonshotai` / Kimi | `moonshotai` / `kimi-public-api` | Language | `openai` / `chat-completions` | `verified-compatible` | `stable` | https://platform.kimi.ai/docs/api/chat | 2026-08-08 |
| `volcengine` / ARK | `volcengine` / `ark-cn-beijing` | Language | `openai` / `chat-completions` | `verified-compatible` | `stable` | https://www.volcengine.com/docs/82379/1330626 | 2026-08-08 |
| `volcengine` / ARK | `volcengine` / `ark-cn-beijing` | Language | `openai.responses` / `responses` | `verified-compatible` | `stable` | https://www.volcengine.com/docs/82379/1585128 | 2026-08-08 |
| `volcengine` / ARK | `volcengine` / `ark-cn-beijing` | Image | `ark-images` / `images-generations` | `native` | `stable` | https://api.volcengine.com/api-docs/view?action=ImageGenerations&serviceCode=ark&version=2024-01-01 | 2026-08-08 |
| `volcengine` / ARK | `volcengine` / `ark-cn-beijing` | Video task | `ark-native` / `video-generation-tasks` | `native` | `stable` | https://api.volcengine.com/api-docs/view?action=CreateContentsGenerationsTasks&serviceCode=ark&version=2024-01-01 | 2026-08-08 |
| `groq` / Groq | `groq` / `groq-cloud` | Language | `openai` / `chat-completions` | `verified-compatible` | `stable` | https://console.groq.com/docs/openai | 2026-08-05 |
| `groq` / Groq | `groq` / `groq-cloud` | Language | `openai.responses` / `responses` | `verified-compatible` | `stable` | https://console.groq.com/docs/responses-api | 2026-08-05 |
| `groq` / Groq | `groq` / `groq-cloud` | Transcription | `groq-audio-transcriptions` / `audio-transcriptions` | `native` | `stable` | https://console.groq.com/docs/speech-to-text | 2026-08-06 |
| `xai` / xAI | `xai` / `xai-public-api` | Language | `openai` / `chat-completions` | `verified-compatible` | `stable` | https://docs.x.ai/developers/rest-api-reference/inference/chat | 2026-08-06 |
| `xai` / xAI | `xai` / `xai-public-api` | Language | `openai.responses` / `responses` | `verified-compatible` | `stable` | https://docs.x.ai/developers/rest-api-reference/inference/responses | 2026-08-06 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Language | `anthropic-messages` / `messages` | `verified-compatible` | `stable` | https://platform.minimax.io/docs/api-reference/text-chat-anthropic | 2026-08-06 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Language | `openai` / `chat-completions` | `verified-compatible` | `experimental` | https://platform.minimax.io/docs/api-reference/text-chat-openai | 2026-08-06 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Language | `openai.responses` / `responses` | `verified-compatible` | `experimental` | https://platform.minimax.io/docs/api-reference/responses-create | 2026-08-06 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Image | `minimax-image` / `image-generation` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/image-generation-t2i | 2026-08-08 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Speech | `minimax-speech` / `speech-http` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/speech-t2a-http | 2026-08-08 |
| `deepseek` / DeepSeek | `deepseek` / `deepseek-api` | Language | `openai` / `chat-completions` | `verified-compatible` | `stable` | https://api-docs.deepseek.com/api/create-chat-completion | 2026-08-05 |
| `deepseek` / DeepSeek | `deepseek` / `deepseek-api` | Language | `openai.responses` / `responses` | `verified-compatible` | `stable` | https://api-docs.deepseek.com/guides/responses_api | 2026-08-05 |
| `deepseek` / DeepSeek | `deepseek` / `deepseek-api` | Language | `anthropic-messages` / `messages` | `verified-compatible` | `stable` | https://api-docs.deepseek.com/guides/anthropic_api | 2026-08-08 |
| `deepseek` / DeepSeek | `deepseek` / `deepseek-beta-api` | Language | `openai` / `chat-completions` | `verified-compatible` | `experimental` | https://api-docs.deepseek.com/guides/tool_calls | 2026-08-08 |
| `cohere` / Cohere | `cohere` / `public-api` | Embedding | `cohere-native` / `v2` | `native` | `stable` | https://docs.cohere.com/v2/reference/embed | 2026-08-06 |
| `cohere` / Cohere | `cohere` / `public-api` | Rerank | `cohere-native` / `v2` | `native` | `stable` | https://docs.cohere.com/v2/reference/rerank | 2026-08-06 |
| `deepgram` / Deepgram | `deepgram` / `public-api` | Transcription | `deepgram-prerecorded` / `prerecorded` | `native` | `stable` | https://developers.deepgram.com/reference/speech-to-text/listen-pre-recorded | 2026-08-06 |
| `elevenlabs` / ElevenLabs | `elevenlabs` / `public-api` | Speech | `elevenlabs-native` / `text-to-speech` | `native` | `stable` | https://elevenlabs.io/docs/api-reference/text-to-speech/convert | 2026-08-04 |

The caller-supplied `openai-compatible` builder exposes either Chat Completions or Responses with a
caller-selected provider ID and `custom-endpoint` or `local` platform. Those claims are
`generic-compatible` and `experimental`; by design they have no Siumai-owned official source or
verification date.

## Exact provider-native claim matrix

Native resources, sessions, and jobs remain provider-owned lifecycle APIs rather than additional
portable model families.

| Facade feature / provider | Provider / platform | Kind / surface | Fidelity | Stability | Official source | Verified |
|---|---|---|---|---|---|---|
| `openai` / OpenAI | `openai` / `openai-api` | Resource / `responses-resource-lifecycle` | `native` | `stable` | https://developers.openai.com/api/reference/resources/responses/methods/create | 2026-08-06 |
| `openai` / OpenAI | `openai` / `openai-api` | Resource / `conversations-basic-items` | `native` | `stable` | https://developers.openai.com/api/reference/resources/conversations/methods/create | 2026-08-08 |
| `openai` / OpenAI | `openai` / `openai-api` | Resource / `files-basic-lifecycle` | `native` | `stable` | https://developers.openai.com/api/reference/resources/files/methods/create | 2026-08-08 |
| `openai` / OpenAI | `openai` / `openai-api` | Resource / `vector-stores-basic-files` | `native` | `stable` | https://developers.openai.com/api/reference/resources/vector-stores/methods/create | 2026-08-08 |
| `openai` / OpenAI | `openai` / `openai-api` | Resource / `skills-directory-lifecycle` | `native` | `experimental` | https://developers.openai.com/api/reference/resources/skills/methods/create | 2026-08-08 |
| `openai-realtime` / OpenAI | `openai` / `openai-api` | Session / `realtime` | `native` | `experimental` | https://developers.openai.com/api/docs/guides/realtime-websocket | 2026-08-06 |
| `openai-realtime` / OpenAI | `openai` / `openai-api` | Session / `realtime-translation` | `native` | `experimental` | https://developers.openai.com/api/docs/guides/realtime-translation | 2026-08-06 |
| `anthropic` / Anthropic | `anthropic` / `anthropic-api` | Resource / `files` | `native` | `experimental` | https://platform.claude.com/docs/en/api/files-create | 2026-08-06 |
| `anthropic` / Anthropic | `anthropic` / `anthropic-api` | Job / `message-batches` | `native` | `stable` | https://platform.claude.com/docs/en/api/creating-message-batches | 2026-08-06 |
| `anthropic` / Anthropic | `anthropic` / `anthropic-api` | Resource / `token-counting` | `native` | `stable` | https://platform.claude.com/docs/en/api/messages-count-tokens | 2026-08-06 |
| `anthropic` / Anthropic | `anthropic` / `anthropic-api` | Resource / `skills` | `native` | `experimental` | https://platform.claude.com/docs/en/api/skills/create-skill | 2026-08-06 |
| `google` / Google Gemini | `google` / `gemini-api` | Resource / `files-metadata` | `native` | `stable` | https://ai.google.dev/gemini-api/docs/files | 2026-08-08 |
| `google` / Google Gemini | `google` / `gemini-api` | Job / `veo-predict-long-running` | `native` | `experimental` | https://ai.google.dev/gemini-api/docs/veo | 2026-08-08 |
| `alibaba` / Alibaba | `alibaba` / `alibaba-model-studio` | Job / `video-tasks` | `native` | `experimental` | https://www.alibabacloud.com/help/en/model-studio/text-to-video-api-reference | 2026-08-06 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Resource / `files` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/file-management-upload | 2026-08-08 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Resource / `images` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/image-generation-t2i | 2026-08-08 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Job / `video-tasks` | `native` | `experimental` | https://platform.minimax.io/docs/api-reference/video-generation-v2-create | 2026-08-08 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Resource / `music` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/music-generation | 2026-08-08 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Resource / `speech-http` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/speech-t2a-http | 2026-08-08 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Job / `speech-async-tasks` | `native` | `experimental` | https://platform.minimax.io/docs/api-reference/speech-t2a-async-create | 2026-08-08 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Resource / `responses-input-tokens` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/responses-input-tokens | 2026-08-09 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Resource / `voice-cloning` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/voice-cloning-clone | 2026-08-09 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Resource / `voice-design` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/voice-design-design | 2026-08-09 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Resource / `voice-management` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/voice-management-get | 2026-08-09 |
| `minimax` / MiniMax | `minimax` / `minimax-api` | Resource / `voice-delete` | `native` | `stable` | https://platform.minimax.io/docs/api-reference/voice-management-delete | 2026-08-09 |

## Fidelity

| Value | Meaning |
|---|---|
| `native` | Siumai implements the provider's material native protocol/resource semantics for the claimed scope. |
| `verified-compatible` | Siumai uses a compatibility protocol with a named, tested dialect profile for this provider and scope. |
| `generic-compatible` | Users may supply an endpoint and credentials through a generic compatibility builder; Siumai makes no named provider claim. |

## Public stability

| Value | Meaning |
|---|---|
| `stable` | The Siumai family contract is supported by the normal semver policy. |
| `experimental` | The protocol may be native, but the provider-owned session, job, or stream API may change in a breaking release. |

Fidelity and stability are independent. Realtime can be both `native` and
`experimental`.

## Evidence for named profiles

A built-in `native` or `verified-compatible` claim must include:

1. An official provider documentation URL.
2. A `verified_at` date and platform/API-mode scope.
3. A provider-owned policy for meaningful dialect differences.
4. Offline success, error, and applicable stream fixtures.
5. Model lifecycle entries only where an official source exists.

Lifecycle entries use `active`, `deprecated`, `retired`, or `rolling-alias`, and may
name a replacement. They are advisory. CI validates their structure and staleness;
it does not scrape websites or generate Rust from another SDK's model union.

## Generic compatibility

The generic OpenAI-compatible builder remains an escape hatch for private gateways
and unlisted services. It requires an explicit endpoint policy and credential
audience, exposes only protocol-baseline behavior, and does not inherit a named
provider's fidelity claim.

## Maintenance workflow

Provider changes are audited against official documentation first. The local
Vercel AI SDK checkout is secondary evidence for fixtures and edge cases. A
maintainer updates the provider policy, lifecycle data, and behavior fixtures in
one change; a static mega-list of remote gateway models is not accepted.
