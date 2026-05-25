# AI SDK Provider Market Expansion — Handoff

Status: Closed
Last updated: 2026-05-26

## Current State

The workstream is closed. Market evidence and reference-package audits now support the shipped provider
surface and the explicit split of remaining media work into follow-on lanes.

PMX-010, PMX-020, PMX-030, PMX-040, PMX-050, PMX-060, PMX-070, and PMX-080 are complete.

## Active Task

- Task ID: PMX-080
- Owner: planner
- Files:
  - `docs/workstreams/ai-sdk-provider-market-expansion/*`
- Validation:
  - verify-rust-workstream records fresh final gate evidence.
  - review-workstream has no blocking findings.
  - `python .agents/skills/siumai-ai-sdk-maintenance/scripts/resolve_ai_sdk_repo.py`
- Status: COMPLETE
- Review: Complete
- Evidence: `docs/workstreams/ai-sdk-provider-market-expansion/EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Treat DeepInfra full catalog parity as deferred unless new product evidence changes the cost/benefit.
- Treat Gateway as P0 for design/proof because AI SDK defaults through Gateway and npm downloads show large adoption, even though downloads are partly transitive.
- Treat Cerebras as the first high-ROI onboarding candidate because its AI SDK package wraps OpenAI-compatible behavior with bounded provider quirks.
- Treat Mistral as an audit-first candidate, not an automatic native-provider rewrite.
- PMX-020 found that Gateway is a native AI SDK provider-protocol boundary. The recommended first proof is language + embedding only; split a dedicated Gateway implementation lane if it expands to media, OIDC, admin resources, tools, or broad stream-part refactors.
- PMX-040 landed Cerebras through the shared OpenAI-compatible path rather than a native crate. The supported
  public surface is chat/language-model only through `provider_ext::cerebras`, `providers::cerebras`,
  `Provider::cerebras()`, `SiumaiBuilder::cerebras()`, registry lookup, and model catalog constants.
  Cerebras request-body transformation rewrites assistant `reasoning_content` history to `reasoning`, and
  GLM structured-output responses that return text with `finish_reason: tool_calls` are normalized to a
  stop finish, even when no repeated tool-call part is present, while repeated tool-call parts are dropped
  when they do appear. Non-text families are rejected before transport use.
- PMX-030 landed the bounded Gateway proof as a native provider-protocol crate. The supported first surface is
  language non-stream/stream, embedding, explicit model ids, `AI_GATEWAY_API_KEY` or explicit API key auth,
  custom base URL/header/transport config, typed `GatewayOptions`, registry factory/catalog/default registry
  wiring, and facade exports through `provider_ext::gateway`, `providers::gateway`, and `Provider::gateway()`.
  Gateway media families, OIDC, admin resources, provider-defined tools, and full model catalog generation
  remain follow-ons.
- PMX-050 audited `@ai-sdk/mistral` against Siumai's OpenAI-compatible Mistral preset/facade. Mistral remains
  on the shared compat runtime. The bounded fixes were request-body only: strip unsupported common `top_k`, keep
  `stop` for `stopSequences`, and preserve `reasoning_effort` for `mistral-medium-3` and `mistral-medium-3.5`,
  matching the current AI SDK package support list. A native Mistral provider lane is deferred unless future
  upstream behavior exceeds the compat runtime.
- PMX-060 audited `@ai-sdk/azure`, `@ai-sdk/amazon-bedrock`, and `@ai-sdk/google-vertex` against Siumai's
  provider-owned crates, registry factories, facade exports, and no-network tests. Azure and Google Vertex are
  already sufficiently covered for this polish lane. Bedrock gained the bounded package-surface alias
  `amazon_bedrock()` across `provider_ext::bedrock`, `compat::Provider`, and `SiumaiBuilder`, matching AI SDK's
  canonical `amazonBedrock` export while preserving `bedrock()`.
- Bedrock Anthropic and Bedrock Mantle are follow-ons, not PMX-060 fixes. They need dedicated factory/auth/model
  catalog/request gates because upstream exposes them as sub-provider packages.
- PMX-070 audited `@ai-sdk/deepgram`, `@ai-sdk/elevenlabs`, `@ai-sdk/fal`, and `@ai-sdk/replicate` against
  Siumai's stable speech, transcription, image, and video families. No provider implementation lands in this
  lane. Deepgram is the first dedicated audio-provider candidate because it has the strongest media-package
  download signal and a narrow speech/transcription surface. ElevenLabs is second, especially for TTS/voice
  workflows. Replicate and Fal are deferred to media-provider lanes after queue/polling policy and
  model-specific request-shape gates are explicit.

## Blockers

- None.

## Next Recommended Action

- Open a dedicated follow-on only for the provider lane that actually gets approved next. Deepgram and
  ElevenLabs remain audio candidates; Replicate and Fal remain media candidates with polling policy gates.
