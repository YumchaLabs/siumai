# AI SDK Provider Market Expansion — TODO

Status: Draft
Last updated: 2026-05-25

## M0 — Scope And Evidence Freeze

- [x] PMX-010 [owner=planner] [deps=none] [scope=docs/workstreams/ai-sdk-provider-market-expansion]
  Goal: Freeze market evidence, problem statement, target state, non-goals, and execution boundaries.
  Validation: DESIGN.md, MILESTONES.md, EVIDENCE_AND_GATES.md, WORKSTREAM.json, and HANDOFF.md exist and agree.
  Evidence: `docs/workstreams/ai-sdk-provider-market-expansion/DESIGN.md`
  Handoff: Planner opened the lane from npm download data and local AI SDK package inventory.

## M1 — Gateway Contract Proof

- [x] PMX-020 [owner=codex] [deps=PMX-010] [scope=repo-ref/ai/packages/gateway,docs/workstreams/ai-sdk-provider-market-expansion]
  Goal: Audit `@ai-sdk/gateway` package behavior and decide the minimal Siumai implementation boundary.
  Validation: Add a gateway contract inventory documenting endpoints, headers, provider options, model families, metadata, and explicit non-goals.
  Review: review-workstream for scope control before implementation.
  Evidence: `docs/workstreams/ai-sdk-provider-market-expansion/GATEWAY_INVENTORY.md`
  Handoff: DONE. Gateway is a native AI SDK provider-protocol boundary, not an OpenAI-compatible preset. Recommended first proof is language + embedding only; split a dedicated Gateway workstream if implementation expands to media, OIDC, resources, tools, or stream-part refactors.

- [x] PMX-030 [owner=codex] [deps=PMX-020] [scope=siumai-provider-gateway,siumai-registry,siumai]
  Goal: Land the smallest no-network Vercel Gateway proof if PMX-020 confirms a viable minimal boundary.
  Validation: focused nextest for the new or touched provider/registry tests; cargo fmt check for touched crates.
  Review: review-workstream before accepting completion.
  Evidence: `docs/workstreams/ai-sdk-provider-market-expansion/EVIDENCE_AND_GATES.md`
  Handoff: DONE. Gateway is implemented as a native provider-protocol proof in `siumai-provider-gateway` with language non-stream/stream, embedding, typed request options, registry factory/catalog/default registry wiring, and facade exports through `provider_ext::gateway`, `providers::gateway`, and `Provider::gateway()`. Media, admin resources, OIDC, provider-defined tools, and full model catalog generation remain follow-ons.

## M2 — Cerebras High-ROI Provider Onboarding

- [x] PMX-040 [owner=codex] [deps=PMX-010] [scope=repo-ref/ai/packages/cerebras,siumai-provider-openai-compatible,siumai]
  Goal: Add Cerebras support through the lowest-coupling OpenAI-compatible path that matches AI SDK package behavior.
  Validation: focused nextest for Cerebras preset/model/facade tests; cargo fmt check for touched crates.
  Review: Local review-workstream pass; no blocking findings. External review still recommended before commit/merge.
  Evidence: `docs/workstreams/ai-sdk-provider-market-expansion/EVIDENCE_AND_GATES.md`
  Handoff: DONE. Cerebras is implemented as an OpenAI-compatible preset/facade with chat/language-model-only support, provider-owned model constants, request-body reasoning replay rewrite, GLM structured-output `tool_calls` finish normalization for text responses, repeated tool-call dropping when present, registry/provider catalog resolution, and no-network tests. Do not promote to a native provider unless future upstream behavior exceeds the shared OpenAI-compatible runtime.

## M3 — Mistral And Existing High-Download Provider Deepening

- [x] PMX-050 [owner=codex] [deps=PMX-010] [scope=repo-ref/ai/packages/mistral,siumai-provider-openai-compatible,siumai]
  Goal: Audit `@ai-sdk/mistral` against Siumai's Mistral preset/facade and close concrete package-surface gaps.
  Validation: audit notes plus focused tests for any fixed request/model/public-surface gaps.
  Review: review-workstream if code changes land.
  Evidence: `docs/workstreams/ai-sdk-provider-market-expansion/MISTRAL_AUDIT.md`
  Handoff: DONE. Mistral remains on the OpenAI-compatible runtime. PMX-050 fixed three bounded request-shape
  drifts: Mistral now strips unsupported common `top_k`, keeps `stop` for `stopSequences`, and preserves
  `reasoning_effort` for the current AI SDK-supported medium reasoning models (`mistral-medium-3` and
  `mistral-medium-3.5`). Do not open a native Mistral provider lane unless future upstream behavior exceeds the
  shared compat runtime.

- [ ] PMX-060 [owner=unassigned] [deps=PMX-010] [scope=siumai-provider-azure,siumai-provider-amazon-bedrock,siumai-provider-google-vertex,docs]
  Goal: Audit high-download enterprise providers for docs/test/API polish gaps that block real usage.
  Validation: focused no-network tests or docs updates for discovered Azure, Bedrock, and Vertex gaps.
  Review: review-workstream if code changes land.
  Evidence: `docs/workstreams/ai-sdk-provider-market-expansion/ENTERPRISE_POLISH_AUDIT.md`
  Handoff: Keep auth-heavy live tests optional unless a no-network equivalent cannot prove the contract.

## M4 — Audio And Media Provider Decision

- [ ] PMX-070 [owner=unassigned] [deps=PMX-010] [scope=repo-ref/ai/packages/deepgram,repo-ref/ai/packages/elevenlabs,repo-ref/ai/packages/fal,repo-ref/ai/packages/replicate,docs]
  Goal: Decide whether audio/media packages should enter Siumai's near-term provider roadmap.
  Validation: decision note ranks Deepgram, ElevenLabs, Fal, Replicate, and related media packages by market signal, implementation cost, and fit with stable families.
  Review: planner review before new provider work is opened.
  Evidence: `docs/workstreams/ai-sdk-provider-market-expansion/AUDIO_MEDIA_DECISION.md`
  Handoff: Split one dedicated provider lane only after the decision note picks a concrete target.

## M5 — Closeout

- [ ] PMX-080 [owner=planner] [deps=PMX-020,PMX-040,PMX-050,PMX-060,PMX-070] [scope=docs/workstreams/ai-sdk-provider-market-expansion]
  Goal: Close the lane or split any remaining broad work into narrower provider workstreams.
  Validation: verify-rust-workstream records fresh final gate evidence.
  Review: review-workstream has no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`
  Handoff: Summarize shipped providers, deferred packages, and remaining risks.
