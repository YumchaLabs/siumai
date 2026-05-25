# AI SDK Provider Market Expansion — Handoff

Status: Draft
Last updated: 2026-05-25

## Current State

The workstream is open. Market evidence shows Siumai already covers the major direct providers, while the
next high-impact work is enterprise provider polish and audio/media provider prioritization.

PMX-010, PMX-020, PMX-030, PMX-040, and PMX-050 are complete. PMX-060 is the next unresolved task.

## Active Task

- Task ID: PMX-060
- Owner: unassigned
- Files:
  - `repo-ref/ai/packages/azure/*`
  - `repo-ref/ai/packages/amazon-bedrock/*`
  - `repo-ref/ai/packages/google-vertex/*`
  - `siumai-provider-azure/*`
  - `siumai-provider-amazon-bedrock/*`
  - `siumai-provider-google-vertex/*`
  - `siumai-registry/*`
  - `siumai/*`
- Validation:
  - focused no-network tests or docs updates for any Azure, Bedrock, or Vertex polish gaps.
  - `cargo fmt --check` for touched crates.
- Status: READY_TO_AUDIT
- Review: Pending
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

## Blockers

- None currently.

## Next Recommended Action

- Continue with PMX-060 enterprise provider polish audit for Azure, Bedrock, and Google Vertex. Keep it
  audit-first: only land code if a high-value gap is concrete, bounded, and can be proven with no-network
  focused tests.
