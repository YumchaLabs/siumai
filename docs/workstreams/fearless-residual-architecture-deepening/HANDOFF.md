# Fearless Residual Architecture Deepening - Handoff

Status: Closed
Last updated: 2026-05-27

## Current State

The workstream is closed. FRAD-010 through FRAD-070 are complete, and the final closeout gates pass.

## Active Task

None. Do not reopen this lane for mechanical cleanup; open a narrower follow-on only if new
behavioral evidence requires it.

## Decisions

- Use one durable workstream for all five candidates because they share the same post-AI-SDK
  residual architecture review and closeout gate.
- Do not open a new ADR yet. Existing ADRs cover the direction; `ContentPart` remains controlled by
  ADR-0008.
- Execute in dependency order: registry descriptor first, protocol dialects second, bridge codecs
  third, test harness fourth, `ContentPart` gate/move fifth.
- FRAD-020 concentrated built-in provider default-model lookup, factory resolution, and enabled
  factory registration in `registry::provider_descriptor`; the public helper functions now cross
  that seam, and catalog projection uses `ProviderCatalogDescriptor`.
- FRAD-030 moved OpenAI-compatible, OpenAI Chat, Perplexity, DeepSeek, xAI, and Mistral message
  conversion into `utils::message_dialect` with dialect-local tests, while `utils::*` keeps stable
  compatibility re-exports for existing call sites.
- FRAD-040 split bridge request normalization into per-wire-format codec modules for OpenAI
  Responses, OpenAI Chat Completions, Anthropic Messages, and Gemini GenerateContent. The parent
  `normalize.rs` now keeps public wrapper functions, request hook/loss-policy flow, and shared
  helpers.
- FRAD-050 deepened test harnesses: factory family override requirements are named
  `FactoryFamilyOverrideContract` scenarios, provider public-path source guards use a
  `ProviderPublicPathModule` manifest object, and built-in registry parity setup crosses
  `BuiltInProviderRegistryHarness`.
- FRAD-060 moved high-value production legacy `ContentPart` usage to explicit compat imports and
  recorded `FRAD-060-content-part-root-move-decision.md`: the low-level root move remains blocked
  until a full provider/protocol/bridge root-move fixture suite exists.
- FRAD-070 reviewed and closed the lane. The closeout gate initially exposed
  OpenAI Responses hosted tool-result `providerExecuted` loss, a stale video facade source guard,
  and a feature-gated Vertex xAI audio guard; all were fixed and the six-crate closeout gate now
  passes.

## Next Recommended Action

No active action for this workstream. The only known residual risk is the ADR-0008 `ContentPart`
root move blocker recorded by FRAD-060.
