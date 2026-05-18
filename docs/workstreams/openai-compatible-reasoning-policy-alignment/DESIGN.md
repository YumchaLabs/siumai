# OpenAI-Compatible Reasoning Policy Alignment - Design

Status: Active
Last updated: 2026-05-18

## Problem

OpenAI-compatible reasoning request behavior is implemented in more than one place:

- `OpenAiCompatibleBuilder` maps `reasoning(...)`, `reasoning_budget(...)`, and thinking aliases.
- `OpenAiCompatibleConfig` repeats the same provider-id branching.
- Protocol request normalization also understands provider-specific reasoning option aliases.

This makes the runtime semantics harder to change safely. A provider preset should own catalog
data and defaults, while request-time behavior should live behind an explicit provider policy.

## Reference

Vercel AI SDK keeps generic `openai-compatible` behavior simple and configurable:

- generic compat models pass `reasoning_effort` from provider options or top-level reasoning;
- provider-specific packages use provider-owned option schemas and mapping helpers;
- `mapReasoningToProviderEffort` and `mapReasoningToProviderBudget` centralize cross-provider
  reasoning conversion instead of scattering per-call logic.

Siumai already has AI SDK-style helpers in `siumai-core::utils::reasoning`. This lane applies the
same idea to OpenAI-compatible preset/runtime policy.

## Target State

- A single OpenAI-compatible reasoning policy module owns provider-id to request-parameter mapping.
- Builder and config fluent APIs call the same policy rather than duplicating provider matches.
- Existing public methods remain source-compatible in this beta line.
- Generic OpenAI-compatible providers retain the current `enable_reasoning` / `reasoning_budget`
  behavior for legacy Siumai callers.
- Provider-specific typed options continue to normalize through existing provider option handling.

## Scope

- Add a protocol-layer reasoning policy module under `siumai-protocol-openai`.
- Replace duplicated reasoning/thinking mapping in the OpenAI-compatible builder and config.
- Add focused tests that prove shared policy semantics for DeepSeek, Alibaba/Qwen, MoonshotAI, xAI,
  OpenRouter, and generic providers.
- Update workstream index and evidence.

## Non-Goals

- Do not redesign top-level `LanguageModelReasoning` call options in this lane.
- Do not remove provider catalog/base URL/auth/model registries.
- Do not change response reasoning extraction or streaming event semantics.
- Do not broaden into message conversion dispatch unless a test exposes required coupling.

## Architecture Direction

The policy should be a deep module: callers express a simple intent (`thinking`, `thinking_budget`,
`reasoning`, `reasoning_budget`), and the module returns provider-native request parameters. This
keeps locality high: adding or correcting a provider reasoning rule should touch one policy and its
tests, not builder and config call sites.
