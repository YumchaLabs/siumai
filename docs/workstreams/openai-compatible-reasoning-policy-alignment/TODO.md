# OpenAI-Compatible Reasoning Policy Alignment - TODO

Status: Completed
Last updated: 2026-05-18

## M0 - Scope And Evidence Freeze

- [x] OCR-010 [owner=planner] [deps=none] [scope=docs/workstreams/openai-compatible-reasoning-policy-alignment]
  Goal: Freeze the reasoning policy target and AI SDK reference points.
  Validation: workstream docs exist and agree.
  Evidence: `DESIGN.md`
  Handoff: This lane focuses on request-side reasoning mapping only.

## M1 - Shared Reasoning Policy

- [x] OCR-020 [owner=codex] [deps=OCR-010] [scope=siumai-protocol-openai/src/standards/openai/compat]
  Goal: Add a shared OpenAI-compatible reasoning policy that returns provider-native request
  parameters for all existing fluent reasoning APIs.
  Validation: focused protocol tests pass.
  Review: builder/config must not duplicate provider-id reasoning matches.
  Evidence: `siumai-protocol-openai/src/standards/openai/compat/reasoning.rs`
  Handoff: `OpenAiCompatibleReasoningPolicy` owns DeepSeek, Alibaba/Qwen/SiliconFlow, MoonshotAI,
  xAI, OpenRouter, and generic legacy request-parameter mapping.

## M2 - Builder And Config Integration

- [x] OCR-030 [owner=codex] [deps=OCR-020] [scope=siumai-provider-openai-compatible,siumai-protocol-openai]
  Goal: Route `OpenAiCompatibleBuilder` and `OpenAiCompatibleConfig` reasoning methods through the
  shared policy.
  Validation: builder/config regression tests pass.
  Review: no duplicated `match provider_id` reasoning maps remain in call sites.
  Evidence: builder/config tests.
  Handoff: Provider-specific providerOptions normalization remains in `compat::spec`; fluent
  defaults now enter through shared policy patches.

## M3 - Verification And Closeout

- [x] OCR-040 [owner=planner] [deps=OCR-030] [scope=docs/workstreams/openai-compatible-reasoning-policy-alignment]
  Goal: Run focused gates, record evidence, and close or split follow-ons.
  Validation: fmt/check/nextest/diff gates recorded.
  Review: no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Split response-policy or message-dispatch cleanup into separate lanes.
