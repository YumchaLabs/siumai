# OpenAI-Compatible Usage Policy Alignment - TODO

Status: Completed
Last updated: 2026-05-18

## M0 - Scope And Evidence Freeze

- [x] OCU-010 [owner=planner] [deps=none] [scope=docs/workstreams/openai-compatible-usage-policy-alignment]
  Goal: Freeze the usage policy target state and AI SDK evidence anchors.
  Validation: workstream docs exist and agree.
  Evidence: `docs/workstreams/openai-compatible-usage-policy-alignment/DESIGN.md`
  Handoff: This lane focuses only on usage policy, not broader reasoning/message dispatch cleanup.

## M1 - Usage Policy Module

- [x] OCU-020 [owner=codex] [deps=OCU-010] [scope=siumai-protocol-openai/src/standards/openai]
  Goal: Introduce a deep OpenAI-compatible usage policy module that owns provider usage extraction
  and conversion.
  Validation: focused protocol tests for provider usage conversion pass.
  Review: ensure no new call-site provider matches are introduced.
  Evidence: `siumai-protocol-openai/src/standards/openai/compat/usage.rs`
  Handoff: Provider usage conversion is now local to `compat::usage`; generic OpenAI-compatible,
  Alibaba/Qwen, DeepSeek, Groq, MoonshotAI, xAI, DeepInfra, and xAI Responses semantics have
  focused tests.

## M2 - Runtime Integration

- [x] OCU-030 [owner=codex] [deps=OCU-020] [scope=siumai-protocol-openai,siumai-provider-openai-compatible]
  Goal: Route stream and non-stream OpenAI-compatible response handling through the policy and keep
  provider family `include_usage` defaults aligned with AI SDK.
  Validation: `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
  Review: issue #20 SiliconFlow regression must remain covered.
  Evidence: `siumai-protocol-openai/src/standards/openai/compat/streaming.rs`,
  `siumai-protocol-openai/src/standards/openai/compat/transformers.rs`,
  `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/tests.rs`
  Handoff: Stream, non-stream, completion, and xAI Responses paths now call the policy. Old
  provider usage dispatch was removed from `utils.rs`.

## M3 - Verification And Closeout

- [x] OCU-040 [owner=planner] [deps=OCU-030] [scope=docs/workstreams/openai-compatible-usage-policy-alignment]
  Goal: Record final gates and split any non-usage hardcoding cleanup into a follow-on.
  Validation: formatting, focused nextest, and diff checks recorded.
  Review: no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Completed. Remaining provider hardcoding candidates are reasoning parameter mapping,
  message conversion dispatch, and provider-specific tool fallback policy; they should be split
  into future lanes only when they have a concrete behavior target.
