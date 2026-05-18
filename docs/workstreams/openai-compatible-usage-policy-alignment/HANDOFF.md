# OpenAI-Compatible Usage Policy Alignment - Handoff

Status: Completed
Last updated: 2026-05-18

## Current State

The lane is complete. OpenAI-compatible usage extraction and conversion now live behind
`OpenAiCompatibleUsagePolicy` in `siumai-protocol-openai/src/standards/openai/compat/usage.rs`.
Generic OpenAI-compatible behavior remains opt-in for stream usage, while built-in presets keep
their provider-family `include_usage` defaults.

## Next Task

No remaining task in this lane.

## Constraints

- Do not broaden into reasoning parameter mapping in this lane.
- Do not remove provider catalog, base URL, auth, model, or capability registries.
- Preserve generic openai-compatible opt-in stream usage behavior.
- Preserve SiliconFlow issue #20 regression coverage.

## Follow-On Candidates

- Reasoning parameter mapping currently still has provider-specific dispatch.
- Message conversion dispatch still selects provider-specific OpenAI-compatible message mappers.
- SiliconFlow tool-call JSON-in-text fallback can become a provider response policy if another
  provider needs the same behavior.
