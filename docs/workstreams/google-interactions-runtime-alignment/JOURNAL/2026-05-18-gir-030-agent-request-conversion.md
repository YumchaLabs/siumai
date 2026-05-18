# GIR-030 Agent Request Conversion

Date: 2026-05-18

## Summary

Implemented agent-mode request conversion for Google Interactions. Agent requests now use the
dedicated `agent` branch with `background: true` and provider-owned `agent_config` lowering.

## Scope

- Agent-mode bodies send `agent` instead of `model`.
- Agent-mode bodies set `background: true` and do not send `stream` on the POST body.
- `agentConfig` is translated to the Interactions `agent_config` wire shape.
- Model-only fields are warned and dropped for agents: tools, generation config, structured output,
  and deprecated `imageConfig`.
- Runtime execution remains fail-fast; this task only prepares request bodies.

## Evidence

- `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_agent --no-fail-fast`
- `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_request --no-fail-fast`
- `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast`
- `git diff --check`

## Next Slice

GIR-040 should parse completed Interactions responses into stable `ChatResponse` content, usage,
finish reason, and `provider_metadata.google`.
