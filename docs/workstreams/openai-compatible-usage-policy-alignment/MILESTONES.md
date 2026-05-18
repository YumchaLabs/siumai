# OpenAI-Compatible Usage Policy Alignment - Milestones

Status: Completed
Last updated: 2026-05-18

## M0 - Scope

Exit criteria:

- Workstream docs exist.
- Usage policy is separated from broader provider hardcoding cleanup.
- AI SDK references are listed.

Result: Completed.

## M1 - Policy

Exit criteria:

- Usage extraction and conversion live behind one OpenAI-compatible policy module.
- Existing provider-specific token semantics are still represented.
- Call sites do not embed new provider usage matches.

Result: Completed in `siumai-protocol-openai/src/standards/openai/compat/usage.rs`.

## M2 - Runtime

Exit criteria:

- Streaming and non-streaming OpenAI-compatible response handling call the policy.
- SiliconFlow built-in streaming requests `stream_options.include_usage`.
- Generic openai-compatible remains opt-in.

Result: Completed. Stream, non-stream, completion, and xAI Responses paths are wired through the
policy or its Responses usage helpers.

## M3 - Closeout

Exit criteria:

- Focused tests pass.
- Formatting/check gates are recorded.
- Follow-on provider hardcoding candidates are named or explicitly deferred.

Result: Completed. Follow-ons are deferred to narrower lanes for reasoning parameters, message
conversion dispatch, and provider-specific tool fallback policy.
