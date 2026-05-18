# OpenAI-Compatible Usage Policy Alignment

Status: Completed
Last updated: 2026-05-18

## Why This Lane Exists

Issue #20 exposed that OpenAI-compatible streaming usage is not only a provider preset problem.
The provider runtime also needs a single place to declare how usage is requested, extracted, and
converted.

## Relevant Authority

- `repo-ref/ai/packages/openai-compatible/src/openai-compatible-provider.ts`
- `repo-ref/ai/packages/openai-compatible/src/chat/openai-compatible-chat-language-model.ts`
- `repo-ref/ai/packages/deepseek/src/chat/deepseek-chat-language-model.ts`
- `repo-ref/ai/packages/alibaba/src/alibaba-provider.ts`
- `repo-ref/ai/packages/alibaba/src/convert-alibaba-usage.ts`
- `repo-ref/ai/packages/moonshotai/src/convert-moonshotai-chat-usage.ts`
- `repo-ref/ai/packages/xai/src/convert-xai-chat-usage.ts`
- `repo-ref/ai/packages/google-vertex/src/xai/google-vertex-xai-provider.ts`
- `siumai-provider-openai-compatible/src/providers/openai_compatible/config/family_defaults.rs`
- `siumai-protocol-openai/src/standards/openai/utils.rs`

## Problem

OpenAI-compatible usage behavior is split across preset config and protocol helper functions.
`include_usage` defaults now live in provider family defaults, but usage extraction and conversion
still dispatch on provider ids inside `siumai-protocol-openai`. That makes future provider fixes
look like small matches instead of an explicit provider runtime policy.

## Target State

- Generic `openai-compatible` does not guess stream usage by default.
- Built-in provider presets declare stream usage defaults in one provider-family policy.
- Usage extraction and conversion are represented as one OpenAI-compatible usage policy interface.
- Protocol streaming and non-streaming response transformers consume the policy instead of
  embedding usage provider matches at the call site.
- Issue #20 remains fixed for SiliconFlow and the existing provider-specific token semantics remain
  covered by tests.

## In Scope

- OpenAI-compatible chat usage extraction and conversion.
- Stream `include_usage` default policy.
- Tests for generic, SiliconFlow, DeepSeek, Alibaba/Qwen, MoonshotAI, xAI, Groq, DeepInfra, and
  Google Vertex xAI usage behavior.
- Changelog entries already prepared for v0.11.0-beta.8.

## Out Of Scope

- Reasoning-parameter mapping cleanup.
- Message conversion dispatch cleanup.
- Model catalog, base URL, auth, capability, and provider option key registries.
- Non-OpenAI-compatible provider runtimes.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| AI SDK treats generic `openai-compatible` as opt-in for stream usage. | High | `createOpenAICompatible` passes `includeUsage: options.includeUsage`. | Generic providers could receive unsupported `stream_options`. |
| Provider packages own usage converters when token semantics differ. | High | Alibaba, DeepSeek, MoonshotAI, xAI converters in `repo-ref/ai`. | Provider-specific token accounting will remain hidden in protocol utilities. |
| Provider names are acceptable in preset data and tests. | High | Existing model/catalog/config registries are provider-defined data. | Over-removing provider ids would make preset behavior harder to audit. |

## Architecture Direction

Create a deep OpenAI-compatible usage policy module. The interface should answer:

- should this preset request stream usage by default?
- where can usage live in a response payload?
- how is provider usage converted into the unified `Usage` shape?

The protocol transformer should call this interface through its provider id. Provider-specific
knowledge remains local to the usage policy module, not spread across stream and non-stream
transformers.

## Closeout Condition

This lane can close when:

- usage provider dispatch is local to one policy module,
- old duplicate usage helper paths are removed or narrowed,
- focused provider usage tests pass,
- package-level OpenAI-compatible nextest passes,
- formatting/check gates are recorded,
- and any remaining provider hardcoding is explicitly classified as a follow-on lane.

Closeout result: completed. Usage provider dispatch is local to `compat::usage`, runtime call sites
consume the policy, issue #20 remains covered, and non-usage provider dispatch candidates are
deferred to narrower future lanes.
