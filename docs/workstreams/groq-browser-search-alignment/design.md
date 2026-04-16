# Groq Browser Search Alignment - Design

Last updated: 2026-04-14

## Historical problem

`repo-ref/ai/packages/groq` exposes a provider-defined `browserSearch()` tool with these runtime
semantics:

- public tool id: `groq.browser_search`
- supported only on `openai/gpt-oss-20b` and `openai/gpt-oss-120b`
- the final Groq Chat Completions payload must contain `{ "type": "browser_search" }`
- unsupported models must not receive that tool on the wire
- unsupported models must emit a precise AI SDK-style warning instead of a generic compat warning

Siumai previously missed that behavior in multiple layers:

- there was no shared `groq.browser_search` tool factory on the stable `Tool` surface
- `provider_ext::groq` did not expose Groq provider tool constructors
- the shared OpenAI-compatible chat path only serialized function tools, so provider-defined Groq
  tools were silently dropped
- the generic OpenAI-compatible warning middleware marked every provider-defined tool as
  unsupported, which produced the wrong behavior for Groq `browser_search`

That meant Groq browser search was not merely undocumented; it was impossible to express
correctly through the public Rust surface.

## Implemented design

### 1. Shared stable tool factory

`siumai-spec::tools::groq` now owns the stable provider-defined tool constructor:

- `groq::BROWSER_SEARCH_ID`
- `groq::browser_search()`
- `groq::browser_search_named(...)`

`provider_defined_tool("groq.browser_search")` now also resolves to the same default tool shape.

This keeps Groq aligned with the existing OpenAI/Anthropic/Google/xAI provider-tool factory
pattern.

### 2. Compat warning allowlist

The shared OpenAI-compatible warning middleware now accepts an allowlist of provider-defined tool
ids that should not emit the generic
`unsupported { feature: "provider-defined tool <id>" }` warning.

This keeps the shared compat runtime reusable for providers that tunnel server-side tools through
Chat Completions but need provider-owned behavior.

### 3. Provider-owned Groq middleware

Groq now installs a dedicated model middleware before the generic compat warnings layer.

That middleware:

- detects whether the original request includes `groq.browser_search`
- checks whether the selected model is one of the Groq-supported GPT-OSS browser-search models
- appends `{ "type": "browser_search" }` to the final JSON payload when supported
- preserves mixed function tools instead of replacing the existing tool array
- restores `tool_choice` when the request only contains browser search and the generic OpenAI tool
  serializer would otherwise drop it
- emits the exact AI SDK-style unsupported warning when the current model does not support browser
  search

### 4. Public Rust surface alignment

The Groq provider extension facade now exposes:

- `siumai::provider_ext::groq::tools::*`
- `siumai::provider_ext::groq::provider_tools::*`

This matches the existing OpenAI/xAI/Anthropic provider-tool namespace story on the public facade.

## Validation

The implemented behavior is locked by:

- shared tool-factory tests in `siumai-spec`
- compat warning middleware tests with the Groq allowlist enabled
- Groq runtime tests for:
  - supported-model injection
  - unsupported-model warning behavior
  - mixed function + browser search tool payloads
  - `tool_choice: required` preservation
- facade compile guards for `provider_ext::groq::{tools, provider_tools}`

## Remaining follow-up

- Audit whether Groq exposes additional provider-defined tools in `repo-ref/ai/packages/groq`
  beyond `browserSearch`, and decide whether they belong on the stable shared `Tool` surface or
  should remain provider-owned escape hatches.
- Revisit stream/event-level parity if Groq later exposes provider-executed browser-search call
  events that should map onto stable stream parts or provider metadata.
- Continue comparing Groq typed request/response metadata against `repo-ref/ai/packages/groq/src/*`
  so browser search lands as one step in the wider provider-surface parity pass rather than a
  one-off patch.
