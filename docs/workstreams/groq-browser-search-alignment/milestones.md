# Groq Browser Search Alignment - Milestones

Last updated: 2026-04-14

## GBS-M0 - Scope locked

Acceptance criteria:

- the AI SDK Groq browser-search behavior is audited from `repo-ref/ai/packages/groq/src/*`
- the mismatch is framed as a runtime/public-surface gap rather than a documentation-only issue
- the implementation plan chooses provider-owned middleware over leaking temporary request state

Status: completed

## GBS-M1 - Stable tool factory exists

Acceptance criteria:

- `siumai-spec::tools::groq::browser_search()` exists
- `provider_defined_tool("groq.browser_search")` resolves to the same default tool shape
- the default tool name stays aligned with AI SDK (`browser_search`)

Status: completed

## GBS-M2 - Compat runtime can defer provider-owned warnings

Acceptance criteria:

- the generic OpenAI-compatible tool-warning middleware supports an allowlist
- Groq can suppress the generic unsupported warning for `groq.browser_search`
- unsupported-model warnings still come from the provider-owned Groq layer with precise details

Status: completed

## GBS-M3 - Groq runtime behavior matches AI SDK

Acceptance criteria:

- supported GPT-OSS models receive `{ "type": "browser_search" }` on the final wire payload
- unsupported models do not receive the browser-search tool on the wire
- mixed function tools remain intact
- provider-defined browser search no longer loses `tool_choice: required`

Status: completed

## GBS-M4 - Public surface, docs, and changelog are aligned

Acceptance criteria:

- `provider_ext::groq::{tools, provider_tools}` expose the browser-search factory
- public compile/runtime tests cover the new surface
- a dedicated workstream folder and `Unreleased` changelog entries document the change

Status: completed
