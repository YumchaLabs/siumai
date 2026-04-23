# Anthropic Vertex Package Surface Alignment - Design

Last updated: 2026-04-22

## Problem

Compared with `repo-ref/ai/packages/google-vertex/src/anthropic/index.ts` and
`repo-ref/ai/packages/google-vertex/src/anthropic/google-vertex-anthropic-provider.ts`, Siumai's
Anthropic-on-Vertex slice still had a few package-boundary drifts:

- no dedicated `GoogleVertexAnthropicProviderSettings` on the provider-owned/public Rust surface
- no package-level constructor aliases matching `vertexAnthropic` / `createVertexAnthropic`
- no dedicated `GoogleVertexAnthropicMessagesModelId` type alias or up-to-date curated model-id
  subset aligned with the audited upstream union
- no audited Vertex-supported Anthropic tool subset exposed under a dedicated sub-entry module
- the public Anthropic-on-Vertex metadata export still only exposed the older wide helper names,
  not the narrower audited `AnthropicMessageMetadata` container/iteration surface
- provider-owned and registry/unified builder construction still leaned too hard on explicit
  `base_url`, even though the AI SDK package constructs the Anthropic Vertex base URL from
  `project + location` or env fallbacks
- the wrapper still let structured-output `auto` mode fall back to Anthropic's native
  model-family heuristic, even though the upstream Vertex Anthropic package explicitly disables
  native structured outputs and defaults JSON-schema requests to the reserved `json` tool path

That made diffing against the reference package noisy and kept the same wrapper provider behaving
inconsistently across `Provider`, `provider_ext`, and registry/unified-builder entry paths.

## Goals

- Mirror the honest `@ai-sdk/google-vertex/anthropic` package-root names where Rust has a truthful
  equivalent.
- Reuse one truthful construction story across provider builder, package settings wrapper, and
  registry/unified builder paths.
- Expose only the Vertex-supported Anthropic tool subset instead of pretending the full Anthropic
  tool catalog is available.
- Re-export the narrowed Anthropic typed metadata surface on the Anthropic-on-Vertex facade too.

## Non-goals

- Do not fabricate a callable TypeScript-style provider object on the Rust facade.
- Do not widen the Vertex Anthropic wrapper into image/embedding/video families that the current
  upstream package does not expose.
- Do not delete the older wide `AnthropicMetadata` helper; it remains useful as a broader Rust
  convenience layer.

## Chosen design

### 1. Add a real package-level settings wrapper

The provider-owned/public Anthropic-on-Vertex surface now exposes
`GoogleVertexAnthropicProviderSettings`.

It mirrors the honest package-level input subset:

- `project`
- `location`
- `base_url`
- `headers`
- `fetch`

and keeps the Rust-only auth analogue:

- `token_provider`

The settings wrapper converts into the provider runtime through:

- `into_builder()`
- `into_builder_for_model(model)`
- `into_config_for_model(model)`

### 2. Align base-URL derivation across all construction paths

The AI SDK package derives the Anthropic Vertex base URL from `project + location` and only uses
`baseURL` as an override.

Rust now follows the same principle on all audited paths:

- `VertexAnthropicBuilder` now accepts `project(...)` and `location(...)`
- `GoogleVertexAnthropicProviderSettings` forwards those fields instead of forcing early string
  synthesis
- the registry/unified-builder factory for `anthropic-vertex` now also derives the same base URL

The shared derived form is:

- `https://{location == "global" ? "" : location + "-"}aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/anthropic/models`

Rust centralizes that shape through `google_vertex_anthropic_base_url(...)` in the shared Vertex
auth helper module.

### 3. Mirror the package-level constructor aliases honestly

The stable public Anthropic-on-Vertex facade now exposes:

- `vertex_anthropic()`
- `create_vertex_anthropic()`

It also keeps the existing Rust-first `Provider::anthropic_vertex()` path and adds a matching
`Provider::vertex_anthropic()` alias so the naming story is easier to compare against the audited
AI SDK package.

### 4. Expose only the supported Anthropic tool subset

The Anthropic-on-Vertex facade now exposes the audited Vertex-supported Anthropic tool subset under
dedicated submodules:

- `tools`
- `provider_tools`
- `hosted_tools`

This subset includes:

- `bash_20241022`
- `bash_20250124`
- `text_editor_20241022`
- `text_editor_20250124`
- `text_editor_20250429`
- `text_editor_20250728`
- `computer_20241022`
- `web_search_20250305`
- `tool_search_regex_20251119`
- `tool_search_bm25_20251119`

That keeps the wrapper surface close to the upstream package without falsely advertising newer
Anthropic tools that Vertex does not currently accept.

### 5. Re-expose the model-id surface more honestly

The upstream package also exports a dedicated `GoogleVertexAnthropicMessagesModelId` type.

Rust now mirrors that boundary with:

- `GoogleVertexAnthropicMessagesModelId = String`
- an audited `models::{chat, ALL_CHAT}` subset that matches the current upstream union

The older Rust-only constants such as `claude-sonnet-4-5-latest` are intentionally removed from
the curated public subset because they are not part of the current audited AI SDK package contract.
The runtime still accepts arbitrary explicit model ids as strings; only the curated public/exported
subset is narrowed back to the audited upstream package surface.

### 6. Re-export the narrowed Anthropic metadata surface on the wrapper path

The Anthropic-on-Vertex facade now re-exports both the older wide helper and the narrower AI SDK
style message metadata surface:

- `AnthropicMetadata`
- `AnthropicMessageMetadata`
- `AnthropicMessageContainerMetadata`
- `AnthropicMessageContainerSkill`
- `AnthropicUsageIteration`

This matters because the wrapper still stores its public metadata under the Anthropic provider root,
so downstream users should not need to import the base Anthropic package facade just to access the
audited typed metadata contract.

### 7. Make Vertex structured-output defaults match the wrapper contract

The upstream `createVertexAnthropic()` wrapper passes `supportsNativeStructuredOutput: false` into
the shared Anthropic language-model runtime. That means JSON-schema structured outputs default to
the reserved `json` tool fallback on the Vertex wrapper path, even when the underlying Anthropic
model family would otherwise support native structured outputs.

Rust now mirrors that wrapper-level contract in two places:

- `VertexAnthropicConfig::new(...)` seeds default provider options with
  `structured_output_mode = jsonTool`
- the Vertex Anthropic request/stream spec synthesizes that same effective default for direct
  request transforms and stream conversion, so the request body and the stream-end/text replay path
  cannot drift apart

Explicit `OutputFormat` overrides remain available as an escape hatch on the Rust provider-owned
surface, but the default audited wrapper behavior is now the same `json` tool fallback story as
the current AI SDK package.

## Validation

Locked by:

- provider-builder/unit tests in
  `siumai-provider-google-vertex/src/providers/anthropic_vertex/{builder.rs,settings.rs}`
- public compile guards in `siumai/tests/public_surface_imports_test.rs`
- public builder/runtime alignment tests in `siumai/tests/anthropic_vertex_builder_alignment_test.rs`
- targeted `cargo nextest` runs for the provider crate and the public `siumai` facade

## Remaining follow-up

- Re-audit the `google-vertex/anthropic` tool subset when upstream Vertex Anthropic adds or drops
  supported Anthropic hosted tools.
- Decide later whether the unified-builder surface should also grow a dedicated
  `base_url_for_anthropic_vertex(...)` usage example in docs beyond the current generic Vertex
  helper story.
