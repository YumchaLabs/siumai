# ADR-0012: Provider Annotations Follow Semantic Nodes

## Status

Accepted

## Date

2026-08-05

## Context

The provider-neutral language request intentionally contains only semantics that can be carried by
more than one provider. Provider APIs nevertheless attach useful behavior to specific prompt nodes:
prompt-cache breakpoints belong to content blocks, tool loading and caller restrictions belong to
tool definitions, and some replay metadata belongs to a message or generated part.

Call-scoped provider options cannot represent those relationships safely. Encoding node locations as
`message_index` and `content_index` inside one top-level option object makes the request fragile:
middleware that inserts, removes, combines, or reorders messages silently changes what an index
means. Copying the legacy provider-options map onto every data structure would avoid index drift but
would reintroduce untyped recursive JSON, ambiguous merge behavior, and protected-field risks.

Provider-only request DTOs are still useful for direct provider APIs, but requiring them for every
provider feature would make the unified language interface a least-common-denominator path. The
unified interface needs a bounded Rust-native way to retain provider intent without promoting each
provider field into `siumai-core`.

## Decision

Siumai keeps call configuration and durable node annotations as separate contracts.

### Call configuration

`CallOptions` and `ProviderOptions` describe how one invocation is configured. They retain explicit
provider, route, model, runtime-step, call, and checked-raw precedence layers. The selected provider
owns validation and deterministic merge behavior. A foreign namespace in call configuration is an
error because it is ambiguous which provider should execute the option.

### Durable node annotations

Messages, message content parts, and model-visible tool definitions may carry provider annotations
next to the semantic node they modify.

- `Message` owns message annotations.
- `MessagePart` wraps one neutral `ContentPart` together with content annotations.
- `ToolSpec` owns tool annotations.

Annotations are typed, namespaced, bounded, serializable, and target-specific. Provider crates own
the annotation structs and implement the core erasure contract with one associated `Target`, so one
annotation type cannot be attached to several semantic node kinds. Core retains the provider
namespace, optional API mode, and validated object value, but does not understand provider fields or
merge them.

Each node may contain at most one annotation value for a provider namespace. A provider that needs
several fields defines one node-specific option struct and validates their relationships together.
There is no annotation precedence stack and no recursive merge algorithm.

### Typed-only construction

Normal Rust construction is typed-only. Node annotations do not expose the call option system's raw
override constructor. Deserialization remains possible for durable or dynamic workflows, but it
reapplies namespace, shape, protected-field, depth, field-count, and byte limits; the selected
provider then deserializes and validates its exact typed annotation before encoding.

Byte accounting measures each annotation as an exact single-namespace JSON map, including the
namespace and erased envelope. Request aggregation sums those single-entry maps, producing an exact
value for one namespace per node and a conservative upper bound when a node carries several. Core
checks nesting depth, field count, and protected fields before serializing the retained value for
byte accounting.

Annotations cannot set credentials, authorization, endpoints, audiences, proxy or TLS policy,
redirect policy, host headers, or arbitrary transport headers.

### Foreign history annotations

Provider annotations stored in conversation history are not call configuration. A provider codec
reads only annotations in its namespace and validates their API mode and semantic target before
encoding. Other provider namespaces remain inert and are not sent to that provider. They may remain
in durable history so switching away from and later back to a provider does not destroy replay
information.

Provider-native state whose removal changes the meaning or validity of the conversation does not
belong in an optional annotation. It remains a bounded `ProviderOpaque` item with provenance and is
handled by the explicit history-projection loss policy.

### Content ownership

`ContentPart` remains the provider-neutral content algebra. Annotation storage is not repeated in
every enum variant. `MessagePart` is the request/history envelope, with constructors and accessors
that keep ordinary text/tool/media usage concise.

`LanguageResponse` remains response-directional and continues to expose bare `ContentPart` values.
Provider response metadata is not converted into request annotations automatically. A runtime that
appends portable response content to history wraps it with empty annotations. Provider state that is
required for continuation—such as reasoning signatures, encrypted content, fallback blocks, or a
paused provider operation—uses a provenance-bearing `ProviderOpaque` item instead.

## Options considered

### Option A: Request-level annotation side table keyed by indexes

Rejected because middleware and runtime history mutation can invalidate indexes without changing the
annotated data. Making indexes stable would require synthetic node IDs and mutation bookkeeping that
are unnecessary when the annotation can live beside its node.

### Option B: Add an untyped provider map to every message, content variant, and tool

Rejected because it duplicates the legacy map surface, permits ambiguous values, and forces every
provider to repeat shape, security, merge, and diagnostics rules.

### Option C: Put provider-specific fields in the shared request

Rejected because prompt caching, server-tool controls, reasoning signatures, and similar fields do
not have one portable meaning across providers. Core would become a provider capability catalog.

### Option D: Require provider-native request types for every provider feature

Rejected because it would make the unified interface convenient only for basic text and tools. A
typed provider annotation is a narrower extension seam that preserves provider ownership.

### Option E: Store typed annotations next to semantic nodes

Chosen because the relationship survives mutation, the provider owns the schema, foreign history is
inert, and the neutral request does not learn provider-specific fields.

## Consequences

### Positive

- Prompt-cache and tool annotations cannot silently move to another node after middleware changes.
- Provider features remain available through the unified language model trait.
- Direct provider APIs and the unified API can reuse the same provider-owned option types.
- Durable history can preserve multiple providers' replay hints without treating them as active
  configuration or sending them across provider boundaries.
- Core enforces bounds and protected fields without implementing provider schemas.

### Costs

- `Message.content` uses `MessagePart` rather than bare `ContentPart`, which is a deliberate public
  API break.
- Protocol codecs and runtimes must unwrap parts explicitly and preserve annotations when rebuilding
  history.
- Provider crates need separate types for call-level options and node-level annotations when the
  remote API has both.
- Dynamic callers that previously inserted arbitrary nested JSON must deserialize a validated
  request shape or use a provider-direct API instead of bypassing typed construction.

## Migration

1. Add target-marked `ProviderAnnotations` and `TypedProviderAnnotation` contracts to
   `siumai-core`.
2. Introduce `MessagePart` and migrate canonical language requests, runtime history, and request
   codecs from bare content parts.
3. Add message annotations to `Message` and tool annotations to `ToolSpec`; keep
   `LanguageResponse` response-directional.
4. Implement provider-owned annotation types for demonstrated node-level behavior, beginning with
   Anthropic Messages prompt caching and tool controls.
5. Replace provider option fields that identify content by numeric indexes with node annotations.
6. Keep required provider-native replay state in `ProviderOpaque`; do not encode it as an optional
   annotation merely to avoid projection handling.

## References

- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-prompt.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-function-tool.ts`
- `docs/architecture/public-api.md`
- `docs/plans/2026-08-04-001-refactor-siumai-next-revival-plan.md`
