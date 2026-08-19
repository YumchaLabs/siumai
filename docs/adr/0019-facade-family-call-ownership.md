# ADR-0019: Facade Family Call Ownership

## Status

Accepted

## Date

2026-08-19

## Context

Siumai has two valid public paths: provider-owned APIs for exact provider capabilities and
provider-neutral family traits for portable application code. The underlying ownership was sound,
but the facade exposed the portable path through `siumai::families::*`, paired every operation with
an additional `*_with_options` helper, and also re-exported runtime `generate` and `stream` at the
crate root and through the prelude. Runtime independently owned `AgentInput`, even though its string,
message, message-list, and complete-request conversions were provider-neutral language semantics.

This made the primary journey harder to recognize and created several visually similar call paths.
The README required callers to construct a complete request, manually assemble exact-target
`CallOptions`, choose a suffixed helper, and search response content to display text. The API still
preserved complete results and provider fidelity, but users had to understand internal layering for
an ordinary call.

Restoring the historical universal client would reduce syntax at the cost of ownership. It would
need to construct or hide providers, infer capabilities, duplicate Registry, or recover native
provider methods through enums, `Any`, or downcasting. Keeping the old namespace as aliases beside a
new namespace would leave two documented compatibility boundaries and repeat the stability problem.

The facade therefore needs one Rust-first call shape that works for concrete and Registry-resolved
models, preserves complete responses, binds typed provider intent to the actual target, and leaves
native resources on their concrete owners.

## Decision

### Root family modules

The `siumai` facade owns six canonical root modules:

| Module | Default operations | Bound call type and terminal |
|---|---|---|
| `language` | `generate`, `stream` | `LanguageCall::{generate, stream}` |
| `embedding` | `embed` | `EmbeddingCall::embed` |
| `rerank` | `rerank` | `RerankCall::rerank` |
| `image` | `generate` | `ImageCall::generate` |
| `speech` | `synthesize` | `SpeechCall::synthesize` |
| `transcription` | `transcribe` | `TranscriptionCall::transcribe` |

Each module exposes `call` for advanced configuration. The family-specific public call type borrows
one selected live model and owns the complete request. It supports one replaceable `CallOptions`
baseline and ordered typed provider-option patches. Both `with_options` and
`with_provider_options` return `Result<Self, ProviderOptionError>` and validate the assembled
candidate before accepting it. Terminal methods consume the call.

A small private call-state implementation may share storage and exact-target option assembly. It
does not own cross-family validation, a universal execution trait, error normalization, or a common
result enum. Each family module retains its request validation, operation name, response type, and
error contract.

The public `siumai::families` umbrella and its seven `*_with_options` helpers are removed. Runtime
single-call helpers remain available as `siumai::runtime::{generate, stream}` but are no longer bare
crate-root or prelude exports.

### Provider-neutral language input and text inspection

`siumai-core` owns `LanguageInput`. It normalizes a string, one `Message`, a `Vec<Message>`, or a
complete `LanguageRequest` immediately and losslessly into the canonical request. It stores no
model, route, option, validation, or wire state. Facade language calls and runtime Agent entry points
share this adapter; the duplicate public `AgentInput` type is removed.

`LanguageResponse` and `PartialLanguageOutput` expose `text_parts()` and `output_text()`.
`output_text()` concatenates canonical text parts in order without a separator or normalization,
returns `None` when no text part exists, and returns `Some("")` when present text parts concatenate
to an empty string. It excludes reasoning, refusals, tools, citations, media, and provider-native
state.

Text inspection is display-oriented. It does not replace the complete response, provider metadata,
termination, usage, warnings, or `project_assistant_history()`.

### Exact live model identity

The facade accepts an already selected model handle. It does not accept a Registry plus a route
string, construct a provider, discover credentials, choose a route, or resolve a target again.

Preflight follows one order:

1. resolve the call deadline;
2. validate the portable request;
3. validate exact provider-option selection for the same live model;
4. dispatch once through the existing family trait.

Exact-target semantics continue to include provider scope, model family, API mode, optional
canonical Registry route, and configured-instance identity. Facade-originated validation errors add
canonical route context when present. Provider errors, established stream lifecycle, and bounded
partial output pass through unchanged.

### Provider fidelity and Registry

Concrete models and Registry-resolved trait objects implement the same family trait and use the
same root module. Application call code does not match on provider identity.

Typed provider options use the ordinary bound call. Typed annotations remain attached to the
message, content part, or tool they modify. Complete provider metadata remains available on the
family response through provider-owned typed views.

Registry intentionally exposes only the erased family contract. It does not downcast or recover
provider-native capabilities. Applications retain a concrete configured provider or model when
they need files, batches, catalogs, sessions, hosted tools, or media jobs.

### Compatibility boundary

The six root modules, their default operations, their `call` entry points and public call types,
their complete family response types, and typed provider-option ownership are the canonical facade
compatibility boundary.

A later beta break to this boundary is allowed only with all of the following release evidence:

- an explicit architecture rationale naming the affected canonical symbols;
- an exact old-symbol-to-new-symbol migration map and behavioral notes;
- synchronized root README, facade rustdoc, and compile-checked examples;
- a changelog entry and release review that call out the break;
- Cargo-native feature checks for the affected no-default, provider, and Registry shapes.

Compatibility aliases are not a substitute for this evidence. New convenience APIs should extend
the canonical modules unless a new semantic owner is demonstrated.

## Options considered

### Option A: Restore a universal `Siumai` client

Rejected. It would duplicate provider construction and Registry, obscure exact configured-instance
identity, and require capability matching or downcasting for native APIs.

### Option B: Add root aliases while retaining `siumai::families`

Rejected. Two public namespaces would create two compatibility boundaries and leave documentation,
imports, and future migrations ambiguous.

### Option C: Add generic convenience methods directly to family traits

Rejected. Generic input methods would compromise object-safe use or exclude Registry trait objects.
The facade functions keep the core traits small and object-safe.

### Option D: Use one universal call builder and response enum

Rejected. Family requests, validation, operation names, error contracts, and lifecycle semantics are
not interchangeable. A universal type would encode a least-common-denominator capability bag.

### Option E: Use root family modules with family-specific bound calls

Chosen. It reduces ceremony while preserving the existing ownership, exact target, complete result,
and trait-object boundaries.

## Consequences

### Positive

- One application function can call a concrete or Registry-resolved model without provider
  matching.
- Plain language prompts and complete requests share one canonical path.
- Typed provider intent is adjacent to the call it modifies and is checked against the executing
  model before dispatch.
- Complete responses, metadata, annotations, streams, and native escape paths remain lossless.
- The facade has one documented namespace and a reviewable compatibility boundary.

### Costs

- Callers upgrading from `0.11.0-beta.10` must update `siumai::families` imports, suffixed option
  helpers, facade runtime imports, and custom `AgentInput` conversions.
- Advanced calls add one fallible builder step before the async terminal operation.
- Applications using Registry and provider-native APIs must retain both the Registry snapshot and
  the concrete provider instead of expecting runtime downcasts.
- Family call wrappers contain deliberate shallow repetition so each family remains explicit.

## Verification

No-default facade contracts compile every root family module. Registry contracts exercise erased
family handles and route-aware option binding. OpenAI and Anthropic fixtures exercise real typed
options, annotations, metadata, and complete responses without live credentials. The
`provider_switching` example compiles one application function against concrete, explicitly erased,
and Registry-resolved language models while retaining concrete provider resources. Flagship examples
compile with their individual provider features. Migration and release documentation carry the
exact beta.10 symbol map and Cargo-native gates.

## References

- `docs/architecture/overview.md`
- `docs/architecture/public-api.md`
- `docs/architecture/registry.md`
- `docs/adr/0013-provider-identity-and-family-registration.md`
- `docs/adr/0015-validation-ownership-and-forward-compatibility.md`
- `docs/migration/siumai-next.md`
- `docs/plans/2026-08-18-0108-refactor-unified-facade-ergonomics-plan.md`
