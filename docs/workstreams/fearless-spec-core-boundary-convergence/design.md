# Fearless Spec/Core Boundary Convergence - Design

Last updated: 2026-05-15

## Context

The previous fearless-refactor workstreams moved Siumai toward the right architecture:

- provider family APIs are the preferred execution center
- protocol conversion is no longer expected to live in `siumai-core`
- registry construction is becoming family-first
- facade compatibility paths are more explicit

The remaining risk is that `siumai-spec` and `siumai-core` are still too wide. Some spec types carry
runtime semantics, and some core modules still hold provider or protocol residue. This makes future
provider alignment harder because new behavior can land in the broadest crate instead of in the
crate that owns the responsibility.

This workstream is the boundary-convergence pass after `fearless-boundary-hardening`. It is not a
provider parity track. Its purpose is to make the crate boundaries small enough that future provider
work naturally follows the architecture.

Vercel AI remains the reference shape at the architectural level:

- stable model specifications are distinct from provider implementations
- prompt/input shapes and response/result shapes are separate enough to prevent accidental mixing
- UI/runtime helpers are layered above serializable model contracts
- provider-specific metadata and options are carried through typed extension points, not central
  core implementation modules

## Decision

Treat `siumai-spec` as serializable contract space and `siumai-core` as provider-agnostic runtime
space. Any runtime handle, provider-specific bridge, protocol serializer, or compatibility downcast
that crosses those boundaries must either move to its owning crate or be documented as a temporary
compatibility artifact.

This is a fearless refactor track. When old structure conflicts with the target boundaries and has a
clear replacement path, prefer removal or relocation over adding another compatibility layer.

## Goals

- Remove runtime-only handles and cancellation semantics from `siumai-spec`.
- Split oversized AI SDK surface files along prompt, result, UI, and runtime concerns.
- Separate request-side provider options from response-side provider metadata where the current type
  shape encourages mixing concerns.
- Keep provider defaults, hosted tool factories, and stream bridge logic out of `siumai-core` unless
  they are genuinely provider-agnostic.
- Make `LlmClient` visibly compatibility-only for stable family paths.
- Narrow facade and registry re-exports so stable APIs do not mirror entire implementation crates.
- Add source guards before large moves where regressions are easy to reintroduce.

## Non-goals

- Do not use this track to implement missing provider features.
- Do not preserve undocumented compatibility paths simply because they existed before.
- Do not split files mechanically when the ownership boundary is already clear.
- Do not move gateway/server concerns into `siumai-core`.
- Do not make `siumai-spec` depend on async runtime crates for convenience.

## Target Boundaries

### `siumai-spec`

Owns serializable provider-agnostic data contracts:

- prompt and message data shapes
- request and response data containers
- provider option and provider metadata maps
- usage, finish reason, and shared result data
- stable JSON-compatible extension points

Forbidden:

- runtime cancellation handles
- async stream types
- HTTP execution
- provider construction
- provider-specific wire conversion
- retry, middleware, or transport policy

### `siumai-core`

Owns provider-agnostic runtime primitives:

- family traits
- shared errors
- HTTP abstractions
- retry and middleware contracts
- generic streaming carriers
- provider-agnostic tool/runtime contracts

Forbidden:

- provider-specific URL normalization
- provider-specific request/response transformers
- provider-specific stream event serialization
- provider-owned hosted tool factories
- stable family execution through generic `LlmClient` downcasts

### `siumai-bridge`

Owns bridge contracts, route policy, and dispatch between normalized model calls and protocol
targets.

Expected:

- bridge-owned request and response adapters
- explicit protocol target selection
- bridge tests for route behavior and fallback policy

### `siumai-protocol-*`

Owns wire-format conversion:

- provider/protocol request serializers
- provider/protocol response parsers
- stream event decoders and encoders
- protocol-owned warning, usage, metadata, and finish-reason extraction

### `siumai-provider-*`

Owns provider clients, provider-specific options, metadata, hosted tools, resources, endpoint
defaults, and extensions.

### `siumai-registry`

Owns lookup, provider factories, build-context propagation, handle caching, and explicit
compatibility adapters.

Expected:

- stable family handles use native family model paths
- compatibility generic clients live behind `compat_*` naming
- registry modules do not re-export broad core or provider surfaces as stable API

### `siumai`

Owns the facade:

- stable public modules and preludes
- feature aggregation
- provider extension exports
- explicit compatibility entry points

Expected:

- no heavy conversion logic
- no provider extension implementation bodies in `lib.rs`
- no stable prelude exports for compatibility-only names

## Refactor Strategy

### 1. Add Boundary Guards First

Add source guards for the easiest regressions:

- runtime crates or runtime handles entering `siumai-spec`
- stable registry family paths calling `as_*_capability()` or `compat_*_client*`
- provider-specific bridge targets or serializers living in `siumai-core`
- broad facade or registry re-exports that mirror implementation crates

### 2. Remove Runtime Semantics From `siumai-spec`

Move `CancelHandle` and related cancellation runtime semantics out of spec. Keep only serializable
request data, response data, and provider-neutral metadata contracts in `siumai-spec`.

### 3. Split The AI SDK Surface By Responsibility

Split oversized type modules such as `types/ai_sdk.rs` along stable ownership lines:

- prompt and message input data
- generated result and response data
- UI message data
- runtime helpers and handles outside spec

The goal is to make dependency direction visible in file ownership, not merely to reduce file size.

### 4. Separate Prompt-Side And Response-Side Content Views

Review content parts that currently carry both request-side `providerOptions` and response-side
`providerMetadata`. Prefer separate prompt and response projections or explicit adapters so provider
metadata cannot leak into request construction by accident.

Near-term decision: use an adapter-first migration instead of immediately adding another public
non-V4 content enum. The AI SDK V4 `language_model_v4::{prompt,content}` split is already the
concrete directional projection:

- request producers normalize request replay/detail/cache data into `providerOptions`.
- UI message conversion treats UI-layer `providerMetadata` names as request metadata and funnels
  them through `ui_request_*` adapter helpers.
- response parsers and stream parsers continue to emit response-side `providerMetadata`.
- legacy `ContentPart` remains only as an audited compatibility carrier while direct construction
  paths are migrated through adapters.
- bridge request normalization uses request-side adapter helpers when it must emit legacy
  `ContentPart`, keeping the compatibility carrier at the adapter edge instead of scattering
  dual-use construction through protocol parsing branches.
- provider-owned synthetic request builders should use local request adapter helpers when they must
  emit legacy `ContentPart` for downstream protocol transformers.
- a broader stable non-V4 prompt/content projection should be introduced only after the remaining
  direct `ContentPart` construction paths have a clear migration target; otherwise it would create a
  second public migration surface before the V4 projection has paid off.

### 5. Move Provider-Specific Core Residue

Relocate provider-specific stream bridge logic, custom event serialization, hosted tool factories,
and model defaults to bridge, protocol, provider, or registry crates according to ownership.

### 6. Tighten Compatibility Boundaries

Keep `LlmClient` and generic builder-era paths available only as explicit compatibility surfaces.
Stable family APIs should call native family model paths and should not grow new downcast bridges.

### 7. Narrow Facade And Registry Exports

Reduce broad re-exports to stable, intentional modules. Move compatibility-only names under
`compat` or `experimental` paths, with migration notes where public users are affected.

## Source Guard Ideas

Potential guards:

- `siumai-spec` source test rejects imports from `tokio`, `tokio_util`, `futures`, `reqwest`, and
  `siumai-core`.
- registry boundary tests reject stable family handle execution through `compat_*_client*`.
- core boundary tests reject provider-specific protocol module names under `siumai-core/src`.
- facade boundary tests reject compatibility construction aliases from stable preludes.
- bridge tests ensure provider-specific stream encoders live in bridge/protocol/provider crates.

## Suggested Validation

Use focused validation as slices land:

```text
cargo check -p siumai-spec --no-default-features
cargo nextest run -p siumai-core --no-fail-fast
cargo nextest run -p siumai-bridge --no-default-features --features "openai,anthropic,google" --no-fail-fast
cargo nextest run -p siumai-registry --no-default-features --features "openai,anthropic,google" --no-fail-fast
cargo nextest run -p siumai --no-default-features --features "openai,anthropic,google" --no-fail-fast
```

For provider or protocol moves, also run the affected crate checks:

```text
cargo check -p siumai-protocol-openai --features openai-standard --no-default-features
cargo check -p siumai-provider-openai --features openai --no-default-features
cargo check -p siumai-provider-anthropic --features anthropic --no-default-features
cargo check -p siumai-provider-gemini --features google --no-default-features
```

## Closeout State

Closed on 2026-05-16. `siumai-spec` is now treated as passive data contracts and directional
projection helpers; runtime cancellation, HTTP default policy, provider construction, bridge
customization, and provider-owned protocol behavior live outside the spec crate. `siumai-core`
keeps provider-agnostic runtime contracts and guard coverage rejects provider defaults, model
fixtures, provider-specific stream bridge logic, hosted tool factories, and provider map handling
from returning. The facade and registry surfaces are narrowed to stable family-first imports, with
builder-era and broad catch-all surfaces explicitly classified under `compat` or `experimental`.

The remaining broad removal of dual request/response fields from legacy `ContentPart` is deferred
as a separate compatibility-breaking design after adapter-first migration coverage is wider.
