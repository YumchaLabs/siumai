# Fearless Refactor V4 - Focused Wrapper Contract

Last updated: 2026-03-11

## Purpose

This document defines the contract for a **focused wrapper provider**.

A focused wrapper provider is a provider-owned package that:

- exposes provider-owned `Config` / `Client` entry types
- may reuse a shared runtime internally
- intentionally supports only a subset of family capabilities
- must not leak deferred capabilities back through wrapper-level traits

Current examples:

- `deepseek`
- `xai`
- `groq`

For package-tier decisions, see `provider-package-alignment.md`.
For the public entry ordering, see `public-api-story.md`.

## Core rule

If a provider-owned wrapper intentionally supports only part of the family surface,
the wrapper must be **narrow by construction, narrow by metadata, narrow by factory,
and narrow by tests**.

That means:

1. supported capabilities are explicit
2. unsupported capabilities fail fast
3. wrapper-level trait exposure matches the declared capability set
4. public-path parity confirms no accidental fallback to shared compat behavior

## When to use this contract

Apply this contract when all of the following are true:

- the provider deserves provider-owned `Config` / `Client` types
- the provider reuses a shared runtime or shared protocol layer internally
- only a subset of family capabilities is intentionally in scope today
- promoting every compat/runtime capability into the native story would create fake symmetry

Do **not** use this contract for:

- full provider-owned packages that intentionally expose a broad family surface
- compat vendor views that should remain under `openai_compatible`
- temporary spikes that are not yet part of the public story

## Required invariants

### 1. Metadata invariant

`native_provider_metadata` must declare only the capabilities the focused wrapper actually owns.

Examples:

- `xai`: `chat + streaming + tools + vision + speech`
- `groq`: `chat + streaming + tools + audio`
- `deepseek`: `chat + streaming + tools + vision + thinking`

If metadata says a capability is deferred, the wrapper must not surface it elsewhere.

### 2. Factory invariant

The native provider factory must:

- materialize the provider-owned wrapper client on supported paths
- reject unsupported family paths with `UnsupportedOperation`
- avoid delegating unsupported family builders to generic text/compat clients

Typical reject points:

- `embedding_model_with_ctx(...)`
- `image_model_with_ctx(...)`
- any future focused family path not intentionally supported

### 3. Wrapper client invariant

The provider-owned wrapper client must align with the same capability boundary.

Required behavior:

- `capabilities()` returns only the native capability set
- `as_*_capability()` returns `None` for deferred capabilities
- provider-owned methods for deferred capabilities fail fast instead of silently delegating

This is important even if the shared runtime technically knows how to send a request.

The public wrapper contract is authoritative.

### 4. Public-path invariant

Top-level construction paths must preserve the same boundary:

- `Siumai::builder()`
- `Provider::*()`
- config-first `*Client::from_config(...)`

For deferred capabilities:

- operations must fail with `UnsupportedOperation`
- no HTTP request should be emitted
- wrapper clients should not report the deferred capability through trait downcasts

### 5. Documentation invariant

Docs, examples, and matrices must state the boundary explicitly.

Required documentation effects:

- matrix row mentions the boundary decision
- workstream TODO/milestone notes mention the decision
- examples do not imply the deferred capability is part of the native wrapper story

## Checklist

When promoting or auditing a focused wrapper provider, verify all items below.

### Surface

- [ ] provider-owned `Config` / `Client` exist
- [ ] builder paths converge to config-first construction
- [ ] public wrapper naming stays provider-owned

### Capability boundary

- [ ] `native_provider_metadata` matches intended capability scope
- [ ] wrapper `capabilities()` matches metadata
- [ ] unsupported `as_*_capability()` accessors return `None`
- [ ] unsupported direct methods fail fast with `UnsupportedOperation`

### Factory

- [ ] supported family paths materialize provider-owned clients/models
- [ ] unsupported family paths reject explicitly
- [ ] no generic fallback reintroduces an unsupported capability

### Tests

- [ ] registry contract test locks declared capability set
- [ ] registry contract test locks provider-owned wrapper materialization
- [ ] registry contract test locks unsupported family rejection
- [ ] top-level public-path test locks fail-fast behavior
- [ ] top-level public-path test locks “no request emitted”
- [ ] top-level public-path test locks “no unsupported capability leak”

### Docs

- [ ] provider feature matrix mentions the boundary
- [ ] TODO mentions the boundary
- [ ] milestones mention the boundary
- [ ] examples stay within the supported native story

## Preferred test pattern

Use shared helper assertions instead of re-encoding negative checks per provider.

Current helper locations:

- `siumai/tests/provider_public_path_parity_test.rs`
- `siumai-registry/src/registry/factories/contract_tests.rs`

The preferred assertion shape is:

1. assert the operation fails with `UnsupportedOperation`
2. assert no capture transport request was emitted
3. assert wrapper clients do not expose deferred capabilities through `as_*_capability()`

## Non-goals

This contract does **not** require:

- inventing provider-owned metadata/resources just for symmetry
- promoting all compat/runtime features into the native wrapper story
- splitting every wrapper into a full provider package immediately
- hiding the fact that the implementation may reuse shared compat internals

## Decision heuristic

If a provider-specific package starts accumulating:

- stable native family routes
- stable typed metadata
- stable typed resources/tools
- meaningful divergence across multiple capability families

then it may be time to move from **focused wrapper** toward a fuller provider-owned package.

Until then, keep the boundary explicit and narrow.
