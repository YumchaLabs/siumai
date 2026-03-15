# Fearless Refactor V4 - Public API Story

Last updated: 2026-03-11

## Purpose

This document defines the recommended public-facing story for Siumai V4.

It exists to keep README guidance, example structure, migration notes, and provider-specific escape hatches aligned
around one simple rule:

```text
registry-first -> config-first -> builder convenience
```

The implementation may continue to converge internally, but the public story should already be stable enough for users
to learn once and reuse across providers.

## Surface ladder

### 1. Registry-first

Use this for application code, multi-provider routing, and default usage.

Recommended entry points:

- `registry::global().language_model("provider:model")?`
- `text::generate`, `text::stream`
- `embedding::embed`
- `image::generate`
- `rerank::rerank`
- `speech::synthesize`
- `transcription::transcribe`

Why this is first:

- it is the most stable app-level abstraction
- it matches the family-model-centered architecture
- it keeps provider selection and provider-specific translation out of application code

### 2. Config-first

Use this for provider-specific setup, tests, infrastructure wiring, and direct access to provider-owned construction.

Recommended entry points:

- `siumai::providers::<provider>::*Config`
- `siumai::providers::<provider>::*Client::from_config(...)`
- provider-owned config helpers such as `with_model(...)`, transport settings, retry settings, and provider-local auth setup

Why this is second:

- provider crates are where provider complexity belongs
- config-first construction is explicit and testable
- builders should converge to this path internally rather than compete with it

### 3. Builder convenience

Use this for quick starts, migration, side-by-side comparisons, and compatibility-preserving ergonomics.

Allowed surfaces:

- `siumai::compat::Siumai`
- `Provider::*()` builders
- builder chains that end in the same provider-owned config/client path used by config-first construction

Policy:

- builders are convenience, not architecture
- no important capability should be reachable only through builders
- examples using builders should be labeled as convenience or compatibility-oriented

## Provider-specific escape hatches

### Preferred: typed provider extensions

Provider-specific request and response details should prefer typed surfaces under `provider_ext::<provider>`.

Examples:

- provider-specific typed request options
- request extension traits for attaching provider options
- typed response metadata helpers

Why this is preferred:

- it keeps provider complexity explicit but discoverable
- it avoids leaking raw provider-option maps into otherwise stable code
- it lets wrapper providers keep Rust-first naming while still converging on AI SDK-like behavior internally

### Fallback: raw provider options

Raw `with_provider_option(...)` style escape hatches should remain available for unsupported vendor knobs,
but they are not the preferred first stop in docs when a typed provider extension exists.

## `compat` policy

`compat` is a compatibility surface.

That means:

- it may remain available for migration and ergonomic continuity
- it should be documented as temporary or compatibility-oriented
- its documented removal target is no earlier than `0.12.0`
- it should not receive builder-only architectural features
- new capabilities should land on family APIs, config-first clients, or `provider_ext` first; `compat`
  may mirror those features later but should not be their only public home
- compatibility imports should stay explicit (`siumai::compat::*` or `siumai::prelude::compat::*`)
- when possible, docs should import it explicitly so users understand they are opting into compatibility behavior

## Documentation rules

To keep the public story coherent, docs and examples should follow these rules:

1. Show registry-first before config-first.
2. Show config-first before builder convenience.
3. Keep provider-specific knobs near provider-owned types and `provider_ext`.
4. Label builder examples as convenience or compatibility demos.
5. Prefer typed provider extensions over raw provider-option maps in docs.
6. If a wrapper provider reuses a shared runtime internally, keep the public entry types provider-owned.

For the negative-capability rules on focused wrapper packages, see `focused-wrapper-contract.md`.

## Provider Package Tiers In Docs

Docs and examples should describe provider surfaces using three tiers:

- full provider package: provider-owned `Config` / `Client` plus typed extension surface
- focused provider package: only the provider-specific capabilities we intentionally support
- compat vendor view: typed vendor helpers layered on the shared OpenAI-compatible runtime

Documentation policy by tier:

- full packages should show config-first examples under `provider_ext::<provider>`
- focused packages should stay narrow and should not invent fake symmetry with OpenAI or Anthropic
- compat vendor views such as `openrouter` and `perplexity` should be documented as typed vendor views over `OpenAiCompatibleClient`, not as proof that every compat preset needs a dedicated client package
- compat presets such as `siliconflow`, `together`, and `fireworks` should remain in the OpenAI-compatible story unless they accumulate clearly provider-owned public semantics worth promotion

Focused wrapper packages should also follow the checklist in `focused-wrapper-contract.md`.
Compat vendor views should follow the checklist in `compat-vendor-view-contract.md`.

## Relation to the AI SDK reference

Siumai should continue to borrow the AI SDK's implementation layering ideas:

- provider-owned request shaping
- thin convenience builders
- typed provider escape hatches
- shared runtime helpers beneath the provider boundary

But Siumai should not mechanically mirror AI SDK public naming.

The public contract remains Rust-first:

- family-oriented top-level functions across `text`, `embedding`, `image`, `rerank`, `speech`, and `transcription`
- provider-owned config/client types
- explicit `provider_ext` escape hatches
- `compat` clearly separated from the preferred default path

## What still counts as incomplete

This document does not mean every provider is already perfectly converged internally.

V4 can still be “public-story complete�?before every secondary provider finishes internal cleanup, as long as:

- the recommended construction order is consistent
- the same role is assigned to the same public surface everywhere
- compatibility surfaces are clearly labeled
- new provider work follows this ladder instead of inventing new public entry points
