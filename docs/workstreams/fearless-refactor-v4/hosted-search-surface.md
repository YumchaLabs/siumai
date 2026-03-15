# Fearless Refactor V4 - Hosted Search Surface

Last updated: 2026-03-07

## Purpose

This document defines how Siumai V4 should model **provider-hosted search** and adjacent
provider-native search-like request knobs.

The key question is:

- should V4 introduce a new Stable unified hosted-search request surface now, or
- should hosted search remain under provider-owned extension APIs for this cycle?

## Short decision

**Decision for V4:** do **not** add a new Stable unified hosted-search request surface yet.

For V4, hosted-search behavior should remain in the **Extension** layer:

- `siumai::provider_ext::perplexity::*`
- `siumai::provider_ext::openrouter::*`
- future provider-owned hosted-search modules for providers such as OpenAI, xAI, or Google

The Stable family-level surface stays focused on:

- `ChatRequest`
- `Tool` / `ToolChoice`
- provider-agnostic response parts
- provider-agnostic execution families such as `text::generate` / `text::stream`

Raw `with_provider_option(...)` remains the low-level compatibility escape hatch, but common
provider-native search knobs should prefer typed helpers under `provider_ext::<provider>`.

## Why not unify it now?

Because the current ecosystem does **not** have one coherent hosted-search shape.

### 1. Request semantics do not line up cleanly

Examples already visible in this repo:

- Perplexity exposes search policy knobs such as `search_mode`, recency filters,
  image-return toggles, and nested `web_search_options`.
- OpenRouter currently exposes adjacent vendor knobs such as `transforms`, which are
  provider request-shaping controls, not a true hosted-search contract.
- xAI uses a different search configuration story.
- OpenAI and Google increasingly surface search/grounding through other provider-native APIs
  or hosted tool concepts rather than one plain request object.

These are related capabilities, but they are not yet one stable abstraction.

### 2. Response semantics also differ

A stable surface should ideally define not only request knobs, but also the resulting output story:

- citations
- sources
- search metadata
- result provenance
- images or related questions
- grounding diagnostics

Today, those outputs vary significantly by provider and often arrive through provider-specific
metadata or provider-defined content parts.

### 3. Forced unification would freeze the wrong shape

If V4 adds a Stable hosted-search request type too early, we risk:

- encoding one provider's terminology as the cross-provider model
- creating a surface that still needs provider-specific escape hatches for most real use cases
- making later corrections more expensive because the surface would already be treated as Stable

V4 should prefer a smaller correct Stable surface over a larger premature one.

## What should be Stable vs Extension?

### Stable

Stable should include only semantics that already hold across providers with low ambiguity:

- ordinary tool calling through `Tool` and `ToolChoice`
- provider-agnostic prompt/message structure
- structured output semantics that map cleanly enough across providers
- provider-agnostic response content parts

### Extension

Hosted search should stay in `provider_ext::<provider>` when any of the following is true:

- the request knobs are provider-specific
- the response metadata is provider-specific
- the provider executes search internally rather than as a normal tool loop
- the provider combines search with answer packaging in a non-portable way

That is the current state for OpenRouter and Perplexity.

### Compatibility-only

The raw fallback remains:

- `ChatRequest::with_provider_option("provider", json!(...))`

This stays useful for:

- newly released provider flags
- experimental preview parameters
- downstream migration periods before typed helpers exist

But it should not be the preferred public story once a provider-owned typed helper exists.

## Current V4 recommendation by provider family

### Perplexity

Recommended V4 public path:

- `siumai::provider_ext::perplexity::PerplexityOptions`
- `siumai::provider_ext::perplexity::PerplexityChatRequestExt`

Why:

- the search knobs are request-level and provider-specific
- the semantics are useful and common enough to deserve typed helpers
- they are still too Perplexity-shaped to promote into Stable V4

### OpenRouter

Recommended V4 public path:

- `siumai::provider_ext::openrouter::OpenRouterOptions`
- `siumai::provider_ext::openrouter::OpenRouterChatRequestExt`

Why:

- current OpenRouter request knobs in this repo are vendor-routing / transform controls
- they are not a hosted-search abstraction
- they belong in provider-owned typed extensions rather than the Stable family surface

### Future providers

For OpenAI, xAI, Google, and others:

- if the provider exposes real hosted-tool builders that align with an existing hosted-tool model,
  they can live under provider-owned hosted tool modules
- if the provider exposes request-only search knobs, they should start as provider-owned typed options
- promotion to Stable should happen only after several providers converge on the same semantics

## Graduation criteria for a future Stable hosted-search surface

A future Stable hosted-search surface should only ship once all of the following are true:

1. **At least three providers** expose materially similar request semantics.
2. **Request semantics converge** on a shared minimum shape that is not provider-name cargo culting.
3. **Response semantics converge** enough to define stable output expectations.
4. **Provider-specific escape hatches become optional**, not mandatory for common use.
5. **No-network parity tests** can lock both request mapping and final transport-boundary behavior.
6. **Public examples remain honest**, meaning the Stable path is truly the recommended path.

If those criteria are not met, the feature belongs in the Extension layer.

## What V4 should do next

### Do now

- keep hosted-search semantics in provider-owned typed request helpers
- continue replacing raw `with_provider_option(...)` usage with typed provider-owned helpers where usage is common
- add no-network request-body guards for those typed helpers
- document the Extension-vs-Stable boundary clearly

### Do not do now

- do not add `ChatRequest.hosted_search`
- do not add a generic `with_search(...)` builder on the Stable family surface
- do not force OpenRouter transforms and Perplexity search knobs into one cross-provider struct
- do not claim provider-native search is part of the Stable tool-loop contract yet

## Example public story

Preferred V4 story:

```rust
use siumai::prelude::unified::*;
use siumai::provider_ext::perplexity::{
    PerplexityChatRequestExt, PerplexityOptions, PerplexitySearchMode,
};

let request = ChatRequest::new(vec![ChatMessage::user("latest news").build()])
    .with_perplexity_options(
        PerplexityOptions::new().with_search_mode(PerplexitySearchMode::Academic),
    );
```

This keeps:

- `ChatRequest` provider-agnostic
- search semantics provider-owned
- migration away from raw provider-option maps straightforward

## Final recommendation

For V4, the correct architecture is:

- **Stable**: keep hosted search out of the family-level request contract
- **Extension**: use provider-owned typed APIs for OpenRouter / Perplexity / future providers
- **Compatibility**: retain raw provider options as the last-resort escape hatch

That gives us a cleaner public story now without freezing the wrong abstraction too early.
