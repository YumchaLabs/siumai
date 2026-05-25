# Ollama Unified Provider Surface - Design

Last updated: 2026-05-25

## Problem

Ollama already had a provider-owned wrapper path, but the package shape still had one structural
collision:

- `providers/ollama/models.rs` meant runtime model-listing behavior, not model constants
- the public facade/catalog/default-model helper therefore could not reuse a consistently named
  provider-owned `models` constant surface
- provider catalog and `get_default_models()` still depended on a separate handwritten list

That made Ollama the odd provider out: the public Rust facade wanted `models` to mean curated
model ids, while the provider crate used the same name for a runtime capability type.

## Implemented design

### 1. Runtime listing moved out of the `models` name

The runtime model-listing implementation now lives in:

- `providers/ollama/model_listing.rs`

This keeps the behavior intact while freeing `models.rs` for the public model-constant story.

### 2. `models.rs` now means curated provider-owned model constants

`siumai-provider-ollama` now exposes a curated provider-owned surface in:

- `providers/ollama/models.rs`

with explicit public families:

- `chat`
- `embedding`

plus:

- `CHAT`
- `EMBEDDING`
- `ALL_CHAT`
- `ALL_EMBEDDING`
- `all_models()`

The older `model_constants.rs` compatibility layer has been removed. `models.rs` is now the only
provider-local model catalog for Ollama.

### 3. Facade, registry, and defaults now share one source

The curated Ollama model surface now feeds:

- `provider_ext::ollama::{chat, embedding, model_sets}`
- provider catalog supported-model output
- `get_default_models()`

So the provider package, facade, and registry now describe the same curated Ollama subset instead
of maintaining a second handwritten list.

## Validation

The implemented surface is locked by:

- provider-local Ollama package tests
- public-surface compile guards on `provider_ext::ollama`
- registry catalog tests for native Ollama metadata/model output

## Closeout

The deferred follow-up is complete. Future changes should update `providers/ollama/models.rs`
directly whenever the supported default subset intentionally changes.
