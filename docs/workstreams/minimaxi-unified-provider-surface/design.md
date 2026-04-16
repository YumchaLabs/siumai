# MiniMaxi Unified Provider Surface - Design

Last updated: 2026-04-10

## Problem

MiniMaxi already had a provider-owned wrapper story, but two structural drifts were still
visible:

- the public facade and provider catalog did not share a provider-owned curated model source
- catalog output still hand-listed a partial subset and omitted image models that the provider
  surface already supports
- stream metadata rekeying only normalized final `ChatResponse` / `StreamEnd`, while stable
  stream-part finish metadata could still leak the borrowed `anthropic` provider root

That meant the package shape was close, but the important model/data-structure boundary still had
multiple competing sources of truth.

## Implemented design

### 1. Keep the provider wrapper, add one curated model surface

`siumai-provider-minimaxi` now exposes provider-owned curated model-family constants in:

- `providers/minimaxi/models.rs`

The new surface keeps the public families explicit:

- `chat`
- `speech`
- `video`
- `music`
- `image`

plus top-level defaults and grouped lists:

- `CHAT`
- `SPEECH`
- `VIDEO`
- `MUSIC`
- `IMAGE`
- `ALL_*`
- `all_models()`

The older `model_constants.rs` remains as the broader compatibility layer instead of the canonical
catalog/facade source.

### 2. Facade and registry now reuse that same source

`provider_ext::minimaxi` now exposes:

- `models::{chat, speech, video, music, image, model_sets}`

and the registry catalog now builds MiniMaxi supported-model output from the same provider-owned
constants instead of a second handwritten list.

This also fixes the previous image-family omission in provider catalog output.

### 3. Stable stream metadata rekeying now includes typed stream parts

The MiniMaxi Anthropic-derived stream wrapper now normalizes provider metadata across:

- raw custom events
- typed stream parts
- final `StreamEnd`

So finish-part metadata no longer leaks `providerMetadata["anthropic"]` when the public provider
identity is `minimaxi`.

## Validation

The implemented surface is locked by:

- provider-local MiniMaxi package tests
- public-surface compile guards on `provider_ext::minimaxi`
- registry catalog tests for native MiniMaxi metadata/model output
- focused stream metadata rekey tests for typed finish parts and final `StreamEnd`

## Remaining follow-up

- Revisit the curated MiniMaxi model subset if the provider grows more first-class public family
  ids.
- Decide later whether the broader legacy `model_constants.rs` surface should be slimmed down once
  downstream migration pressure is low enough.
