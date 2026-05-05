# Completion Family Alignment - TODO

Last updated: 2026-04-07

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Workstream goals

- Close the structural AI SDK `completionModel()` gap with a stable Rust family surface.
- Keep completion inside the family-model architecture instead of hiding it inside generic-client
  exceptions.
- Reuse the shared runtime stream contract.
- Align the audited OpenAI-family completion behavior with `repo-ref/ai`.

## Track A - Stable family contracts

- [x] Add `CompletionRequest`.
- [x] Add `CompletionResponse`.
- [x] Keep structured prompt messages on the stable request boundary.
- [x] Carry completion-family `providerOptions` through the normal shared provider-options map.
- [x] Keep request-level HTTP overrides available on completion requests.

## Track B - Core/runtime architecture

- [x] Add `CompletionCapability`.
- [x] Add the stable `CompletionModel` family trait.
- [x] Add `LlmClient::as_completion_capability()`.
- [x] Add `ProviderCapabilities::completion`.
- [x] Reuse `ChatStream` / `ChatStreamEvent` for completion streaming.
- [-] Add a completion-specific stream event enum.
  - rejected in favor of a single shared runtime stream contract

## Track C - Registry and public facade

- [x] Add provider-factory hooks for completion family models.
- [x] Add `ProviderRegistryHandle::completion_model(...)`.
- [x] Add `CompletionModelHandle`.
- [x] Cache completion family models separately in the registry.
- [x] Export the family through `siumai::completion`.
- [x] Cover the public import surface for completion handles and helpers.
- [x] Re-export runtime `StreamRequestOptions` on the stable public family/prelude surface.
  - `siumai::prelude::unified::*`, `siumai::completion::*`, and `siumai::text::*` now all expose
    the same runtime-only stream-request structure used by `ChatRequest` / `CompletionRequest`
  - public-surface compile tests now pin both the trait export (`CompletionCapability`) and the
    explicit request-structure path for `includeRawChunks`

## Track D - OpenAI-family completion parity

- [x] Route non-stream completions to `/completions`.
- [x] Route streamed completions to `/completions` SSE.
- [x] Materialize structured prompts with the audited AI SDK rules.
- [x] Reuse the shared runtime stream lane for completion deltas and terminal responses.
- [x] Emit explicit warnings for unsupported completion options.
- [x] Normalize completion provider options from deprecated, canonical, and provider-owned keys.
- [x] Support streaming `include_usage` on the completion path.
- [x] Emit the deprecated `providerOptions['openai-compatible']` warning on completion responses.
- [x] Add native OpenAI completion generate/stream support on the real `/completions` path.
- [x] Add native Azure OpenAI completion generate/stream support on the real `/completions`
  deployment path with `api-version`.
- [x] Preserve raw completion `choices[0].logprobs` provider metadata on compat/native completion
  responses instead of reusing chat-only metadata extraction.
- [x] Align completion streaming `includeRawChunks` / stable `raw` part ordering with AI SDK.
  - `CompletionRequest.stream_options` now carries runtime-only raw-chunk intent on the stable
    completion request boundary
  - `siumai::completion::StreamOptions.include_raw_chunks` maps to that runtime lane
  - OpenAI-compatible, native OpenAI, and native Azure completion SSE now emit stable
    `stream-start`, `raw`, `response-metadata`, `text-*`, and terminal `finish` parts while
    keeping legacy `ContentDelta` / `StreamEnd`

## Track E - Follow-up audit

- [x] Re-audit `repo-ref/ai` for any non-OpenAI-compatible provider that still exposes
  `completionModel()`.
  - native OpenAI and Azure are complete
  - upstream AI SDK also exposes completion-family access on the audited extra wrappers:
    `deepinfra`, `fireworks`, `togetherai`, and `google-vertex-maas`
  - Siumai already keeps those providers on the completion-family public/runtime path through the
    shared OpenAI-compatible completion factory, matching the current upstream package split
  - the previous `together` compat vs `togetherai` native-rerank split is now collapsed behind
    canonical `togetherai`; the old public `together` alias is retired in favor of canonical
    `togetherai` on both unified and explicit compat builder surfaces
  - AI SDK chat-only compat providers `mistral` and `perplexity` now also have explicit
    public-path guard coverage: canonical builder/provider/config clients do not expose
    completion capability, and registry `completion_model(...)` rejects those ids directly
- [x] Decide whether to add broader completion public-path fixtures beyond the current registry,
  provider, and public-surface coverage.
  - native completion public-path parity now explicitly covers `openai` and `azure`
  - first-class wrapper completion public-path parity now explicitly covers canonical
    `togetherai`, `deepinfra`, `google-vertex-maas` (`vertex-maas`), and `fireworks`
  - audited completion stream request fixtures now also pin runtime-only `includeRawChunks`
    behavior so raw-chunk intent stays off the provider wire payload on both native and compat
    wrapper completion routes
  - lower-contract OpenAI-compatible URL coverage now explicitly locks the canonical
    `/completions` endpoint for every audited completion-capable compat wrapper:
    `fireworks`, `togetherai`, `deepinfra`, and `vertex-maas`
  - TogetherAI and DeepInfra now also pin their canonical chat/embedding/completion endpoints at
    the compat-spec layer instead of relying only on higher-level public-path parity coverage
  - Vertex MaaS lower-contract URL coverage now uses the real derived
    `/projects/{project}/locations/{location}/endpoints/openapi` base URL instead of the config
    placeholder base URL
