# Completion Family Alignment - Milestones

Last updated: 2026-04-07

This workstream tracks the completion-family refactor as a focused structural alignment pass.

## CFA-M0 - Scope locked

Acceptance criteria:

- The AI SDK completion references are identified.
- The missing Rust completion-family architecture gap is explicitly documented.
- The workstream keeps Rust-first naming and layering.

Status: completed

## CFA-M1 - Stable completion family exists

Acceptance criteria:

- `siumai-spec` exposes dedicated completion request/response types.
- The request shape can carry structured prompt messages plus provider options.
- The response shape can carry text, finish reason, usage, warnings, and metadata.

Current state:

- `CompletionRequest` and `CompletionResponse` are now stable public types.
- Completion requests keep structured prompt messages and share the existing provider-options
  boundary.

Status: completed

## CFA-M2 - Core/runtime family surface exists

Acceptance criteria:

- `siumai-core` exposes a completion capability trait.
- Completion has a family model contract instead of only a generic-client hook.
- Runtime streaming reuses the shared semantic stream carrier.

Current state:

- `CompletionCapability`, `CompletionModelV3`, and `CompletionModel` now exist.
- `LlmClient` and `ClientWrapper` can forward completion capability.
- Completion streaming reuses `ChatStream` / `ChatStreamEvent`.

Status: completed

## CFA-M3 - Registry and facade expose completion natively

Acceptance criteria:

- Provider factories can build completion-family models.
- Registry handles can resolve `"provider:model"` completion paths.
- The main facade exposes family-oriented completion helpers.

Current state:

- `ProviderFactory::completion_model_family*` now exists.
- `ProviderRegistryHandle::completion_model(...)` and `CompletionModelHandle` are public.
- `siumai::completion::{complete, stream, stream_with_cancel}` is exported.
- `siumai::prelude::unified::*` now also re-exports `CompletionCapability`, so completion-family
  trait imports line up with the other stable model-family capability traits.
- The stable runtime stream-request data structure is also now re-exported on the normal facade:
  `StreamRequestOptions` is available from `siumai::prelude::unified::*`,
  `siumai::completion::*`, and `siumai::text::*`, which keeps explicit
  `includeRawChunks` request construction on the public family surface instead of hiding it in
  internal crates.

Status: completed

## CFA-M4 - OpenAI-family provider parity lands

Acceptance criteria:

- Non-stream completion uses the real `/completions` route on the audited OpenAI-family paths.
- Stream completion uses the real `/completions` SSE route on the audited OpenAI-family paths.
- Prompt materialization and unsupported warnings match the audited AI SDK behavior closely.

Current state:

- OpenAI-compatible completion generate/stream now routes through `/completions`.
- Native OpenAI completion generate/stream now routes through the real `/completions` path.
- Native Azure OpenAI completion generate/stream now routes through Azure `/completions` with
  deployment URL plus `api-version`.
- Structured prompts are materialized with the audited AI SDK rules across the audited
  OpenAI-family providers.
- Unsupported `topK`, `tools`, `toolChoice`, and structured `responseFormat` now emit warnings.
- Streaming `include_usage`, provider-option normalization, and raw completion `logprobs`
  metadata are wired through the completion path.
- Completion streams now also honor runtime-only `includeRawChunks` and emit AI SDK-style stable
  part ordering on the audited OpenAI-family paths:
  `stream-start -> raw -> response-metadata -> text-start -> text-delta ... -> text-end ->
  finish`, while preserving legacy `ContentDelta` / `StreamEnd`.

Status: completed

## CFA-M5 - Documentation and release notes catch up

Acceptance criteria:

- Structural alignment docs no longer show completion as a Red gap.
- The completion workstream is documented in `docs/workstreams`.
- Root and crate changelogs mention the new family surface.

Status: completed

## CFA-M6 - Broader provider audit

Acceptance criteria:

- We know whether any additional providers should expose completion-family support.
- Remaining completion work is framed as provider-surface policy or fixture depth, not missing
  architecture.

Current state:

- Native OpenAI and Azure are complete on the dedicated provider path.
- OpenAI-compatible presets that advertise chat-family access, including Fireworks and the
  TogetherAI-style compat surface, also inherit completion-family support through the shared
  compat factory.
- The upstream AI SDK provider set also exposes completion-family access on the audited extra
  wrappers `deepinfra`, `fireworks`, `togetherai`, and `google-vertex-maas`, and Siumai already
  keeps those providers on the same public/runtime completion lane through the shared compat
  factory.
- The AI SDK-style TogetherAI naming split is now closed: canonical `togetherai` is a unified
  provider surface for chat/completion/embedding/image/audio plus native rerank, while `together`
  remains a compat alias.
- The inverse boundary is now also pinned on the public paths: AI SDK chat-only compat providers
  `mistral` and `perplexity` keep completion disabled across canonical
  `Siumai::builder()` / `Provider::*()` / config-first clients, and registry
  `completion_model(...)` rejects those ids.
- Public-path parity coverage now also locks completion generate/stream request equivalence across
  builder/provider/config/registry surfaces for native `openai` and `azure`, plus the audited
  first-class wrapper surfaces `togetherai`, `deepinfra`, `vertex-maas`, and `fireworks`,
  including the runtime-only `includeRawChunks` no-wire guarantee on streamed completion
  requests.
- Lower-contract URL alignment now also has explicit chat/embedding/completion audit helpers for
  every audited completion-capable compat wrapper: Fireworks, TogetherAI, DeepInfra, and Vertex
  MaaS.
- Vertex MaaS lower-contract coverage now locks the real project/location-derived
  `/endpoints/openapi` base URL instead of the static config placeholder URL.

Status: completed
