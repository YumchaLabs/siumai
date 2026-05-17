# Fearless Refactor V4 - Handoff

Last updated: 2026-05-17

## Current State

The workstream remains active. The public V4 story is already mostly aligned around
family-model execution, registry-first app code, config-first provider code, and explicit
compatibility surfaces.

The OpenAI-compatible internal boundary cleanup lane is now closed:

- Provider model catalog split is complete and committed.
- `OpenAiCompatibleClient` production shell has been split into focused submodules:
  - `openai_client/types.rs`
  - `openai_client/runtime.rs`
  - `openai_client/compatibility.rs`
- The parent OpenAI-compatible client regression suite has been moved to `openai_client/tests.rs`,
  leaving `openai_client.rs` as a thin shell plus `mod tests;`.
- The completion runtime is now a directory module:
  - `openai_client/completion/mod.rs` owns completion request/response orchestration.
  - `openai_client/completion/streaming.rs` owns the SSE stream state and converter.
  - `openai_client/completion/tests.rs` owns completion no-network contracts and source guards.
- Repeated simple provider settings adapters now share `simple_compat_provider_settings!`; only
  generic OpenAI-compatible settings, Vertex MaaS settings, and Alibaba settings stay hand-written
  because they own different construction behavior. Provider settings contract tests now live in
  `settings/tests.rs`.
- Provider family default-model data has been split into `config/family_defaults.rs`; `config.rs`
  remains the compatibility lookup facade used by runtime/default-model callers.
- Built-in provider registry data has been split into `config/builtin_providers.rs`; `config.rs`
  still owns generic provider config construction and lookup/capability facade functions.
- OpenAI-compatible builder reasoning defaults have been split into `builder/reasoning.rs`; the
  parent `builder.rs` shell now keeps construction, config convergence, HTTP wiring, and build
  orchestration instead of owning provider-specific thinking/reasoning parameter mapping.
- OpenAI-compatible builder tests now live in `builder/tests.rs`, leaving the parent builder shell
  focused on construction and config convergence.
- OpenAI-compatible request-option extension tests now live in `ext/request_options/tests.rs`,
  leaving `ext/request_options.rs` focused on public extension traits and provider-options map
  merging.
- OpenAI-compatible settings tests now live in `settings/tests.rs`, leaving `settings.rs` focused on
  settings construction code and the simple-settings macro.
- Existing capability modules remain execution owners:
  - `chat`
  - `completion`
  - `embedding`
  - `image`
  - `audio`
  - `rerank`
  - `models`
- Closeout audit result: the remaining large OpenAI-compatible files are intentional
  capability-local implementations, static provider data, or public facades. Further splitting would
  mostly move data without improving ownership or reducing runtime coupling.

## Next Step

Stop Track J. Open a new follow-on only when there is a concrete behavior bug, public API drift, or
real ownership/coupling problem. The config facade is now thin enough that further config splitting
should happen only if new provider families need dedicated registry ownership.

## Guardrails

- Keep public paths and behavior compatible.
- Keep `LlmClient` as a compatibility bridge, not the architectural center.
- Keep capability execution logic in capability modules.
- Keep provider catalog data in provider-family catalog modules.
- Keep completion SSE state and event conversion in `openai_client/completion/streaming.rs`.
- Keep simple provider settings adapters on the macro path unless the provider has real divergent
  construction behavior.
- Keep provider settings contract tests in `settings/tests.rs`; `settings.rs` should stay focused
  on settings construction code.
- Keep provider family defaults in `config/family_defaults.rs`; `config.rs` should remain a lookup
  facade rather than a mixed data owner.
- Keep built-in provider registry data in `config/builtin_providers.rs`; `config.rs` should not own
  `build_builtin_providers` or the static built-in provider map.
- Keep provider-specific builder reasoning defaults in `builder/reasoning.rs`; `builder.rs` should
  remain a construction shell rather than another provider-option mapping table.
- Keep builder contract tests in `builder/tests.rs`; `builder.rs` should expose the construction
  interface rather than carrying the full parity suite inline.
- Keep request-option contract tests in `ext/request_options/tests.rs`; `ext/request_options.rs`
  should remain the public extension-trait shell.
- Prefer focused no-network tests and source guards over broad rewrites.
- Do not split OpenAI-compatible modules solely because a file is large; require a clear ownership,
  coupling, or test-locality improvement.

## Residual Risks

- Provider-family catalog tables and built-in registry data will still grow as vendors are added;
  split only when a provider family gains independent ownership, not by row count.
- Capability executors remain provider-specific by nature. Treat their size as acceptable while the
  module owns one request/response protocol boundary and has focused no-network coverage.
- Future public-path bugs should be fixed through the same config-first / family-model contract
  path, not by adding another builder-only adapter branch.

## Validation Commands

- `cargo fmt -p siumai-provider-openai-compatible`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast builder`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast request_options`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast completion`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast settings`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast config`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast openai_client`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
- `git diff --check`
