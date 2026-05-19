# Fearless Module Deepening — Handoff

Status: Complete
Last updated: 2026-05-19

## Current State

This workstream was opened from an architecture review of the current Siumai crate layout against
the local Vercel AI SDK reference under `repo-ref/ai`.

FMD-010 is complete: the docs capture the deepening candidates and chose provider-defined tool
catalog extraction as the first executable proof slice.

FMD-020 is complete: canonical provider-defined hosted-tool catalogs were moved out of
`siumai-spec`/`siumai-core` and into protocol/provider-owned modules. `siumai::tools::*` remains as
the compatibility facade for downstream imports.

FMD-030 is complete: core and registry now use provider ids as the primary identity. Closed
`ProviderType` remains only as compatibility classification behind `siumai-registry::provider::legacy`
or passive spec compatibility docs/tests.

FMD-040 is complete: `LlmClient` and `ClientWrapper` were physically moved under
`siumai_core::compat::client`, with `siumai_core::client` kept as a migration alias. Registry and
facade generic-client imports now use explicit `compat::client` paths; the old
`siumai_registry::LlmClient` root alias has migration docs and guards. Stable family registry
handles remain unchanged and still avoid compat/downcast paths for primary execution.

FMD-050 is complete: production built-in factories no longer call the old broad
`registry::factory::build_*_client(...)` generic-client helpers. OpenAI and Anthropic compatibility
paths now use provider-owned private typed builders that are shared with native family methods.
The remaining public `registry::factory::build_*_client(...)` functions are documented and marked as
deprecated compatibility shims; all-provider registry validation now includes Google Vertex so the
Google Vertex xAI factory is covered by the advertised gate.

FMD-060 is complete: Gemini GenerateContent request normalization was moved out of the monolithic
`siumai-bridge/src/request/normalize.rs` body and into the narrow
`siumai-bridge/src/request/normalize/gemini_generate_content.rs` adapter. The public Gemini bridge
wrappers remain unchanged and delegate to the adapter; source guards now prevent Gemini typed parser
policy from returning to the monolith.

FMD-070 is complete: OpenAI Responses internals were deepened around three concrete behavior seams.
Responses request base body construction and typed option post-processing now live behind
`siumai-protocol-openai/src/standards/openai/transformers/request/responses/responses_request_builder.rs`.
Responses hosted/dynamic output helpers now live behind
`siumai-protocol-openai/src/standards/openai/transformers/response/responses/hosted_tools.rs`.
Responses provider metadata/source/logprobs aggregation now lives behind
`siumai-protocol-openai/src/standards/openai/transformers/response/responses/metadata.rs`.
Source guards prevent those policies from returning to the monolithic transformer files, and
protocol plus facade OpenAI fixture gates passed.

FMD-080 is complete: the legacy `ContentPart` decision is recorded in
`FMD-080-content-part-directional-boundary-decision.md`. This lane will **not** open the breaking
namespace-move child workstream yet. Instead, FMD-090 should land one non-breaking request-side
adapter proof slice by extracting bridge request normalization's legacy `ContentPart` construction
helpers into a narrow `siumai-bridge/src/request/legacy_content.rs` module. The public legacy
carrier remains available at existing paths.

FMD-090 is complete: request-side legacy `ContentPart` construction was extracted into
`siumai-bridge/src/request/legacy_content.rs`. OpenAI Chat Completions, OpenAI Responses,
Anthropic Messages, and Gemini GenerateContent request normalization now route request part
construction through that adapter; source guards prevent the helper definitions and request-side
`provider_metadata` population from drifting back into the monolithic request normalizer. The
public legacy `ContentPart` carrier was not renamed or removed.

FMD-100 is complete: the 47k-line `provider_public_path_parity_test.rs` monolith was split into a
shared harness plus provider-local modules under `siumai/tests/provider_public_path_parity/`.
Registry architecture guards now require the provider-local split and inspect those modules through
a manifest. The public surface import guard remains broad by design.

FMD-110 is complete: fresh closeout verification passed, the workstream docs are marked complete,
and remaining breaking work has been split by decision rather than left implicit.

## Active Task

- Task ID: none
- Status: CLOSED
- Evidence: see `EVIDENCE_AND_GATES.md` section
  `2026-05-19 — FMD-110 closeout verification`.

## Decisions Since Last Update

- Open a new lane instead of reusing broad historical workstreams because the relevant lanes are
  closed/superseded and this review produced a new set of concrete Module-deepening tasks.
- Use FMD-020 as the first coding task because provider-defined tool catalog ownership is a small,
  independently testable spec/provider residue fix.
- Keep legacy `ContentPart` work later in the lane and allow splitting it into a child workstream
  because it is likely a breaking public-shape slice.
- Complete FMD-010 after verifying the required planning files exist and the JSON task pointer is
  parseable.
- Complete FMD-020 by moving hosted-tool catalog ownership to protocol/provider crates and keeping
  only compatibility re-exports at `siumai::tools::*`.
- Keep passive `ProviderDefinedTool` and `Tool::ProviderDefined` data shapes in `siumai-spec`, but
  remove catalog lookup helpers from spec data carriers.
- Include Groq and xAI provider crates in the FMD-020 proof because the old spec catalog owned their
  provider facts.
- Complete FMD-030 by removing `provider_type()` from `LlmClient`/`ClientWrapper`, adding
  provider-id-first core validation/report APIs, making registry provider catalog lookup
  provider-id-first, and documenting `ProviderType` as legacy compatibility classification.
- Keep public `ProviderType`, `ProviderInfo::provider_type`, and `ProviderMetadata::provider_type`
  for compatibility. New primary code should use provider ids and only derive `ProviderType` inside
  explicit compatibility seams.
- Complete FMD-040 by moving the physical generic-client implementation to
  `siumai_core::compat::client`, keeping `siumai_core::client` as a lower-level migration alias,
  adding `siumai::compat::client` and `siumai_registry::compat::client`, and removing the registry
  root `LlmClient` export.
- Document the registry-root generic-client migration path:
  `siumai_registry::LlmClient` -> `siumai_registry::compat::client::LlmClient`.
- Update provider crate internal imports to use `core_compat::client::LlmClient` so provider roots
  do not keep local broad `client` aliases to core compatibility internals.
- Complete FMD-050 by routing OpenAI and Anthropic production factory construction through
  provider-owned private typed builders, deprecating legacy broad `registry::factory::build_*_client`
  helpers, documenting their migration path, and making `all-providers` include `google-vertex`.
- Complete FMD-060 by extracting the Gemini GenerateContent request parser into
  `siumai-bridge/src/request/normalize/gemini_generate_content.rs`, leaving `normalize.rs` with only
  public wrapper delegation for that protocol slice and adding source guards for the new adapter
  boundary.
- Complete FMD-070 by extracting OpenAI Responses request body construction to
  `responses_request_builder`, hosted/dynamic output item helpers to `responses/hosted_tools.rs`,
  and response metadata/source/logprobs aggregation to `responses/metadata.rs`, with source guards
  and protocol/facade OpenAI fixture gates.
- Complete FMD-080 by deciding to keep `ContentPart`'s breaking namespace move deferred and to use
  FMD-090 for a non-breaking request-side bridge adapter extraction proof slice.
- Complete FMD-090 by extracting request-side legacy `ContentPart` adapters to
  `siumai-bridge/src/request/legacy_content.rs`, routing OpenAI/Anthropic/Gemini request
  normalization through that module, and guarding that request normalization does not populate
  response `provider_metadata`.
- Complete FMD-100 by splitting provider public-path parity scenarios into provider-local modules
  while preserving the `provider_public_path_parity_test` binary and registry architecture guards.

## Blockers

- None known.
- Do not move or rename public `ContentPart` in this lane. The breaking namespace move remains
  deferred to a dedicated follow-on if/when ADR-0008 preconditions are met.

## Next Recommended Action

1. If the team wants to continue compatibility cleanup, open a new focused workstream for the
   breaking public `ContentPart` namespace move / response-side generated-output adapter deepening.
2. Keep future provider additions inside provider/protocol-owned catalogs and provider-local parity
   modules; the new source guards should fail if provider-owned facts drift back into spec/core.
3. Do not split broad public-surface import tests unless a concrete locality problem appears; they
   remain broad by design.
