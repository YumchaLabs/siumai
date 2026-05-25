# Compatibility Surface Breaking Convergence

Status: Closed
Last updated: 2026-05-25

## Why This Lane Exists

The public-surface deepening lane closed with the remaining compatibility shims classified, not
deleted. That was correct because ADR-0007 and ADR-0008 still protect several migration paths.
Those retained paths now need a narrower breaking-convergence lane so removals and pre-removal
work happen deliberately instead of remaining as broad compatibility ballast.

## Relevant Authority

- ADRs:
  - `docs/adr/0007-llmclient-demotion-policy.md`
  - `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
- Existing docs:
  - `docs/architecture/public-surface.md`
  - `docs/migration/migration-0.11.0-beta.7.md`
- Related workstreams:
  - `docs/workstreams/fearless-public-surface-deepening/`
  - `docs/workstreams/fearless-public-surface-deepening/compatibility-shim-audit.md`
  - `docs/workstreams/fearless-architecture-convergence/`
  - `docs/workstreams/content-part-compat-namespace-break/`
  - `docs/workstreams/fearless-module-deepening/`

## Problem

Several compatibility surfaces remain intentionally available, but they still keep old architecture
shapes discoverable:

- `siumai::compat::types::*` and `siumai::prelude::compat::types::*` expose the whole broad core
  type namespace.
- `siumai_core::client` and `siumai_core::core::client` are lower-level migration aliases for
  `siumai_core::compat::client`.
- `ProviderCompatibilityFactory` and `ProviderFactory::compat_*_client*` still provide the generic
  `LlmClient` construction seam needed by method-style and extension-only paths.
- Legacy `ContentPart` still appears in public payload paths while ADR-0008 defers a full namespace
  break until directional adapters and fixture parity are mature enough.

## Target State

- Broad facade compatibility type imports are narrowed into named migration modules or explicit
  curated exports.
- Lower-level generic client aliases have deprecation/removal conditions, source guards, and as
  much safe call-site migration as possible.
- Registry generic-client factories are isolated behind the smallest compatibility facet possible,
  with stable family and extension paths kept native.
- ADR-0008's future breaking slice is either executed safely or blocked by concrete unmet
  preconditions documented in this lane.

## In Scope

- Replace broad facade compat type wildcard exports with smaller named compatibility groups when
  public compile tests can prove retained imports.
- Move internal or test imports off `siumai_core::client` / `siumai_core::core::client` where an
  explicit `siumai_core::compat::client` import is valid.
- Reduce or document remaining `ProviderCompatibilityFactory` / `compat_*_client*` dependency
  points and add guards against new stable-family uses.
- Evaluate ADR-0008's future-breaking conditions and execute only safe, well-documented namespace
  movement.
- Update migration and architecture docs when a public path changes.

## Out Of Scope

- Removing `LlmClient` before ADR-0007's provider and registry migration conditions are met.
- Breaking `ChatMessage` / `ChatResponse` serde compatibility without fixture parity.
- Reopening the closed public-surface deepening lane.
- Broad provider runtime behavior changes.
- Promoting extension-only music/file/skill surfaces to stable families without a separate ADR.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Broad `compat::types::*` can be narrowed without changing runtime behavior. | High | It is a facade re-export layer with public import tests. | Keep a deprecated bridge for one release and document the narrowing path. |
| `siumai_core::client` aliases cannot be removed immediately, but internal call sites can prefer `compat::client`. | High | ADR-0007 keeps migration paths but discourages new generic-client-centered code. | Defer removal while strengthening guards and docs. |
| Registry generic-client factory retirement needs provider/family prerequisites. | Medium | Stable registry handles already use family facets, but method-style construction and extension-only adapters remain. | Split provider-specific prerequisite workstreams instead of deleting the seam. |
| ADR-0008's full `ContentPart` namespace break may still be blocked. | High | ADR-0008 lists explicit future-breaking conditions. | Record unmet conditions and add guards rather than forcing an unsafe break. |

## Architecture Direction

This lane treats compatibility as an edge, not a hidden second architecture. Public imports should
make migration intent visible in the path, and stable model-family code should not depend on generic
`LlmClient` or dual request/response content carriers unless an ADR says the transition still
requires it.

Prefer reversible, test-visible steps:

- narrow facades before deleting implementation;
- migrate internal imports before removing aliases;
- add source guards before relying on convention;
- split follow-ons when a remaining shim depends on provider-wide migration.

## Closeout Condition

This lane can close when:

- broad facade compat types are narrowed or explicitly deferred with a smaller target shape,
- core generic-client aliases have concrete migration evidence and removal criteria,
- registry generic-client construction has the smallest safe compatibility seam,
- ADR-0008 ContentPart movement is either completed or blocked by named unmet conditions,
- evidence gates pass, and
- any remaining breaking work is split into narrower follow-ons.

## Closeout Summary

Closed on 2026-05-25.

The lane completed the safe compatibility-surface narrowing work:

- Facade `compat::types` and `prelude::compat::types` now expose a curated legacy set, with the old
  catch-all mirror moved under `legacy_all`.
- `siumai_core::client` and `siumai_core::core::client` are deprecated migration aliases with
  ADR-0007 removal guidance and source guards against production consumption.
- Registry image, speech, and transcription extras now route through `ProviderExtensionFactory`;
  stable family handles no longer store `ProviderCompatibilityFactory`.
- ADR-0008's `ContentPart` root namespace move was evaluated. The facade-level break is complete,
  while the low-level `siumai-spec::types::ContentPart` / `siumai-core::types::ContentPart` move is
  explicitly deferred behind serde payload parity and provider/protocol fixture coverage.

Residual breaking work is tracked as follow-on candidates, not hidden current-lane work.
