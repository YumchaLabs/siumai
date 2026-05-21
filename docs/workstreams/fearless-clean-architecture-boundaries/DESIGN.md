# Fearless Clean Architecture Boundaries — Design

Status: Closed
Last updated: 2026-05-21

## Why This Lane Exists

Siumai has already completed several Vercel AI SDK alignment and boundary-hardening lanes, but the
current architecture still carries transitional seams:

- stable family model execution and compatibility `LlmClient` construction still share the same
  `ProviderFactory` surface;
- OpenAI-compatible protocol conversion, vendor runtime, and provider presets still overlap across
  protocol, provider, registry, and facade modules;
- legacy `ContentPart` compatibility carriers remain visible enough that request-side
  `providerOptions` and response-side `providerMetadata` knowledge can leak across call sites;
- `siumai-core` still acts as provider interface, provider-utils runtime, generic HTTP/SSE toolkit,
  compatibility host, and family trait home at the same time;
- `siumai-bridge` still owns protocol-target parsing/serialization knowledge that should belong to
  protocol adapters;
- facade exports and family taxonomy are not yet as small and final as the intended public surface.

This lane is the next fearless refactor program. The goal is not a minimal patch. The goal is the
cleanest architecture and seams we can justify from current source, docs, and the local Vercel AI
SDK reference in `repo-ref/ai`.

## Relevant Authority

- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0002-provider-crates-by-provider.md`
  - `docs/adr/0004-experimental-surface-policy.md`
  - `docs/adr/0005-builder-retention-and-convergence-policy.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
  - `docs/adr/0007-llmclient-demotion-policy.md`
  - `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- Architecture docs:
  - `docs/architecture/module-split-design.md`
  - `docs/architecture/public-surface.md`
  - `docs/architecture/provider-extensions.md`
  - `docs/architecture/registry-without-builtins.md`
- Prior workstreams:
  - `docs/workstreams/fearless-refactor-v4/`
  - `docs/workstreams/fearless-architecture-convergence/`
  - `docs/workstreams/fearless-boundary-hardening/`
  - `docs/workstreams/fearless-spec-core-boundary-convergence/`
  - `docs/workstreams/fearless-module-deepening/`
  - `docs/workstreams/ai-sdk-provider-interface-convergence/`
  - `docs/workstreams/protocol-response-generated-output-boundary/`
  - `docs/workstreams/content-part-compat-namespace-break/`
  - `docs/workstreams/video-model-family-alignment/`
- Reference implementation:
  - `repo-ref/ai/packages/provider`
  - `repo-ref/ai/packages/provider-utils`
  - `repo-ref/ai/packages/ai`
  - `repo-ref/ai/packages/openai`
  - `repo-ref/ai/packages/openai-compatible`

## Problem

The previous lanes moved the project in the right direction, but several Modules are still shallow:

1. `ProviderFactory` exposes both stable family construction and compatibility-client construction.
   The Interface is nearly as complex as the migration implementation it hides.
2. OpenAI-compatible support has no single obvious seam. Protocol mapping, runtime behavior, vendor
   configuration, typed options, registry factories, and facade exports are all partially involved.
3. `ContentPart` is classified as compatibility-only, but broad spec/facade exports still make it
   easy for new request/response code to use the legacy carrier directly.
4. `siumai-core` is too wide. It contains family traits, runtime executors, HTTP/SSE utilities,
   middleware, generic streaming, compatibility client types, and provider-spec routing.
5. `siumai-bridge` is not only a bridge contract; it also embeds target protocol knowledge.
6. The public facade still mirrors internals in some places, and the family taxonomy needs to be
   final and explicit after video-family alignment.

If these seams stay blurry, adding new providers and features will keep reintroducing historical
debt into provider-agnostic crates.

## Target State

When this lane closes:

- stable family model execution has a narrow primary `ProviderFactory` Interface;
- compatibility `LlmClient` construction is isolated behind explicit compatibility adapters and no
  longer shapes the primary registry/provider Interface;
- OpenAI-compatible support has one clear architecture:
  - protocol conversion and stream state machines are protocol-owned;
  - runtime adapter behavior is centralized;
  - vendor presets own only provider identity, endpoints, headers, quirks, typed options, and
    metadata;
- request prompt parts, generated output parts, and legacy compatibility content are separate
  seams, with source guards preventing new production direct `ContentPart` construction outside
  audited adapters;
- `siumai-core` has a smaller provider-interface/runtime role, and provider-utils-like behavior is
  isolated either in a new crate or a sharply scoped module with a documented future crate split;
- `siumai-bridge` owns bridge contracts, reports, policy, lifecycle, and customization, while
  target-specific protocol adapters live in protocol/provider-owned modules;
- facade exports are intentionally small:
  - stable family prelude;
  - scoped provider extensions;
  - explicit protocol paths;
  - explicit `compat`;
  - explicit `experimental`;
- family taxonomy is final for this release line: stable families include video if the shipped
  source already treats it as stable; music remains extension-only unless a new ADR says otherwise;
- tests and source guards prove these seams.

## In Scope

- Refactor registry/provider construction around native family model objects.
- Isolate or remove compatibility paths that only exist because old code expected broad
  `LlmClient` clients.
- Clean up OpenAI-compatible protocol/runtime/vendor responsibilities.
- Tighten `ContentPart` request/response compatibility usage.
- Split or isolate provider-utils-like runtime helpers from provider-interface contracts.
- Move target-specific bridge adapters out of `siumai-bridge` where feasible.
- Tighten facade re-exports and docs.
- Update architecture docs, migration notes, source guards, and workstream evidence.
- Delete redundant legacy code when the canonical path is tested.

## Out Of Scope

- Rewriting the entire public API merely to mimic TypeScript names mechanically.
- Adding new providers for feature coverage.
- Changing business semantics without fixture or contract evidence.
- Keeping compatibility paths solely because they are old, if the migration docs and tests support
  deletion.
- Large workspace-wide formatting unrelated to touched files.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Existing closed workstreams should not be reopened for this broad program. | High | `docs/workstreams/INDEX.md` records 0 active lanes and most prior fearless lanes closed/superseded. | If a lane must be reopened, split this program into a smaller follow-on and update the index. |
| `ProviderFactory` remains public for custom registries, but its stable Interface can be narrowed. | High | `docs/architecture/registry-without-builtins.md`, ADR-0006, ADR-0007. | If downstream compatibility requires old methods, keep them in an explicit compat extension trait. |
| OpenAI-compatible vendors benefit from centralized runtime behavior. | High | Multiple provider factories reuse OpenAI-compatible paths; Vercel has a dedicated `@ai-sdk/openai-compatible`. | If provider differences dominate, create provider-specific adapters but keep protocol conversion shared. |
| `ContentPart` cannot be deleted outright yet. | High | ADR-0008 and protocol response workstream preserve serde/legacy payloads. | If replacement surfaces become complete, this lane may include a breaking deletion or deprecated re-export phase. |
| A new provider-utils crate may be desirable but can be phased. | Medium | `siumai-core` currently hosts runtime helpers similar to `@ai-sdk/provider-utils`. | If crate split churn is too high, first create an internal deep module and defer publication. |
| Video should be treated as a stable family for this release line. | Medium | `video-model-family-alignment` is closed and source has `VideoModel` + registry handles; architecture docs still mention 6 families. | If maintainers reject video as stable, update docs and move video to explicit extension before implementation tasks proceed. |

## Architecture Direction

Use Vercel AI SDK as a structural reference, not a naming mandate:

```text
User code
  ↓
siumai facade
  ↓
siumai-registry family handles
  ↓
siumai-provider-* adapters
  ↓
siumai-protocol-* protocol conversion and stream state
  ↓
siumai-core provider Interface + generic runtime contracts
  ↓
provider-utils-like HTTP/SSE/retry/tooling utilities
  ↓
siumai-spec passive data shapes
```

The important rule is ownership:

- `siumai-spec` owns passive provider-agnostic data.
- `siumai-core` owns family traits, generic runtime contracts, middleware contracts, and
  provider-agnostic runtime composition.
- provider-utils-like helpers own generic HTTP/SSE/retry/download/parse/tool utility behavior.
- `siumai-protocol-*` owns wire conversion and protocol stream state machines.
- `siumai-provider-*` owns provider config, auth defaults, typed options, metadata, resources,
  hosted tools, and concrete family model adapters.
- `siumai-registry` owns provider resolution, caching, build overrides, and family handles.
- `siumai` owns stable public imports and feature aggregation.
- `siumai-bridge` owns bridge reports/policies/lifecycle/customization, not target protocol wire
  parsing.

Use the deletion test for each candidate Module:

- deleting a pass-through wrapper should not make users lose a real concept;
- deleting a deep Module should force the same complexity to reappear across many callers.

## Execution Strategy

This workstream is intentionally large, so execution is split into independently provable slices:

1. Freeze seam decisions and source-guard inventory.
2. Narrow registry and compatibility construction.
3. Deepen OpenAI-compatible protocol/runtime/vendor seams.
4. Deepen directional content seams.
5. Isolate provider-utils-like runtime helpers.
6. Move bridge target adapters to owning modules.
7. Tighten facade and family taxonomy.
8. Run integration gates, update docs, and close or split follow-ons.

Each slice must:

- land with tests or source guards;
- record fresh evidence in `EVIDENCE_AND_GATES.md`;
- update `HANDOFF.md`;
- avoid reverting user changes;
- use `cargo fmt` only for touched packages/files where practical;
- prefer `cargo nextest`.

## Closeout Condition

This lane can close when:

- all primary seams above are either implemented or explicitly split into narrower follow-on
  workstreams;
- obsolete compatibility code in the completed seams is removed, not merely hidden;
- architecture docs and migration docs match the shipped behavior;
- source guards cover the highest-risk regressions;
- targeted package gates pass;
- a final review and fresh verification pass are recorded.

## Final Closeout Summary

Status: closed on 2026-05-21.

FCAB-010 through FCAB-150 satisfied the target state for this lane. The shipped architecture now has
explicit seams for registry family construction versus compatibility construction, protocol-owned
OpenAI-compatible conversion, directional prompt/output/compat content, provider-utils-owned
AI SDK-style helper behavior, protocol-backed Gemini bridge normalization, a narrow facade surface,
and a seven-family taxonomy with Music intentionally extension-only.

No immediate child workstream was opened from FCAB-150. The remaining items are deliberate future
candidates rather than blockers:

- delete `siumai-core::utils::*` compatibility aliases only after the documented migration window;
- open a narrower bridge-target-adapter lane if OpenAI/Anthropic direct-pair parsing can move while
  keeping bridge loss/replay policy out of protocol crates;
- open a family-taxonomy ADR only if Music becomes a stable first-class model family;
- fix the local PowerShell/WSL bash environment separately if maintainers want to use the bash
  smoke scripts directly on Windows.
