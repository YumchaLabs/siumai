# Fearless Module Deepening — Design

Status: Complete
Last updated: 2026-05-19

## Why This Lane Exists

Siumai has already completed the broad Vercel-aligned crate split and most provider-interface
convergence work. The remaining architecture friction is no longer a lack of direction; it is that
several Modules still expose wide or shallow Interfaces that keep provider-owned knowledge,
compatibility paths, and protocol conversion policy too close to the stable spec/core surfaces.

This lane records the next fearless-refactor pass: deepen those Modules, improve locality, and make
the target architecture harder to regress.

## Relevant Authority

- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0002-provider-crates-by-provider.md`
  - `docs/adr/0003-provider-ext-export-policy.md`
  - `docs/adr/0004-experimental-surface-policy.md`
  - `docs/adr/0005-builder-retention-and-convergence-policy.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
  - `docs/adr/0007-llmclient-demotion-policy.md`
  - `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- Existing docs:
  - `docs/architecture/module-split-design.md`
  - `docs/architecture/public-surface.md`
  - `docs/architecture/capability-surface.md`
  - `docs/architecture/provider-extensions.md`
  - `docs/architecture/registry-without-builtins.md`
- Related workstreams:
  - `docs/workstreams/ai-sdk-provider-interface-convergence/`
  - `docs/workstreams/fearless-architecture-convergence/`
  - `docs/workstreams/fearless-boundary-hardening/`
  - `docs/workstreams/fearless-spec-core-boundary-convergence/`
  - `docs/workstreams/fearless-content-part-boundary-split/`
  - `docs/workstreams/google-interactions-runtime-alignment/`
  - `docs/workstreams/stream-delta-lossless-boundary/`
- Reference implementation:
  - `repo-ref/ai/packages/provider`
  - `repo-ref/ai/packages/provider-utils`
  - `repo-ref/ai/packages/ai`
  - `repo-ref/ai/packages/*` provider packages

## Problem

The architecture review found these high-leverage seams that still need deepening:

1. `siumai-spec/src/tools.rs` still owns provider-defined tool catalog details for OpenAI,
   Anthropic, Google, Groq, and xAI. This leaks provider-owned knowledge into the data-only spec
   crate.
2. `ProviderType` in `siumai-spec/src/types/common.rs` is still a closed provider enum with many
   concrete providers. This contradicts the open provider-id direction used by provider options,
   registry handles, and provider metadata.
3. `siumai-core` still combines provider-interface contracts, provider-utils-style runtime helpers,
   compatibility `LlmClient`, streaming processors, tooling, UI, and structured-output helpers under
   one broad crate/interface.
4. `LlmClient` remains physically central in `siumai-core/src/client.rs` even though ADR-0007
   demotes it to a compatibility abstraction.
5. `siumai-bridge/src/request/normalize.rs` is a large multi-protocol normalization Module that
   mixes OpenAI, Anthropic, Gemini, provider-defined tools, and legacy `ContentPart` conversion
   policy.
6. Legacy `ContentPart` still acts as a mixed request/response/provider compatibility carrier, so
   callers and tests must repeatedly reason about lossy versus lossless conversion.
7. `siumai-protocol-openai` is the right crate-level seam, but several internal Modules remain
   large enough that request mapping, tool mapping, response item parsing, usage, metadata, and
   streaming accumulation are hard to change independently.
8. `siumai-registry/src/registry/factory.rs` still has old broad `build_*_client` helpers and
   compatibility-shaped parameters beside the family-first `ProviderFactory` seam.
9. Some architecture and parity tests are valuable but oversized, making failures and future
   provider additions harder to localize.

These are not all one patch. This lane exists to sequence them safely, beginning with the smallest
independently provable residue removal.

## Target State

When this workstream closes:

- `siumai-spec` owns provider-agnostic data shapes only. Provider-defined tool catalogs and
  provider-name policies live in provider/protocol/registry-owned Modules.
- Stable public construction and execution paths remain family-first. Compatibility paths are
  physically scoped under explicit compat Modules.
- Protocol bridge Modules expose narrower pair/protocol Adapters instead of concentrating all
  provider and legacy rules in one normalization file.
- Legacy `ContentPart` usage is either reduced through directional request/response adapters or
  split into a dedicated follow-on with clear breaking-slice criteria.
- OpenAI protocol internals are organized around deep submodules whose Interfaces match behavior:
  request, tools, output items, usage/metadata, and stream accumulation.
- Registry construction no longer encourages old `build_*_client` helper usage for new provider
  work.
- Architecture guard tests prevent provider residue from re-entering `siumai-spec`/stable core
  surfaces.

## In Scope

- Moving provider-defined tool catalogs out of `siumai-spec`.
- Introducing or documenting a provider-id-first replacement for closed provider classification.
- Physically isolating `LlmClient` and other compatibility construction helpers.
- Splitting bridge normalization and OpenAI protocol internals by real seams.
- Planning and, where safe, implementing directional `ContentPart` adapters.
- Refactoring oversized architecture/parity tests when it improves locality and future provider
  changes.
- Updating architecture docs, migration notes, and source guards as each slice lands.

## Out Of Scope

- Rewriting provider implementations only for naming symmetry.
- Removing public compatibility APIs without migration notes and deprecation windows.
- Creating a TypeScript-style callable provider object model when Rust config-first construction is
  already the chosen Interface.
- Moving gateway/server concerns into `siumai-core`.
- Reopening completed broad workstreams unless a concrete seam in this lane requires a reference
  update.
- Treating file size alone as a reason to split a Module. A split must improve locality or leverage.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Provider-defined tool catalogs are provider/protocol-owned behavior, not spec data. | High | `docs/architecture/public-surface.md`, `docs/architecture/provider-extensions.md`, `repo-ref/ai/packages/*/src/tool/*` | If catalog helpers must stay in spec for public compatibility, keep deprecated re-exports and move canonical implementations behind provider/protocol Modules first. |
| `ProviderType` is now mostly compatibility/classification policy rather than a stable provider-agnostic data type. | High | ADR-0001 open provider options map; registry provider-id surfaces; provider package split | If downstream users rely heavily on `ProviderType`, introduce `ProviderId` first and deprecate gradually. |
| `LlmClient` can be physically moved or re-exported as compatibility without breaking family-first execution. | Medium | ADR-0007 and registry handle boundary tests | If public breakage is too high, start with module aliasing and source guards before path removal. |
| Bridge request normalization has enough protocol-pair variation to justify real Adapters. | High | Existing `request/pairs/*`, gateway/bridge tests, OpenAI/Anthropic/Gemini fixtures | If pair-level splits duplicate too much code, keep shared primitives and only move provider-specific parsing/mapping policy. |
| Legacy `ContentPart` split is a breaking slice and may need its own follow-on. | High | ADR-0008 and current conversion failure helpers | If the slice is too broad, this lane should produce the adapter seam and split the full rename/removal into another workstream. |

## Architecture Direction

Use the Module/Interface/Depth vocabulary:

- A Module earns its keep when deleting it would make complexity reappear across many callers. A
  Module is shallow when deleting it mostly removes pass-through code.
- This lane should deepen real seams, not add new wrapper layers for symmetry.
- Provider-owned facts must live behind provider/protocol/registry Interfaces:
  - tool ids and provider-native names,
  - provider aliases and default models,
  - protocol transformers and wire quirks,
  - provider-specific metadata/options.
- `siumai-spec` should stay a passive data Module. `siumai-core` should move toward provider
  Interface + provider-utils separation, either by internal namespace first or by crate split after
  smaller seams are clean.
- Compatibility remains allowed, but the Interface must name it honestly (`compat::*`,
  `compat_*_client*`, deprecated re-exports).

The first execution slice is provider-defined tool catalog extraction because it is the smallest
clear violation of provider ownership and can be tested without changing the whole runtime model.

## Closeout Condition

This lane can close when:

- the prioritized residue/deepening slices in `TODO.md` are implemented or explicitly split,
- architecture/source guards prevent the same provider-owned residue from re-entering spec/core,
- targeted `cargo nextest` gates pass for affected crates,
- docs reflect the shipped ownership rules,
- and any remaining breaking slice is recorded as a narrower follow-on with its own target state.

## Closeout Result

Status: complete as of 2026-05-19.

This lane shipped the intended module-deepening target state without changing the public legacy
`ContentPart` carrier shape:

- provider-defined hosted-tool catalog ownership moved out of `siumai-spec` and into
  protocol/provider-owned modules;
- primary identity and registry catalog flows are provider-id-first, with closed `ProviderType`
  retained only as compatibility classification;
- `LlmClient`/`ClientWrapper` and generic-client downcast paths are physically scoped under
  explicit compatibility modules;
- production registry factories use provider-owned typed builders and family-first
  `ProviderFactory` methods instead of old broad `build_*_client` helpers;
- bridge request normalization gained narrower protocol/adapter seams, including the Gemini
  GenerateContent adapter and the request-side legacy `ContentPart` adapter;
- OpenAI Responses request/response internals were split around request body assembly,
  hosted-tool output handling, and metadata/source/logprobs aggregation;
- provider public-path parity tests were split from a monolithic test file into a shared harness
  plus provider-local modules.

The remaining breaking public-shape work is intentionally **not** part of this closed lane. A future
workstream should handle the public `ContentPart` namespace/compatibility break only after ADR-0008
preconditions are met and request/response adapters cover the necessary paths.
