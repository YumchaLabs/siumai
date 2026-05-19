# ContentPart Compatibility Namespace Break — Design

Status: Closed
Last updated: 2026-05-19

## Why This Lane Exists

ADR-0008 deliberately kept legacy `ContentPart` in its old public paths while request and response
directional adapters matured. The previous `fearless-content-part-boundary-split` and
`fearless-module-deepening` lanes completed the first proof slices:

- request-side legacy `ContentPart` construction is now routed through named adapters in bridge
  request normalization;
- response-side generated text projection already has a named `ContentPart` ->
  `GenerateTextContentPart` helper in `siumai-spec`;
- source guards classify high-value direct `ContentPart::...`, `providerOptions`, and
  `providerMetadata` usage.

This lane owns the next breaking slice: make the public namespace tell the truth. Legacy
`ContentPart` should be importable from an explicit compatibility namespace, while stable examples
and recommended surfaces use prompt/model-message parts for input and generated-output parts for
output.

## Relevant Authority

- ADRs:
  - `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0003-provider-ext-export-policy.md`
- Existing docs:
  - `docs/architecture/public-surface.md`
  - `docs/architecture/module-split-design.md`
  - `docs/migration/migration-0.11.0-beta.7.md`
- Related workstreams:
  - `docs/workstreams/fearless-content-part-boundary-split/`
  - `docs/workstreams/fearless-module-deepening/`
  - `docs/workstreams/fearless-spec-core-boundary-convergence/`
  - `docs/workstreams/generate-text-output-alignment/`
  - `docs/workstreams/prompt-model-message-surface-alignment/`

## Problem

`ContentPart` is still visible as a normal stable type through broad unified imports. That makes the
legacy carrier look canonical even though it mixes request-side `providerOptions` and response-side
`providerMetadata`.

The concrete remaining problems are:

1. the old `ContentPart` path does not communicate compatibility-only status strongly enough;
2. public docs and examples still contain old-path imports in some places;
3. response-side generated-output projection is named, but not yet physically isolated as a
   compatibility adapter module;
4. source guards verify directional maps, but do not yet enforce the final public namespace rule;
5. migration notes do not yet teach the actual breaking import path for `ContentPart`.

## Target State

When this workstream closes:

- legacy `ContentPart`, `MessageContent`, and closely-coupled legacy chat content helpers have a
  canonical explicit compatibility namespace;
- stable recommended imports teach `ModelMessage`, `UserContentPart`, `AssistantContentPart`,
  `ToolContentPart`, `GenerateTextContentPart`, and V4 prompt/output parts instead of legacy
  `ContentPart`;
- old broad or root facade imports for `ContentPart` are removed or reduced to consciously
  deprecated migration aliases with tests proving they are not recommended stable prelude exports;
- response-side legacy `ContentPart` -> generated-output projection is physically isolated behind a
  response compatibility adapter module;
- serde compatibility for `ChatMessage` / `ChatResponse` payloads and provider fixture parity is
  preserved;
- migration notes state the new compatibility import and replacement directional imports.

## In Scope

- Adding an explicit compatibility namespace for legacy chat content carriers.
- Moving or aliasing `ContentPart` exports so the canonical public compatibility path is clear.
- Removing `ContentPart` from recommended unified prelude exports if the selected break can support
  it.
- Extracting response-side generated-output projection helpers into a named response adapter module.
- Updating docs, migration notes, examples, and architecture/source guards.
- Running targeted spec, facade, bridge, protocol, and provider parity gates.

## Out Of Scope

- Removing the `ContentPart` enum or changing its serialized shape.
- Rewriting every protocol response parser in one patch.
- Changing `ChatMessage` / `ChatResponse` serde compatibility.
- Removing all legacy convenience constructors if they are still required by compatibility APIs.
- Designing a new unrelated public content model. The directional replacements already exist.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| The compatibility namespace can be introduced before deleting the old enum shape. | High | ADR-0008 allows move-or-re-export for one release cycle | If compile breakage is too high, keep a deprecated old-path alias for one beta while making compat the canonical docs path. |
| `ChatMessage` and `ChatResponse` can keep using legacy content internally while imports move. | High | Serde compatibility depends on the enum shape, not its source import path | If type movement breaks serde derives, use module re-export rather than physical enum relocation. |
| Removing `ContentPart` from `prelude::unified` is the highest-value public signal. | Medium | `prelude::unified` is the recommended stable import surface | If too many public tests depend on it, land the compat namespace and source guards first, then split removal into a later task. |
| Response projection can be isolated without behavior changes. | High | `project_response_content_part_to_generate_text_content_part` is already named and tested | If extraction creates circular module dependencies, keep function exports stable and move implementation behind a private module. |

## Architecture Direction

Use a compatibility edge, not a new canonical model:

- request input remains prompt/model-message parts;
- generated output remains `GenerateTextContentPart` and output-part carriers;
- low-level V4 integrations use `LanguageModelV4*` prompt/content parts;
- legacy `ContentPart` is the serde and migration carrier behind `compat::content` or equivalent.

Breaking public namespace work should be incremental but honest. Prefer a compile-visible break in
recommended facade paths over silently leaving `ContentPart` in the stable prelude. If an old path is
kept temporarily, it must be explicitly deprecated and covered by migration docs and source guards.

## Closeout Condition

This lane can close when:

- the canonical compatibility namespace exists and is documented;
- response-side generated-output projection is adapter-owned;
- stable recommended imports no longer present legacy `ContentPart` as canonical;
- migration docs include before/after imports and directional replacement guidance;
- targeted `cargo nextest` gates pass for spec, facade, bridge/protocol fixtures, and public
  surface import guards;
- any intentionally retained old-path aliases are documented as migration-only with a removal plan.

## Closeout Summary

Closed on 2026-05-19.

Shipped:

- explicit legacy content compatibility namespace through spec/core/facade/prelude layers;
- stable `prelude::unified` no longer exports legacy `ContentPart`;
- response-side legacy `ContentPart` -> generated-output projection moved into
  `siumai-spec/src/types/ai_sdk/response_compat_projection.rs`;
- migration/public-surface docs teach explicit compatibility imports and replacement directional
  request/response content families;
- source guards prove the public namespace rule and response adapter directionality.

Split/deferred:

- broad protocol/parser migration to generated-output parts. Current bridge/protocol response
  encoders target provider-native `ChatResponse` / `ContentPart` shapes directly; forcing one path
  through `GenerateTextContentPart` now would introduce lossy round-trips and is deferred to a
  separate future lane if a generated-output protocol response model becomes natural.
