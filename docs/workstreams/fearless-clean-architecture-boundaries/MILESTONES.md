# Fearless Clean Architecture Boundaries — Milestones

Status: Closed
Last updated: 2026-05-21

## FCAB-M0 — Scope And Regression Guard Baseline

Exit criteria:

- Workstream documents exist and agree.
- Relevant ADRs, docs, and prior lanes are linked.
- Current seam inventory is recorded.
- Source guards cover the highest-risk regressions before large moves start.

Primary evidence:

- `docs/workstreams/fearless-clean-architecture-boundaries/DESIGN.md`
- `docs/workstreams/fearless-clean-architecture-boundaries/TODO.md`
- `docs/workstreams/fearless-clean-architecture-boundaries/EVIDENCE_AND_GATES.md`

Status: complete. FCAB-010 opened the lane; FCAB-020 recorded the seam inventory and refreshed the
baseline source guards.

## FCAB-M1 — Registry Compatibility Is No Longer The Primary Seam

Exit criteria:

- Stable family construction has a narrow primary Interface.
- Compatibility `LlmClient` construction is explicit, isolated, and non-primary.
- Built-in provider factories use native family paths when the family exists.
- Extension-only compatibility adapters are listed and justified.

Primary gates:

- Focused `siumai-registry` factory/handle tests.
- Source guards proving stable family handles do not execute through compatibility clients.

Status: complete. FCAB-030 introduced explicit family, compatibility, and extension factory facets.
FCAB-040 then centralized built-in provider typed-client projection helpers and removed repeated
same-family compatibility glue from factory methods.

## FCAB-M2 — OpenAI-Compatible Architecture Is Deep And Single-Purpose

Exit criteria:

- `siumai-protocol-openai` owns protocol conversion and stream state.
- OpenAI-compatible runtime behavior is centralized.
- Vendor Modules own presets, quirks, typed options, metadata, and facade extension exports only.
- Redundant provider/protocol re-exports are deleted or explicitly documented as compatibility.

Primary gates:

- `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`
- OpenAI-compatible provider tests.
- Registry/public-surface tests for touched vendors.

Status: complete. FCAB-050 completed the protocol-owned OpenAI-compatible completion response and
stream conversion seam. `siumai-protocol-openai` now owns completion conversion and parser state,
while `siumai-provider-openai-compatible` delegates protocol conversion and keeps HTTP execution /
capability routing. FCAB-060 then moved TogetherAI image runtime ownership out of the registry and
into `siumai-provider-togetherai`, leaving the registry factory to compose provider-owned image /
rerank clients and the shared OpenAI-compatible text/audio runtime. Source guards now cover both the
protocol-owned completion seam and the provider-owned TogetherAI image runtime seam.

## FCAB-M3 — Directional Content Seams Replace Legacy Content Drift

Exit criteria:

- Request prompt data, generated output data, and legacy compatibility content are visibly separate.
- New production direct `ContentPart` construction is blocked outside audited adapters.
- Request adapters do not read response provider metadata.
- Response adapters do not emit request provider options except documented legacy payloads.

Primary gates:

- `siumai-spec` content tests.
- Protocol/provider adapter tests.
- Source guards for directional content hygiene.

Status: complete. FCAB-070 completed the export/facade half of M3 by making request prompt content,
generated output content, and legacy compatibility content visible as `content::prompt`,
`content::output`, and `content::compat` namespaces across spec/core/facade surfaces. Stable
`prelude::unified` exposes only `prompt` and `output` navigation modules and keeps legacy
`ContentPart` on explicit compatibility paths. FCAB-080 then moved high-value production response
construction for OpenAI-compatible chat, Anthropic streaming, Cohere, and Ollama behind named
`response_content` adapters and classified the remaining request/inspection/core paths in
`content-part-adapter-audit.md`.

## FCAB-M4 — Provider-Utils Responsibility Is Isolated From Core Interface

Exit criteria:

- HTTP/SSE/retry/download/parse/tooling utilities have a deep provider-utils-like Module or crate.
- Provider/protocol crates import generic utilities from that seam.
- `siumai-core` no longer exposes broad utility mirrors unless classified as stable,
  experimental, or compat.

Primary gates:

- `cargo nextest run -p siumai-core --no-fail-fast`
- Focused provider/protocol package gates for moved imports.

Status: complete for the planned provider-utils isolation slice. FCAB-090 introduced the
`siumai-provider-utils` crate, modeled after Vercel's provider-utils seam. FCAB-100 deepened it into
the canonical home for AI SDK-style spec-only helpers: URL composition, MIME detection, builder
defaults, chat request normalization, data/base64 helpers, downloads, headers, IDs, JSON
instruction/parse helpers, provider options/references, reasoning mapping, runtime metadata, serial
jobs, settings, UTF-8 decoding, and runtime type validation. Matching `siumai-core::utils::*`
modules remain compatibility aliases. The only core-owned utility implementations left are
`cancel` (stable core runtime stream/cancellation wiring) and `streaming_tool_call` (explicit compat
helper over core stream parts). FCAB-110 can proceed without reopening provider-utils ownership.

## FCAB-M5 — Bridge Owns Reports And Policy, Not Target Wire Formats

Exit criteria:

- `siumai-bridge` keeps bridge reports, lifecycle, strictness policy, lossiness reports, and
  customization.
- Protocol-target parsing/serialization adapters move to protocol/provider-owned Modules where
  feasible.
- Any remaining bridge-owned target adapters are documented as temporary shims.

Primary gates:

- `cargo nextest run -p siumai-bridge --features openai,anthropic,google --no-fail-fast`
- Protocol package gates for moved adapters.

Status: complete for the first bridge-target adapter slice. FCAB-110 moved Gemini GenerateContent
request JSON normalization into `siumai-protocol-gemini::standards::gemini::request_bridge`, leaving
`siumai-bridge` with public wrappers, reports, loss policy, hooks, lifecycle, target dispatch, and a
thin compatibility shim. `bridge-target-adapter-audit.md` lists retained bridge-owned
OpenAI/Anthropic direct-pair and normalization paths that should only move when bridge
loss/replay policy can stay out of protocol crates.

## FCAB-M6 — Public Surface And Family Taxonomy Are Final For This Release Line

Exit criteria:

- `siumai::prelude::unified::*` is small and family-first.
- Provider-specific features live under scoped provider extension Modules.
- Protocol, compat, and experimental paths are explicit.
- Stable family list is consistent across source and docs.
- Video status is settled; music remains extension-only unless an ADR says otherwise.

Primary gates:

- Public-surface compile guards.
- Docs consistency checks.
- Family contract tests.

Status: complete. FCAB-120 finalized the facade/public-surface half of M6: root utility helpers are
provider-utils-backed, `prelude::unified` keeps only the narrow AI SDK-style helper subset, legacy
`ContentPart` usage in tests is explicit through compatibility imports, `experimental` uses named
advanced modules instead of a grouped core mirror, and retained broad exports are documented only as
explicit namespaces. FCAB-130 then settled the family taxonomy for this release line: the stable
family list is Language, Embedding, Image, Rerank, Speech, Transcription, and task-oriented Video.
Music remains extension-only through `MusicGenerationCapability` / provider extensions, with no
stable `MusicModel` or registry `music_model(...)` handle unless a future ADR promotes it.

## FCAB-M7 — Closeout

Exit criteria:

- All completed seams have fresh evidence.
- Remaining unresolved work is split into narrower follow-on workstreams or explicitly deferred.
- Migration and architecture docs match shipped behavior.
- `WORKSTREAM.json`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, and `HANDOFF.md`
  agree.

Primary gates:

- Selected smoke/full gate recorded in `EVIDENCE_AND_GATES.md`.
- `verify-rust-workstream` fresh evidence.
- `review-workstream` with no blocking findings.

Status: complete. FCAB-140 completed the integration/verification sweep with a documented Windows
substitute for the unavailable bash smoke path. Fresh evidence covers formatting,
provider-dependency guard semantics, core, provider-utils, bridge, facade source guards, public
surface imports, OpenAI/OpenAI-compatible provider/protocol gates, Anthropic/Gemini protocol gates,
registry focused feature matrix, and `siumai-registry --features all-providers`. FCAB-150 closed the
lane after a final consistency review and planning gate. No immediate child workstream was split;
remaining items are documented as non-blocking future candidates.
