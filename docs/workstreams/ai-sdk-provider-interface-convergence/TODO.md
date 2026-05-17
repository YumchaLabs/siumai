# AI SDK Provider Interface Convergence - TODO

Status: Active
Last updated: 2026-05-18

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## M0 - Program Scope And Inventory

- [x] AIPC-010 [owner=planner] [deps=none] [scope=docs/workstreams/ai-sdk-provider-interface-convergence]
  Goal: Open the durable program lane and freeze the core/spec/registry/protocol/provider target
  seams.
  Validation: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`,
  `WORKSTREAM.json`, and `HANDOFF.md` exist and agree.
  Evidence: `docs/workstreams/ai-sdk-provider-interface-convergence/DESIGN.md`
  Handoff: This task creates the authority for later implementation slices.

- [x] AIPC-020 [owner=planner] [deps=AIPC-010] [scope=docs/workstreams/ai-sdk-provider-interface-convergence]
  Goal: Build the first AI SDK provider-interface parity inventory.
  Validation: `PARITY_INVENTORY.md` names reference packages, current Siumai anchors, drift, and
  first child-slice candidates.
  Evidence: `docs/workstreams/ai-sdk-provider-interface-convergence/PARITY_INVENTORY.md`
  Handoff: Keep updating the inventory as code slices land.

## M1 - Core And Registry Guard Slices

- [x] AIPC-030 [owner=codex] [deps=AIPC-020] [scope=siumai-core,siumai-registry,docs]
  Goal: Audit and strengthen source guards for provider-specific behavior in `siumai-core` and
  generic-client execution in stable registry family paths.
  Validation: `cargo nextest run -p siumai-core --no-fail-fast` and
  `cargo nextest run -p siumai-registry --no-fail-fast`
  Evidence: source guards and boundary tests under the affected crates.
  Handoff: Added a `siumai-core::standards` guard and confirmed the existing registry family-handle
  guard remains green. A stale empty `siumai-core/src/standards/openai` working-tree directory was
  removed locally because the guard correctly rejected it; no tracked code was deleted.

- [x] AIPC-040 [owner=codex] [deps=AIPC-030] [scope=siumai-registry/src/registry/entry]
  Goal: Reduce primary stable-family registry execution through `compat_*_client_with_ctx` or
  `as_*_capability()` where native family models already exist.
  Validation: focused registry handle tests, then `cargo nextest run -p siumai-registry --no-fail-fast`
  Evidence: updated handle tests and factory/handle code.
  Handoff: Primary stable family handles remain free of compat/downcast paths. The remaining image
  and audio compat clients are locked to extension-only edit/variation/streaming/listing/translation
  helper paths where no stable family trait currently owns the behavior.

## M2 - Stream Semantics Convergence

- [x] AIPC-050 [owner=codex] [deps=AIPC-030] [scope=siumai-core,siumai-protocol-*,siumai-extras]
  Goal: Continue migrating AI SDK-stable stream semantics to `ChatStreamEvent::Part(ChatStreamPart)`
  and keep protocol-only replay hints out of stable provider metadata.
  Validation: focused protocol stream tests for the affected provider, then package nextest.
  Evidence: stream parser/serializer tests and `PARITY_INVENTORY.md` status updates.
  Handoff: OpenAI Responses public feature tests, extras gateway tool assertions, and
  Anthropic/Gemini serializer guards now prefer stable stream parts for AI SDK-stable semantics.
  Keep provider-custom inputs only in explicitly named compatibility or provider-native tests.

- [~] AIPC-060 [owner=codex] [deps=AIPC-050] [scope=siumai-extras,siumai-bridge,docs]
  Goal: Move gateway/protocol bridge assertions toward stable-part-first expectations.
  Validation: `cargo nextest run -p siumai-bridge --no-fail-fast` and focused
  `siumai-extras` gateway tests.
  Evidence: bridge/gateway tests and migration notes.
  Handoff: First slice added a stable provider-tool stream part bridge regression test and tightened
  extras gateway code to import bridge-owned OpenAI Responses stream adapters directly from
  `siumai_bridge::stream`. Keep Axum/server transport concerns out of `siumai-core`.

## M3 - Provider Package Parity

- [ ] AIPC-070 [owner=unassigned] [deps=AIPC-020] [scope=siumai-provider-openai-compatible,provider docs]
  Goal: Re-audit OpenAI-compatible promoted vendors so package capability inheritance matches
  AI SDK package surfaces or documented Siumai decisions.
  Validation: focused provider public-path tests for each touched vendor.
  Evidence: provider capability matrix rows and tests.
  Handoff: Split one child workstream per vendor if the fix needs provider-specific API changes.

- [ ] AIPC-080 [owner=unassigned] [deps=AIPC-020] [scope=siumai-provider-*,siumai-protocol-*,docs]
  Goal: Run provider-by-provider package parity slices for native providers.
  Validation: package-specific no-network tests and public import tests.
  Evidence: child workstreams or inventory rows for OpenAI, Azure, Anthropic, Google/Gemini,
  Google Vertex, Bedrock, Cohere, Groq, xAI, DeepSeek, TogetherAI, MiniMaxi, Ollama.
  Handoff: Provider package names should not be copied mechanically; semantic parity wins.

## M4 - Workstream Hygiene And Closeout

- [ ] AIPC-090 [owner=planner] [deps=AIPC-030] [scope=docs/workstreams]
  Goal: Normalize or explicitly defer legacy unknown lanes that this program supersedes.
  Validation: `docs/workstreams/INDEX.md` and child lane `WORKSTREAM.json` files agree.
  Evidence: updated workstream index and affected handoff docs.
  Handoff: Do not rewrite historical status by assumption; mark inferred state clearly.

- [ ] AIPC-100 [owner=planner] [deps=AIPC-040,AIPC-050,AIPC-070,AIPC-080,AIPC-090]
  [scope=docs/workstreams/ai-sdk-provider-interface-convergence]
  Goal: Close this program lane or split remaining work into narrower follow-ons.
  Validation: evidence gates are recorded and `WORKSTREAM.json` status is updated.
  Evidence: `EVIDENCE_AND_GATES.md`, `HANDOFF.md`, `WORKSTREAM.json`
  Handoff: Final closeout must list any compatibility paths that intentionally remain.
