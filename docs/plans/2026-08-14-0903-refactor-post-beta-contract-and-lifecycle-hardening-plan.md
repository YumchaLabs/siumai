---
title: Post-Beta Contract and Lifecycle Hardening - Plan
type: refactor
date: 2026-08-14
deepened: 2026-08-14
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Post-Beta Contract and Lifecycle Hardening - Plan

## Goal Capsule

| Field | Contract |
|---|---|
| Objective | Make the post-`0.11.0-beta.10` workspace safer to release and easier to extend by fixing incorrect execution ownership, eliminating hidden replay, preserving durable state exactly, and deleting volatile model/product eligibility gates that should not be runtime authorities. |
| Priority order | Release finalization and durable correctness first; runtime tool identity and MCP replay/resource safety second; WebSocket settlement and transport authority third; provider eligibility cleanup and internal deepening last. |
| Product boundary | Preserve the six provider-neutral family traits and complementary provider-owned APIs. Do not add a universal client, capability matrix, generic resource trait, second routing layer, or cross-provider option-merger framework. |
| Compatibility posture | Breaking source and snapshot-identity changes are allowed during beta when required for correct ownership. Delete obsolete gates, tests, helpers, docs, and aliases instead of retaining compatibility shells. |
| Evidence hierarchy | Current official provider/library documentation and pinned dependency source are primary. Repository ADRs and deterministic fixtures are next. `repo-ref/ai` is secondary prior art only. |
| Execution posture | Prefer deep private modules, explicit state, bounded adapters, and deletion. Keep scripts thin and let Cargo, release-plz, protocol codecs, and transport implementations remain the authorities for their domains. |
| Tail ownership | Each implementation unit owns its tests, rustdoc, architecture/ADR updates when the durable contract changes, migration guidance for public breaks, and English Conventional Commits. Do not publish, tag, or open a PR unless separately authorized. |
| Stop conditions | Stop only for contradictory current official wire evidence, a pinned dependency that cannot expose the required replay/bounds control without replacement, an external crates.io/GitHub configuration dependency that cannot be staged safely, or overlapping user changes that cannot be isolated. |

---

## Product Contract

### Summary

Siumai's macro architecture is sound after `0.11.0-beta.10`; the next refactor should not reopen it. The highest-value work is to correct five concrete ownership failures:

1. release-plz may create the repository tag and GitHub Release before every independent workspace crate has published;
2. runtime replaces caller-supplied model-visible tools with its local executable set, and its durable tool identity omits provider annotations;
3. MCP can transparently replay a `NeverReplay` tool call and enforce response limits only after rmcp has materialized the payload;
4. durable snapshot semantics differ between the built-in store and external stores, and a missing internal usage-settlement field is accepted as valid v6 data;
5. several providers still use dated model-name allowlists to block wire-compatible caller intent.

The plan fixes those contracts first, then narrows fuzzy raw-body/header validation, hardens Responses WebSocket settlement, and deepens internal hotspots without adding new public frameworks.

### Problem Frame

The previous refactors correctly separated portable semantics, provider wire behavior, transport authority, host policy, and durable runtime state. A smaller set of implementation paths still contradict those decisions:

- correctness is sometimes inferred after the irreversible action has already happened, as with release tagging and MCP replay;
- two concepts with different trust levels are collapsed into one collection, as with model-visible tool definitions and host-executable bindings;
- durable invariants are split between the runtime and one store implementation, so a custom store observes a weaker contract;
- validation still searches arbitrary nested field names or exact model catalogs even when neither can change transport authority or wire encoding;
- actor and stream lifecycles can terminate without one typed, submission-aware outcome.

These are not requests for more validation. They are requests to move or delete validation until every remaining invariant is enforced by the lowest layer that can prove it.

### Requirements

#### Release and repository automation

- R1. A manual publish job runs only for `refs/heads/main`, and the checked-out commit must equal the current remote `main` commit before any publish or finalize command starts.
- R2. Publishing all workspace crates and creating the facade repository tag/GitHub Release are two ordered phases. Phase 2 cannot run unless Phase 1 exits successfully for the entire workspace.
- R3. Phase 1 publishes with git tags/releases disabled. Phase 2 uses release-plz `0.3.157` `git_only = true` with `publish = false` for `siumai`; no custom crate graph, publication order, or tag implementation is introduced.
- R4. The bounded crates.io 429 retry wrapper remains, but it accepts an explicit release-plz config path and retains only a bounded diagnostic tail for retry classification.
- R5. Release PR generation never receives a crates.io publish token. The publish job keeps token auth until all 26 crates have configured crates.io Trusted Publishing; OIDC migration is an operational follow-up and cannot produce a half-token/half-OIDC workflow.
- R6. PR CI includes docs and doctests, deterministic tests are not blanket-retried, and nextest has a measured finite global timeout. Do not create a provider-by-feature matrix or add a live credentialed gate.
- R7. Cargo, release-plz, the two existing bounded repository checkers, and the fixed test-workspace lanes remain the authorities. Do not reintroduce a workspace-version policy file, provider matrix script, or Rust source parser.

#### Runtime tool and durable identity

- R8. `LanguageRequest.tools` remains the caller's model-visible catalog. A tool loop merges it with the trusted local `ToolSet` specs in stable order instead of replacing it.
- R9. A duplicate tool name across caller-visible and local-bound catalogs fails before model I/O. Local execution resolves only calls backed by `ToolSet`; a caller-visible or provider-hosted spec never gains local execution authority merely by being shown to the model.
- R10. The merged model-visible catalog survives every model step, history projection, model transition, checkpoint, and resume without being regenerated from only the local bindings.
- R11. Tool binding and catalog fingerprints cover the complete canonical `ToolSpec`, including provider annotations. Annotation-only semantic changes invalidate approvals and durable resumes.
- R12. Fingerprint version constants and the durable execution ABI identifier are bumped together when U2 changes executable/catalog identity. Old identities fail explicitly; no inference-based migration is added.
- R13. Repeated `ProviderDeferred` observations with the same namespace use deterministic last-observation-wins semantics before checkpointing; unstable sort-and-dedup behavior is deleted.

#### MCP and transport safety

- R14. Siumai disables rmcp's transparent session reinitialization/retry for in-flight requests. A `tools/call` that reaches an expired or uncertain session returns an indeterminate execution outcome and is never automatically submitted a second time.
- R15. MCP limits include a maximum raw message/body size enforced before JSON deserialization for stdio frames, HTTP JSON responses, and HTTP error bodies. Existing post-decode schema/result limits remain defense in depth, not the first bound.
- R16. MCP public `Debug`, `Display`, and source chains do not expose endpoint query values, authorization challenges, remote bodies, raw backend error strings, or private tool payloads. Trusted callers may inspect an explicitly sensitive, bounded source.
- R17. The lifetime notification counter and permanent session-poisoning state are deleted. The bounded broadcast queue and catalog-staleness notifications remain the resource/lifecycle controls.
- R18. `LocalExplicit` resource URLs allow same-origin redirects by default. Cross-origin local/private redirects require an explicit destination grant; public URLs continue per-hop endpoint validation.
- R19. Cancellation and deadlines are propagated to an in-flight MCP request where rmcp exposes a cancellable handle. If the pinned API cannot cancel the remote operation, runtime still reports the effect as indeterminate and does not claim cancellation certainty.

#### Durable snapshots and stores

- R20. Serialized snapshot v6 data requires the `usage_settled` field. Missing or invalid settlement state fails deserialization instead of resetting prior aggregate usage.
- R21. Runtime validates `previous -> candidate` successor semantics before invoking `RunStore::compare_and_swap`. Store implementations own lease, run ID, revision, terminal-write rejection, and atomic CAS, but do not reimplement the runtime state machine.
- R22. Built-in and external stores observe the same successor contract. The built-in store no longer provides a stronger hidden semantic gate than the public `RunStore` seam.
- R23. Snapshot schema version and durable execution ABI are distinct concepts with documented bump rules. The stale `siumai-runtime-durable-v5` identifier is replaced while schema v6 remains unchanged unless the serialized shape actually changes beyond making an already-written field required.
- R24. `snapshot/model.rs`, `durable.rs`, and the completed-step branch in `engine.rs` are split only along existing responsibilities: wire/version, successor validation, checkpoint assembly, terminal state, provider suspension, tool preparation, and approval resolution. Public contracts stay stable unless an earlier requirement mandates a break.

#### Responses WebSocket lifecycle

- R25. A Responses WebSocket turn settles exactly once. An actor exit or turn-channel close before settlement produces one typed `UnexpectedEof`/failed terminal and then EOF; it is never interpreted as clean success.
- R26. The session owns and monitors the actor task. Actor panic, abort, socket failure, close, and caller drop have one explicit cleanup path with bounded task/channel lifetime.
- R27. Submission certainty is represented as `NotSubmitted` or `Indeterminate` for cancellation/timeout/send races. A payload accepted by the sender but not acknowledged cannot be reported as safely retryable cancellation.
- R28. Command enqueue, acknowledgement, session terminal, cancellation, and deadline share one call-control path. No queue or acknowledgement wait may outlive the caller's cancellation/deadline contract.

#### Provider intent and raw authority

- R29. Unknown, future, private, and aliased model IDs remain callable whenever the selected wire can encode the request. Exact model gates remain only when current official evidence proves that model identity changes field encoding or interpretation.
- R30. DeepSeek Responses accepts both currently documented V4 Responses models, including Pro, and its profile/catalog evidence matches runtime behavior.
- R31. Vertex Anthropic one-hour cache intent, structured output, and strict tools are not blocked by positive model allowlists. Feature intent reaches wire for future/private IDs; account/org/model eligibility errors remain provider responses.
- R32. Cohere `output_dimension` is not gated by an exact model list. The stable dimension value set, canonical/typed conflict, and returned-vector-length checks remain strict.
- R33. Gemini image and Veo product-eligibility checks that do not change the wire are removed. Portable-to-wire mapping, MIME constraints, enum shape, and cross-field duration/resolution relationships remain strict.
- R34. Raw provider-body validation enforces bounded JSON, exact canonical fields, exact transport/credential authority, and stable structural conflicts only. It does not recursively reject arbitrary nested names or reapply volatile typed-product limits.
- R35. Anthropic provider/compat/protocol raw-extra paths share one exact top-level canonical policy and allow bounded nested provider-native fields such as remote MCP `headers`, `url`, or `authorization_token` without granting HTTP transport authority.
- R36. OpenAI raw options may carry future nested prompt-cache and other provider-owned fields. Synthetic `prompt_cache_breakpoints` and canonical request/transport fields remain protected; typed/raw conflict behavior is deterministic and documented.
- R37. Transport credential-header collision checks use the exact common protected set plus headers declared by the selected credential applier. Substring heuristics such as any name containing `token` or `secret` are deleted.

#### Internal depth, facade, and documentation

- R38. Core owns validated exact provider-option target matching. Runtime stores target plus erased options and does not duplicate route/scope/family/instance comparison logic.
- R39. Repeated typed-option merge flows are consolidated only inside the owning provider crate with private helpers. No public merger trait or cross-provider utility crate is created.
- R40. OpenAI provider assembly is decomposed into private identity/profile validation, transport construction, WebSocket runtime, and support-manifest stages while preserving the public builder.
- R41. The facade exports `MessagePart` at the root/prelude and places OpenAI shared Chat/Responses enums in a semantically neutral or Chat-accessible namespace without duplicating types.
- R42. Drifted `.env.example`, Hajimi handoff material, stale release-rehearsal wording, unsupported support-policy promises, obsolete runtime tests, and dead helpers are deleted or rewritten. Exact provider claims remain provider-owned evidence rather than a second scripted registry.

### Product Key Decisions

- **Breaking cleanup is preferred to compatibility scaffolding during beta.** Delete obsolete gates, tests, aliases, and serialized identities when they encode the wrong ownership. Governs R8-R13, R20-R24, R29-R42. *(session-settled: user-directed — chosen over retaining compatibility layers that preserve incorrect behavior.)*
- **Official provider contracts are primary; the local AI SDK checkout is secondary.** Use reference code for fixture ideas, not for Siumai's public module/type architecture. Governs R29-R37. *(session-settled: user-directed — chosen over mechanically matching the TypeScript SDK.)*
- **Repository automation stays thin.** Reuse release-plz, Cargo, nextest, Clippy, rustdoc, and bounded orchestration; delete or avoid policy scripts that become duplicate authorities. Governs R1-R7, R42. *(session-settled: user-directed — chosen over expanding policy JSON and custom validation frameworks.)*

### Key Flows

- F1. **Atomic release finalization**
  - Trigger: a maintainer manually dispatches the publish workflow from `main`.
  - Steps: preflight verifies the exact remote-main commit; publish-only release-plz config publishes the complete workspace with bounded 429 retries; only on success does the git-only finalize config create the `siumai` tag and GitHub Release.
  - Outcome: no public repository release exists for a partially published workspace.
  - Covered by: R1-R7.

- F2. **Model-visible tool catalog with trusted execution bindings**
  - Trigger: an Agent/ToolLoop receives caller/provider-visible specs and a host-owned local ToolSet.
  - Steps: runtime validates no name collision, forms one stable visible catalog, sends it on every step, and resolves returned local calls only through ToolSet.
  - Outcome: hosted/unbound tools stay visible but non-executable; local tools stay visible and executable under host policies.
  - Covered by: R8-R13.

- F3. **Durable tool-loop resume**
  - Trigger: a run checkpoints after a tool-visible or approval-relevant state transition and later resumes.
  - Steps: canonical tool identity includes annotations; runtime validates successor semantics before CAS; schema and execution ABI checks fail closed; usage resumes without replacement or double settlement.
  - Outcome: resumed execution has the same model-visible catalog, approvals, usage, and exactly-once tool semantics as uninterrupted execution.
  - Covered by: R10-R13, R20-R24.

- F4. **Never-replay MCP tool call**
  - Trigger: a local runtime dispatch invokes an MCP tool and the remote session expires or the response becomes uncertain after submission.
  - Steps: the transport does not reinitialize/replay the call; pre-decode bounds protect the response path; the runtime receives a sanitized indeterminate failure and does not retry without explicit host policy.
  - Outcome: one logical dispatch causes at most one remote `tools/call` submission.
  - Covered by: R14-R19.

- F5. **Responses WebSocket turn settlement**
  - Trigger: a turn is enqueued, sent, cancelled, timed out, or interrupted by actor/socket failure.
  - Steps: enqueue/send/ack share cancellation and deadline; submission state advances explicitly; the actor owner produces one terminal; un-settled channel closure is converted to failure.
  - Outcome: callers can distinguish safe non-submission from uncertain submission and never see a clean EOF without a terminal.
  - Covered by: R25-R28.

- F6. **Future model with explicit provider intent**
  - Trigger: a caller selects a future/private model plus a typed or bounded raw provider option.
  - Steps: codec/provider validates only stable shape, relationship, resource, canonical, and authority constraints; no dated model allowlist or nested-name heuristic rewrites the intent.
  - Outcome: intent reaches final wire or fails with a typed invariant the owning layer can prove.
  - Covered by: R29-R37.

### Acceptance Examples

- AE1. Covers F1. If the last independently publishable crate fails permanently, the workflow exits without creating `vX.Y.Z` or a GitHub Release. Re-running Phase 1 publishes only still-unpublished crates, then Phase 2 finalizes exactly once.
- AE2. Covers R1. Dispatching the release job from a non-main ref fails before installation/authentication/publish work; moving remote `main` after checkout also fails the commit preflight.
- AE3. Covers F2. A request with one Anthropic hosted-tool annotation and an empty local ToolSet reaches Anthropic wire unchanged and never enters local approval/execution.
- AE4. Covers F2. A caller-visible spec plus a distinct local binding are both sent on the first and later model steps; a same-name pair fails before the model transport is called.
- AE5. Covers F3. Changing only a tool provider annotation changes both binding/catalog identity, rejects an old approval/resume, while JSON object key reordering does not change identity.
- AE6. Covers F4. A test MCP server performs a side effect, then returns session-expired/404; Siumai issues one `tools/call`, returns indeterminate, and does not reinitialize/replay it.
- AE7. Covers R15-R16. Oversized unterminated stdio JSON, chunked HTTP JSON, and non-success body are cut off near the configured raw limit; canary URL/body values appear only behind an explicit sensitive accessor.
- AE8. Covers F3. A serialized v6 snapshot missing `usage_settled` is rejected; two calls separated by snapshot/resume aggregate usage exactly once for completed, failed-partial, and cancelled-partial cases.
- AE9. Covers R21-R22. A deliberately invalid successor is rejected by the durable checkpoint port before a fake external store's CAS method is invoked; built-in and fake stores otherwise observe the same CAS contract.
- AE10. Covers F5. Aborting the WebSocket actor after a turn starts yields one failed terminal and then EOF; recording a payload before cancellation yields `Indeterminate`, while cancelling before enqueue yields `NotSubmitted`.
- AE11. Covers F6. DeepSeek V4 Pro reaches `/responses`; future Vertex Anthropic cache/structured intent, future Cohere dimension, and future Gemini image/Veo options reach their exact bodies.
- AE12. Covers R29-R33. A truly structural invalid value still fails locally: unsupported Cohere dimension, unencodable Gemini pixel size, non-strict constrained output, or invalid provider wire enum.
- AE13. Covers R34-R37. Nested MCP/provider headers and URLs survive bounded raw encoding, while model/input/messages/tools, endpoint, Authorization/API key, credential-declared headers, retry, timeout, and method remain protected across case/separator variants.
- AE14. Covers R13. Two deferred observations for the same `ProviderDeferred` namespace persist only the later payload in the checkpoint.
- AE15. Covers R41-R42. A facade-only contract constructs annotated `MessagePart` content and full Chat options without importing internal crates; stale handoff/rehearsal files and obsolete bad-behavior tests are absent.

### Success Criteria

- The next manual release cannot tag a partially published workspace or run from a non-main commit.
- Runtime preserves all model-visible tool intent while keeping executable bindings host-owned and durable identities annotation-complete.
- One runtime MCP dispatch cannot be replayed invisibly by rmcp, and raw MCP inputs are bounded before deserialization.
- Valid v6 snapshots cannot silently reset usage, and custom stores do not weaken successor validation.
- Responses WebSocket turns always produce one submission-aware terminal outcome.
- The audited DeepSeek, Vertex Anthropic, Cohere, Gemini, OpenAI raw, Anthropic raw, and transport-header paths follow ADR 0015's caller-first ownership rule.
- No new public cross-provider trait, capability matrix, provider-by-model test matrix, release parser, or credentialed CI gate is added.

### Scope Boundaries

#### Included

- Release-plz workflow/config/retry-wrapper refactor and release/documentation tests.
- Runtime model-visible tool merge, full-spec fingerprinting, provider-deferred replacement, durable successor ownership, and bounded internal module extraction.
- MCP replay, pre-decode resource limits, errors, notification lifecycle, and cancellation semantics supported by the pinned dependency.
- OpenAI Responses WebSocket actor/turn settlement and submission certainty.
- DeepSeek Responses, Vertex Anthropic cache/structured output, Cohere dimensions, and Gemini image/Veo eligibility-gate cleanup.
- OpenAI/Anthropic raw body authority and transport credential-header collision narrowing.
- Facade export fixes and deletion/update of directly related stale docs/tests/helpers.

#### Deferred to Follow-Up Work

- OpenAI image, transcription, and speech model-branch audit; MiniMax media/model eligibility audit. These may contain real dialect differences and require a separate field-by-field evidence pass.
- An authenticated/MACed snapshot envelope and a pre-deserialization public snapshot byte codec. The current `RunStore` remains a trusted application boundary.
- Cross-origin local-resource redirect grants beyond a small explicit allowlist API.
- Provider-wide option-merger consolidation outside OpenAI/Gemini, or any cross-provider merger abstraction.
- Empty facade default features, workspace dependency declaration normalization, Deepgram ambient `from_env`, and unrelated dependency upgrades.
- Crates.io Trusted Publishing activation itself; repository changes may land only after all workspace crates are configured by maintainers.
- Credentialed live provider tests, sub2api canaries, or live release gates.

### Dependencies and Sources

- `AGENTS.md`
- `docs/adr/0010-provider-plane-and-host-control-plane.md`
- `docs/adr/0012-provider-annotations-follow-semantic-nodes.md`
- `docs/adr/0015-validation-ownership-and-forward-compatibility.md`
- `docs/architecture/overview.md`
- `docs/architecture/transport-contract.md`
- `docs/releasing.md`
- release-plz `0.3.157` source, especially `PackageConfig::{git_only,publish}` and the git-only release fixtures; verified 2026-08-14.
- rmcp `3.1.2` source, especially `StreamableHttpClientTransportConfig::reinit_on_expired_session`, `JsonRpcMessageCodec`, and reqwest streamable HTTP response handling; verified 2026-08-14.
- DeepSeek Responses API guide: `https://api-docs.deepseek.com/guides/responses_api/`; verified 2026-08-14.
- Vertex Anthropic prompt caching: `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/prompt-caching`; verified 2026-08-14.
- Vertex Anthropic structured outputs: `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/structured-outputs`; verified 2026-08-14.
- Cohere Embed v2 reference: `https://docs.cohere.com/v2/reference/embed`; verified 2026-08-14.
- Gemini image generation: `https://ai.google.dev/gemini-api/docs/image-generation`; verified 2026-08-14.
- Gemini Interactions API: `https://ai.google.dev/api/interactions-api`; verified 2026-08-14.

---

## Planning Contract

### Key Technical Decisions

- KTD1. **Finalize releases in a separate release-plz git-only phase.** Keep normal `release-plz.toml` for release-PR calculation, add explicit publish-only and finalize configs, and pass the config path through the existing retry wrapper. This uses the pinned tool's native `git_only = true` plus `publish = false` seam and avoids a custom publication engine. Governs R1-R7. *(session-settled: user-directed — chosen over additional release policy scripts.)*
- KTD2. **Separate visibility from executability.** `LanguageRequest.tools` is the visible catalog; `ToolSet` is the trusted binding map. Runtime derives a private merged catalog once per run and never uses visibility as proof of execution authority. Governs R8-R13.
- KTD3. **Fingerprint wire-semantic tool identity.** Canonically hash the whole `ToolSpec`, including annotations, alongside host binding policy. Bump both fingerprint domain versions and the durable execution ABI so old approval/resume identities fail explicitly. Governs R10-R12.
- KTD4. **No transparent replay below runtime.** Disable rmcp session reinitialization and classify uncertain `tools/call` outcomes as indeterminate. Discovery may be retried only through an explicit caller/runtime action, not by the transport worker. Governs R14, R19.
- KTD5. **Bound raw bytes before MCP parsing.** Use a Siumai-owned bounded stdio transport built on rmcp's public transport/message types and a bounded HTTP client implementing rmcp's client trait. Do not fork rmcp or add post-hoc JSON tree accounting as the primary defense. Governs R15-R17.
- KTD6. **Runtime owns successor semantics; stores own CAS.** The checkpoint port validates semantic progression immediately before calling the store. Stores remain free to recheck run/terminal/revision safety but do not define a second execution state machine. Governs R20-R24.
- KTD7. **WebSocket session state records submission certainty.** State transitions are `Queued -> NotSubmitted -> Indeterminate -> Settled`; public failure/cancellation exposes the last safe classification rather than guessing from task cancellation. Governs R25-R28.
- KTD8. **Delete product eligibility, retain wire interpretation.** Provider units remove exact-model/dated availability gates when the JSON/protocol representation is unchanged. They retain only codec shape, resource, relationship, and documented model-dependent interpretation branches. Governs R29-R33. *(session-settled: user-directed — chosen over maintaining Siumai capability allowlists.)*
- KTD9. **Raw body authority is path-exact, not name-fuzzy.** Providers protect their exact top-level canonical body fields; transport protects its exact request/auth fields. Nested provider data is inert with respect to transport. Governs R34-R37.
- KTD10. **Deepen private modules after behavioral contracts are fixed.** Extract completed-step stages, snapshot/successor modules, exact-target matching, provider merge helpers, and OpenAI assembly only after their behavioral unit is green. No extraction-by-line-count or new public framework. Governs R24, R38-R40.

### High-Level Technical Design

The diagrams describe ownership and ordering; they do not prescribe exact Rust type names.

```text
LanguageRequest.tools (caller/provider-visible)
                 +
ToolSet.specs()   (host-visible + locally executable)
                 |
        collision validation
                 |
        stable merged visible catalog --------------------------+
                 |                                              |
      every model request / projection                    snapshot continuation
                 |                                              |
        provider wire encoder                            durable resume parity

Returned ToolCall
      |
ToolSet.resolve(name) -- no binding --> typed local preparation failure
      |
trusted binding -> approval -> checkpoint -> dispatch
```

```text
manual workflow_dispatch on main
          |
remote-main commit preflight
          |
Phase 1: release-plz publish config
  - workspace publishing enabled
  - all git tags/releases disabled
  - bounded 429 retries
          |
          +-- any crate failure --> stop; no tag/release
          |
Phase 2: release-plz finalize config
  - only siumai release enabled
  - git_only = true
  - publish = false
  - v{{version}} tag + GitHub Release
```

```text
runtime checkpoint candidate
          |
candidate.validate()
          |
previous.validate_successor(candidate)   <- runtime state semantics
          |
RunStore.compare_and_swap                <- lease/run/revision/atomicity
          |
new StoredRun revision
```

### Sequencing

1. U1 and U4 are independent P0 work and may be developed first, but both must land before another release attempt.
2. U2 follows U4's durable identity decisions so one snapshot/ABI break covers tool annotations and usage/successor semantics together.
3. U3 is independent from U2/U4 at the crate level; it should land before runtime begins relying on MCP `NeverReplay` as a durable guarantee.
4. U5 is independent provider work and can be split into provider-scoped commits while retaining one evidence rule.
5. U6 must follow U3's exact trust-boundary language so provider-body headers are not confused with transport credential headers.
6. U7 is independent of U5/U6 but should land before marking Responses WebSocket stable beyond experimental/session scope.
7. U8 is last: it extracts modules and fixes facade/docs only after behavioral contracts stop moving.

### System-Wide Impact

- **Public API:** MCP limits/errors, facade exports, and WebSocket failure/submission types may break. ToolLoop construction stays familiar; the semantic break is that request tools are preserved rather than discarded.
- **Durable state:** Tool fingerprint domains and durable execution ABI change. Existing snapshots/approvals tied to the old catalog identity fail closed. Snapshot schema remains v6 if only requiredness changes.
- **Transport:** Local redirect policy and exact credential-header collisions become narrower but stronger. Provider-body fields cannot mutate transport because the carrier boundary stays typed.
- **Provider protocols:** Provider eligibility changes affect final request admission, not encoding. Each removal requires final-wire fixtures for known and future model IDs plus a structural negative.
- **Release operations:** The workflow becomes restartable in two phases. An interrupted publish may leave a partially published crates.io workspace, but no repository release; rerunning publish converges before finalize.
- **Security/privacy:** MCP remote source material remains explicitly sensitive; no new raw payload is admitted to default diagnostics or durable snapshots.

### Risks and Mitigations

| Risk | Mitigation |
|---|---|
| release-plz git-only finalize calculates an unexpected version/tag | Pin `0.3.157`; add config parse/dry-run fixtures against a temporary workspace with an existing prior tag; assert output contains only the facade release before enabling the workflow step. |
| publish succeeds but finalize transiently fails | Keep phases separately rerunnable; finalize is git-only and must reject an already-existing mismatched tag while treating the exact existing release as complete. |
| preserving caller-visible tools causes the model to call an unbound tool | Keep the current local resolver fail-closed and document that ToolLoop executes only ToolSet bindings; hosted provider tools remain ProviderOpaque/provider-owned. |
| fingerprint change strands beta snapshots | Bump the execution ABI and migration guide explicitly; do not attempt to reinterpret old annotation-blind identities. |
| custom rmcp adapters duplicate too much upstream code | Implement only bounded I/O and required trait adaptation; keep JSON-RPC/session logic in rmcp; reassess replacement only if public traits cannot enforce pre-parse bounds. |
| removing model gates sends an option a provider rejects | This is the intended caller-first contract; preserve provider response classification and retain true structural negatives locally. |
| narrowing fuzzy security checks permits a credential override | Prove isolation with a negative matrix at provider and transport boundaries, including case/separator variations and credential-applier-declared names. |
| actor monitoring creates double WebSocket terminals | Route all actor/session exits through the existing exactly-once terminal owner and assert terminal-then-EOF behavior under races. |

### Open Questions

No launch-blocking question remains. The following are implementation checks with predetermined fallbacks:

- If rmcp's public HTTP client trait cannot bound a non-SSE JSON response before parsing without reproducing session logic, U3 may replace only the HTTP transport adapter inside `siumai-mcp`; it must not fork the protocol model or worker state machine.
- If crates.io Trusted Publishing has not been configured for every workspace crate, U1 keeps `CARGO_REGISTRY_TOKEN` only in the publish phase and records OIDC as deferred; it must not silently mix authentication modes.

---

## Implementation Units

### U1. Make workspace publication precede repository release

- **Goal:** Make a manual release main-only, restartable, and atomic at the repository tag/release boundary.
- **Requirements:** R1-R7; F1; AE1-AE2; KTD1.
- **Files:**
  - Modify `.github/workflows/release-plz.yml`
  - Modify `release-plz.toml`
  - Create `config/release/release-plz-publish.toml`
  - Create `config/release/release-plz-finalize.toml`
  - Modify `scripts/release_plz_release_with_retry.py`
  - Modify `scripts/tests/test_release_plz_release_with_retry.py`
  - Modify `.github/workflows/ci.yml`
  - Modify `.config/nextest.toml`
  - Modify `docs/releasing.md`
  - Modify `scripts/README.md`
- **Approach:**
  - Add both workflow-level and checked-out-commit `main` preflights before credentials or release commands.
  - Keep `release-plz.toml` as release-PR configuration; the publish config disables every git tag/release while preserving workspace publication; the finalize config disables/restricts other packages and configures `siumai` with `git_only = true`, `publish = false`, and the existing tag/release templates.
  - Extend the existing Python wrapper with a required/explicit config path, same secret-to-`GIT_TOKEN` mapping, dry-run mode, 429 retry only for the publish phase, and a bounded rolling output tail. Do not parse package graphs or publication order.
  - Split the workflow into publish and finalize steps/jobs with a hard success dependency. Remove unused release-job semver tooling and release-PR publish credentials.
  - Add PR docs/doctests, remove blanket nextest retry, and choose a finite timeout from current serial full-lane timing plus margin.
- **Test scenarios:**
  - Wrapper forwards real/dry-run/config arguments without putting credentials in argv.
  - 429 then success retries publish with the same config; non-429 and finalize failures never retry as crates.io rate limits.
  - A temporary release-plz fixture proves publish config produces no tag and finalize config publishes no crate but produces the expected facade tag.
  - Static workflow contract verifies non-main dispatch and remote-main mismatch stop before release.
  - CI docs/doctest path runs for pull requests; deterministic failure is not retried.
- **Verification:** Script unit suite passes; release-plz config fixture shows disjoint outputs; YAML parses; package/checker tests remain green; docs describe the actual two phases and restart behavior.

### U4. Correct durable usage and successor ownership

- **Goal:** Make snapshot/resume semantics identical across stores and reject malformed settlement state.
- **Requirements:** R20-R24; F3; AE8-AE9; KTD6.
- **Files:**
  - Modify `siumai-runtime/src/run.rs`
  - Modify `siumai-runtime/src/durable.rs`
  - Modify `siumai-runtime/src/snapshot/model.rs`
  - Modify `siumai-runtime/src/snapshot/store.rs`
  - Modify `siumai-runtime/src/snapshot/mod.rs`
  - Modify `siumai-runtime/tests/durable_tool_loop_contract.rs`
  - Modify `docs/architecture/overview.md`
  - Modify `docs/migration/siumai-next.md`
- **Approach:**
  - Remove the serde default from `usage_settled` and keep it private but required in v6 wire.
  - Route every runtime CAS, including recovery, through one private commit entry that accepts the prior snapshot and validates `previous -> candidate` before calling the store. Keep store-side run/revision/terminal checks; delete duplicate semantic successor validation from `InMemoryRunStore` after parity fixtures exist.
  - Rename/document `SnapshotEngineVersion` semantics as durable execution ABI at the rustdoc/constant level without coupling it to crate semver. Coordinate its bump with U2.
  - Extract successor validation and checkpoint assembly into private snapshot modules only after behavior is covered; do not change serialized field names except the requiredness already specified.
- **Test scenarios:**
  - v6 JSON missing `usage_settled` fails before resume.
  - completed, failed-partial, and cancelled-partial usage before and after resume aggregate exactly once.
  - invalid step/history/budget/usage/provider/execution-log successor fails before fake store CAS.
  - recovery with an invalid successor fails before the fake store's CAS method is invoked.
  - built-in and fake external store conformance accept the same valid chain and reject CAS/run/terminal violations consistently.
- **Verification:** Runtime focused nextest and JSON-schema/no-default feature checks pass; v6 round trips remain stable; no store implementation must call a private state-machine validator.

### U2. Separate model-visible tools from local execution identity

- **Goal:** Preserve caller/provider-visible tools across runtime execution while binding local approvals/resumes to the complete annotated spec.
- **Requirements:** R8-R13; F2-F3; AE3-AE5, AE14; KTD2-KTD3.
- **Dependencies:** U4 for the final durable execution ABI value and snapshot migration wording.
- **Files:**
  - Modify `siumai-runtime/src/engine.rs`
  - Modify `siumai-runtime/src/tool/binding.rs`
  - Modify `siumai-runtime/src/tool_loop.rs`
  - Modify `siumai-runtime/src/agent.rs`
  - Modify `siumai-runtime/src/durable.rs`
  - Modify `siumai-runtime/src/snapshot/model.rs`
  - Modify `siumai-runtime/src/approval/context.rs`
  - Modify `siumai-runtime/tests/tool_contract.rs`
  - Modify `siumai-runtime/tests/tool_loop_contract.rs`
  - Modify `siumai-runtime/tests/durable_tool_loop_contract.rs`
  - Modify `siumai-runtime/tests/model_switching_contract.rs`
- **Approach:**
  - Add one private visible-catalog preparation path that preserves caller order and appends local specs in ToolSet's deterministic order after collision validation. Use it at establish, seeded establish, later steps, model switches, and resume.
  - Keep `ToolSet` as the only local resolution map. Do not add executable placeholders for caller/provider-visible specs.
  - Canonically hash provider annotations in binding identity. Derive the catalog fingerprint from the merged visible catalog plus exact local binding identities so changing either visible provider intent or executable host policy invalidates durable state.
  - Replace provider-deferred vector sort/dedup with insertion-ordered last-wins normalization keyed by namespace.
  - Bump binding/catalog fingerprint domains and the U4 execution ABI; delete tests that expect caller tools to disappear.
- **Test scenarios:**
  - Agent with empty ToolSet preserves an Anthropic hosted tool through first and subsequent model request bodies.
  - Caller-visible and local tools coexist across model switching and durable resume; local ToolCall resolves, unbound ToolCall fails without executing.
  - Same-name collision fails pre-transport.
  - Annotation-only change alters identity; canonical key reordering does not.
  - Old fingerprint/ABI snapshot and approval fail explicitly.
  - Deferred `queued -> in_progress` persists `in_progress` only.
- **Verification:** Runtime tool/durable/model-switching suites pass; Anthropic request fixture proves hosted annotation reaches wire; no assignment replaces request tools with only `ToolSet::specs()`.

### U3. Remove hidden MCP replay and enforce pre-decode bounds

- **Goal:** Make MCP honor runtime replay certainty, raw resource limits, and sanitized error contracts.
- **Requirements:** R14-R19; F4; AE6-AE7; KTD4-KTD5.
- **Files:**
  - Modify `siumai-mcp/Cargo.toml`
  - Modify `siumai-mcp/src/config.rs`
  - Modify `siumai-mcp/src/client.rs`
  - Modify `siumai-mcp/src/error.rs`
  - Create `siumai-mcp/src/transport.rs` or equivalent private adapter module
  - Modify `siumai-mcp/src/lib.rs`
  - Add/modify MCP direct contract tests in `siumai-mcp/src/` or `siumai-mcp/tests/`
  - Modify `siumai-transport/src/resource.rs`
  - Modify `siumai-transport/tests/transport_contract.rs`
  - Modify `docs/architecture/transport-contract.md`
- **Approach:**
  - Build streamable HTTP transport from explicit rmcp config with reinitialization disabled.
  - Add `max_message_bytes` to MCP limits. For stdio, own child handles and use a bounded rmcp-compatible transport; for HTTP, implement the public rmcp client trait with content-length plus streaming caps for JSON/error bodies and the existing bounded SSE behavior.
  - Replace string-bearing connection/list/close errors with typed phases plus `SensitiveErrorSource`; keep public messages static and bounded.
  - Delete notification lifetime counters/overflow poison while retaining bounded broadcast lag semantics and catalog invalidation.
  - Propagate cancellation where available and always map uncertain transport completion to `EffectCertainty::Indeterminate`.
  - Make LocalExplicit redirects same-origin unless an explicit local destination grant matches.
- **Test scenarios:**
  - Side-effect then 404/session expiry produces one remote call and indeterminate result.
  - Oversized stdio line, JSON success, JSON-RPC error, plain error body, and SSE event fail near raw limits.
  - endpoint query/body/auth-challenge canaries are absent from default diagnostics/source chain and available only through explicit sensitive inspection.
  - More notifications than the former lifetime cap do not poison a drained session; queue lag remains observable.
  - local A -> local B redirect is blocked before B receives a request; same-origin redirect succeeds.
  - cancellation either sends MCP cancellation or returns indeterminate without replay.
- **Verification:** MCP nextest/clippy and transport resource tests pass serially; dependency features remain narrow; no forked rmcp protocol types or duplicated session worker exist.

### U5. Delete volatile provider eligibility gates

- **Goal:** Align audited providers with caller-first explicit intent while retaining true wire and structural validation.
- **Requirements:** R29-R33; F6; AE11-AE12; KTD8.
- **Files:**
  - Modify `siumai-provider-deepseek/src/language.rs`
  - Modify `siumai-provider-deepseek/src/models.rs`
  - Modify `siumai-provider-deepseek/tests/language_contract.rs`
  - Modify `siumai-provider-google-vertex/src/providers/anthropic_vertex/models.rs`
  - Modify `siumai-provider-google-vertex/src/providers/anthropic_vertex/request_policy.rs`
  - Modify `siumai-provider-google-vertex/src/providers/anthropic_vertex/profile.rs`
  - Modify `siumai-provider-google-vertex/src/providers/anthropic_vertex/tests.rs`
  - Modify `siumai-provider-cohere/src/models.rs`
  - Modify `siumai-provider-cohere/src/configured/model.rs`
  - Modify `siumai-provider-cohere/src/provider_options/cohere.rs`
  - Modify `siumai-provider-cohere/src/configured/tests.rs`
  - Modify `siumai-provider-gemini/src/image.rs`
  - Modify `siumai-provider-gemini/src/veo.rs`
  - Modify `siumai-provider-gemini/src/options.rs`
  - Modify relevant provider evidence under `docs/providers/`
- **Approach:**
  - DeepSeek: remove the runtime known-chat/not-known-responses rejection, add V4 Pro to Responses advisory/evidence, and delete unused closed-list helpers.
  - Vertex Anthropic: remove positive allowlists for one-hour cache and structured output/strict tools. Preserve feature-derived wire projection and protocol structural errors.
  - Cohere: move the dimension value set to one provider-owned constant/validator and remove exact model eligibility.
  - Gemini: remove image/Veo exact-model product eligibility checks; preserve portable mapping, MIME/enums, and duration/resolution cross-field invariants.
  - For each provider, use one known model, one future/private baseline, and one true structural negative; do not replace allowlists with prefix guessing or warnings.
- **Test scenarios:**
  - DeepSeek V4 Pro and future ID reach Responses final wire; unsupported Responses fields still fail.
  - Future Vertex ID encodes 1h cache at message/content/tool targets plus structured output and strict tools; non-strict constrained output still fails.
  - Future Cohere ID encodes 512; 2048, canonical/provider conflict, and response length mismatch fail.
  - Gemini future/known values reach exact image/Veo wire; invalid pixel mapping, MIME, and duration relationship fail.
- **Verification:** Each affected provider's all-feature nextest/clippy lane passes serially; profile/support evidence dates and model claims match the final fixtures.

### U6. Narrow raw body and credential authority validation

- **Goal:** Remove fuzzy/provider-product validation from raw escape hatches without weakening canonical request, credential, or transport isolation.
- **Requirements:** R34-R37; F6; AE13; KTD9.
- **Dependencies:** U3 for the final transport/MCP authority vocabulary.
- **Files:**
  - Modify `siumai-protocol-anthropic/src/messages/options.rs`
  - Modify `siumai-protocol-anthropic/src/messages/request.rs`
  - Modify `siumai-anthropic-compatible/src/options.rs`
  - Modify `siumai-provider-anthropic/src/options.rs`
  - Modify Anthropic protocol/compat/provider option contract tests
  - Modify `siumai-provider-openai/src/configured/provider.rs`
  - Modify `siumai-provider-openai/src/configured/model.rs`
  - Modify `siumai-provider-openai/src/configured/options.rs`
  - Modify `siumai-transport/src/auth.rs`
  - Modify `siumai-transport/src/replay.rs`
  - Modify `siumai-transport/src/request.rs`
  - Modify `siumai-transport/tests/transport_contract.rs`
  - Modify `docs/adr/0012-provider-annotations-follow-semantic-nodes.md`
  - Modify `docs/adr/0015-validation-ownership-and-forward-compatibility.md`
- **Approach:**
  - Remove recursive Anthropic `token/url/header/authorization` name checks at provider, compat, and protocol layers. Retain exact top-level canonical fields, JSON bounds, and provider option relationships in one owner.
  - Restrict OpenAI raw validation to final wire shape, exact canonical/transport fields, and bounds. Remove prompt-cache provider fields and volatile typed-product limits from the raw gate; define raw-last overlay conflict semantics.
  - Replace transport header substring classification with exact common headers plus credential-applier-declared protected names.
  - Update ADR 0012's stale call-option/protected-field wording and make ADR 0015's raw authority table match the implemented exact paths.
- **Test scenarios:**
  - Bounded nested `headers`, `url`, `authorization_token`, future cache options, and future enum strings reach final provider body.
  - Canonical model/input/messages/tools/stream fields and endpoint/method/retry/timeout/auth/credential names fail across case and separator variations.
  - A harmless header such as `x-token-count-mode` is allowed unless the selected credential applier explicitly owns it.
  - Debug/error sentinels never expose accepted nested values.
- **Verification:** Anthropic protocol/compat/provider, OpenAI provider, and transport contract suites pass; duplicate protected lists/recursive heuristics are absent; raw options still cannot alter RequestPlan or CredentialPatch.

### U7. Make Responses WebSocket settlement submission-aware

- **Goal:** Guarantee one terminal and truthful replay certainty for every Responses WebSocket turn.
- **Requirements:** R25-R28; F5; AE10; KTD7.
- **Files:**
  - Modify `siumai-provider-openai/src/configured/responses_websocket.rs`
  - Modify `siumai-core/src/experimental/session.rs` only if the portable experimental session outcome needs a submission-certainty type
  - Modify `siumai-provider-openai/src/configured/mod.rs` and `siumai/src/providers/openai.rs` for curated exports if public types change
  - Modify `siumai-provider-openai/CHANGELOG.md`
  - Modify `docs/migration/siumai-next.md`
- **Approach:**
  - Retain and monitor the actor JoinHandle; make session shutdown await/abort it through one owner.
  - Track per-command submission state before and after sender acceptance. Cancellation/deadline selects use that state to return NotSubmitted or Indeterminate.
  - Convert an un-settled response-channel close to one typed failure terminal, then EOF. Route actor panic/socket error/close through the same terminal sender.
  - Reuse one helper for enqueue, ack, session terminal, cancellation, and deadline; keep command/response queues bounded.
- **Test scenarios:**
  - Actor abort/panic before terminal -> exactly one failure then EOF.
  - Cancel before enqueue -> NotSubmitted; sender records payload then blocks -> Indeterminate.
  - Deadline/cancel/actor-terminal races still yield one terminal and cancel actor resources on drop.
  - Queue-full and ack-wait paths honor caller deadline/cancellation.
- **Verification:** OpenAI provider WebSocket tests, no-default feature check, facade export contract, and Clippy pass; no receiver path returns clean EOF before settlement.

### U8. Deepen hotspots and clean public/documentation seams

- **Goal:** Reduce the cost of future maintenance after the behavioral contracts have stabilized, without adding new public abstractions.
- **Requirements:** R24, R38-R42; AE15; KTD10.
- **Dependencies:** U1-U7 behavior is settled; relevant unit tests are green before extraction.
- **Files:**
  - Modify `siumai-core/src/options.rs`
  - Modify `siumai-runtime/src/options.rs`
  - Modify `siumai-runtime/src/engine.rs` and create private completed-step modules as warranted
  - Modify `siumai-runtime/src/snapshot/` private module layout
  - Modify `siumai-provider-openai/src/configured/provider.rs`
  - Modify private OpenAI/Gemini option modules that still duplicate merge flow
  - Modify `siumai/src/lib.rs`
  - Modify `siumai/src/prelude.rs`
  - Modify `siumai/src/providers/openai.rs`
  - Modify `siumai/tests/facade_contract.rs`
  - Delete `.env.example` if no maintained consumer remains; otherwise update it to match the supported configuration paths
  - Delete `docs/migration/hajimi-adapter-handoff.md`
  - Modify `docs/README.md`
  - Modify `docs/providers/support-policy.md`
  - Modify `docs/releasing.md`
- **Approach:**
  - Move exact target validation/matching to one core-owned validated target/patch seam; runtime stops storing duplicate identity fields. Delete test-only unconsumed accessors if no production caller remains.
  - Extract provider suspension, tool preparation, and approval resolution from the completed-step branch into private state objects with explicit inputs/outputs.
  - Finish U4's private snapshot split by responsibility while retaining wire owners and public exports.
  - Add crate-private option decode/merge helpers only where two or more current families repeat the exact flow; keep provider-specific semantics local.
  - Split OpenAI builder assembly into private stages without changing the public builder.
  - Add facade exports/compile contracts and delete stale repository documentation instead of preserving historical aliases.
- **Test scenarios:**
  - Exact target selection behavior and bounds remain unchanged across core/runtime integration fixtures.
  - Completed, suspended-provider, approval-required, preparation-failure, and no-tool steps produce the same public events/checkpoints after extraction.
  - Provider typed options retain merge/replace/clear/raw-reject behavior after helper consolidation.
  - Facade-only code builds annotated MessagePart and Chat shared enums.
  - Documentation/search checks find no removed handoff types, beta.9 rehearsal claims, or old test commands.
- **Verification:** Focused core/runtime/OpenAI/Gemini/facade suites pass, then workspace fmt/nextest/clippy/docs/MSRV/package gates pass serially; the diff contains no abandoned compatibility scaffolding or unrelated formatting churn.

---

## Verification Contract

### Per-Unit Gates

- U1: `python3 -B -m unittest discover -s scripts/tests -p 'test_*.py'`; release-plz `0.3.157` temporary-workspace config fixtures; YAML parse/action validation; existing package and architecture checkers.
- U4/U2/U8 runtime slices: `cargo nextest run -p siumai-runtime --all-features --test-threads 1`; `cargo check -p siumai-runtime --no-default-features --features json-schema -j 1`; `cargo clippy -p siumai-runtime --all-targets --all-features -j 1 -- -D warnings`.
- U3: `cargo nextest run -p siumai-mcp --all-features --test-threads 1`; `cargo clippy -p siumai-mcp --all-targets --all-features -j 1 -- -D warnings`; focused `siumai-transport` resource tests.
- U5: run each changed provider's all-feature nextest and Clippy serially; run only its directly affected protocol/compat crate when the final wire owner changed.
- U6: serial nextest/Clippy for `siumai-protocol-anthropic`, `siumai-anthropic-compatible`, `siumai-provider-anthropic`, `siumai-provider-openai`, and `siumai-transport`.
- U7: `cargo nextest run -p siumai-provider-openai --all-features --test-threads 1`; OpenAI Responses WebSocket no-default facade feature check; all-target/all-feature Clippy.
- U8: facade contract suite plus the focused crates whose private modules were extracted.

### Integration and Release Gates

Run Cargo commands serially in the shared target directory:

```text
cargo fmt --all -- --check
python3 -B -m unittest discover -s scripts/tests -p 'test_*.py'
python3 -B scripts/check_workspace_boundaries.py
python3 -B scripts/check_package_file_list.py
python3 -B scripts/test-workspace.py flagship --runner nextest
python3 -B scripts/test-workspace.py full --runner nextest
cargo clippy --workspace --all-targets --all-features -j 1 -- -D warnings
cargo check -p siumai --no-default-features --lib -j 1
cargo check -p siumai --no-default-features --features all-providers --lib -j 1
cargo check -p siumai --no-default-features --features openai-responses-websocket,openai-realtime --lib -j 1
cargo doc --workspace --all-features --no-deps -j 1
cargo test --doc --workspace --all-features -j 1
cargo metadata --locked --no-deps
git diff --check
```

The CI MSRV lane at Rust `1.95` and `cargo package --workspace --locked -j 1` remain release-level checks. No live provider or credentialed canary is required.

### Review Gates

- Run `compound-engineering:ce-simplify-code` after each settled behavioral unit, scoped to that unit's diff.
- Run `compound-engineering:ce-code-review` before the final integration commit, with special attention to replay certainty, durable successor ownership, raw authority, and release restartability.
- Treat any P0/P1 correctness or security finding as blocking. P2 cleanup may be deferred only when it does not contradict an R requirement or leave dead compatibility code.

---

## Definition of Done

- [ ] U1-U8 satisfy their cited requirements, flows, acceptance examples, and verification outcomes in dependency order.
- [ ] A failed or partial workspace publish cannot create a repository tag/GitHub Release, and non-main dispatch cannot begin publishing.
- [ ] Runtime preserves caller/provider-visible tools, executes only trusted local bindings, fingerprints complete annotated specs, and resumes with the same catalog/usage/successor semantics.
- [ ] MCP performs no hidden replay, applies raw bounds before parsing, sanitizes default errors, and does not permanently poison long-lived sessions by notification count.
- [ ] Responses WebSocket produces one submission-aware terminal under success, cancellation, timeout, actor failure, socket failure, and drop.
- [ ] DeepSeek, Vertex Anthropic, Cohere, and Gemini audited eligibility gates are removed or retained only with documented encoding/interpretation evidence and paired fixtures.
- [ ] OpenAI/Anthropic raw provider-body fields are forward-compatible without acquiring endpoint, credential, header, retry, timeout, or canonical request authority.
- [ ] Core/runtime/store/provider responsibilities match ADRs and architecture docs; stale ADR wording is corrected at the owning entry.
- [ ] Facade exports and user-facing docs match actual public APIs and release behavior; obsolete handoff/rehearsal files and bad-behavior tests are deleted.
- [ ] Per-unit, integration, docs/doctest, MSRV, packaging, metadata, formatting, Clippy, and diff gates pass serially.
- [ ] No credentials, live payloads, local absolute paths, generated release artifacts, or `repo-ref/` contents enter the diff.
- [ ] No abandoned experiment, compatibility shell, dead helper, duplicate policy list, or unrelated formatting change remains in the final diff.
