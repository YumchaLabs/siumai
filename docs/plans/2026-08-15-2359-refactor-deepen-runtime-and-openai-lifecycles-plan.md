---
title: Deep Runtime and OpenAI Lifecycles - Plan
type: refactor
date: 2026-08-15
deepened: 2026-08-15
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Deep Runtime and OpenAI Lifecycles - Plan

## Goal Capsule

| Field | Contract |
|---|---|
| Objective | Turn the runtime durable state, provider-deferred state, OpenAI Responses WebSocket lifecycle, and OpenAI HTTP/SSE execution paths into deep modules with one state owner each; remove public construction seams and duplicate execution plumbing that expose implementation detail. |
| Priority | P0: runtime state ownership and durable replay; P1: Responses WebSocket lifecycle ownership; P2: lossless OpenAI execution reuse and cleanup. |
| Compatibility posture | Breaking Rust API, snapshot schema, and durable ABI changes are allowed. Remove obsolete constructors, aliases, advanced test seams, dead wire fields, duplicate helpers, stale tests, and migration text instead of preserving compatibility shells. |
| Existing behavior to preserve | Snapshot v7 usage settlement, successor-before-CAS validation, model-visible tool versus local binding separation, annotation-complete tool identity, full replay-scope provider-deferred identity, and WebSocket submission certainty/exactly-once terminal behavior are correct baselines, not features to reimplement. |
| Architectural boundary | Runtime owns multi-step state transitions; snapshot owns durable representation and validation; `RunStore` owns lease fencing, run identity, revision, terminal-write rejection, and atomic CAS/replacement only; provider crates own provider semantics; `siumai-openai-compatible` may own shared OpenAI HTTP/SSE mechanics but never branded OpenAI resources, sessions, evidence, or native fidelity. |
| Evidence hierarchy | Current repository ADRs and executable contract tests are primary for internal ownership. Official provider documentation remains primary for wire semantics. `repo-ref/ai` is secondary prior art for concepts and fixtures only. |
| Execution posture | Implement serially in one shared worktree and keep Cargo commands serial. Scoped commits are optional after green gates, except U1-U4 activate as one durable compatibility-boundary commit so no intermediate commit changes durable semantics while still advertising snapshot v7 / ABI v6. Do not publish, push, or open a PR unless separately authorized. |
| Stop conditions | Stop only for contradictory current wire evidence, an unavoidable dependency cycle, a loss of official OpenAI native fidelity that cannot be represented by the bounded execution seam, or overlapping user changes that cannot be isolated safely. |

---

## Product Contract

### Summary

The previous post-beta hardening work fixed the important behaviors. The remaining problem is structural: correct invariants are still spread across shallow interfaces and multiple mutable representations.

- Durable tool transitions are manually assembled in `engine.rs`, `durable.rs`, and snapshot code.
- Provider-deferred state exists simultaneously in the run report, current stream state, pending snapshot, deserialization checks, and successor comparison.
- Snapshot types expose construction and mutation APIs that only the runtime should use, while validation remains private and retrospective.
- A single Responses WebSocket turn is represented by several partially overlapping state owners.
- Official OpenAI and the configured OpenAI-compatible engine duplicate request-plan, HTTP error, SSE framing, and stream settlement mechanics even though the repository already names the compatibility package as the configured execution engine.

This plan deepens those modules without reopening the provider-neutral family model, inventing a universal workflow framework, or copying the local AI SDK's TypeScript layout.

### Adopter Outcomes

- High-level `Agent`, `ToolLoop`, structured-run, and durable users keep the same supported journeys, but malformed multi-tool responses can no longer leave partially committed semantic state.
- External `RunStore` implementers retain the small lease/revision/CAS adapter contract and no longer need to understand or construct runtime-internal snapshot state graphs.
- Runtime operators gain one inspectable journal/ledger/checkpoint lineage while serialized snapshots are explicitly documented as sensitive, authority-bearing persistence data.
- OpenAI users keep typed options, native Responses carriers, resources, Realtime, and high-level Responses WebSocket APIs while duplicated stateless HTTP/SSE lifecycle code is removed beneath them.
- Direct `siumai-openai-compatible` provider authors gain one documented, versioned execution extension instead of copying transport, error, and SSE settlement loops.

### Problem Frame

The architecture currently has good policy boundaries but weak locality:

1. **One fact has several writers.** Tool lifecycle, provider-deferred observations, and WebSocket turn settlement can be mutated through multiple paths.
2. **Public APIs expose invalid construction space.** Callers can assemble durable state graphs whose correctness can only be checked later by private validation.
3. **Internal tests depend on construction detail.** Some tests prove behavior by manually appending events or injecting public low-level WebSocket adapters instead of exercising the supported high-level contract.
4. **Shared execution leverage stops too high.** `siumai-openai-compatible` owns a configured execution engine, but official OpenAI still duplicates stateless HTTP/SSE mechanics because the current seam erases native carrier fidelity.

The desired shape is one authoritative state owner with multiple read-only projections:

- one completed-step plan before mutation;
- one durable tool journal;
- one provider-deferred ledger;
- one snapshot checkpoint writer;
- one WebSocket turn lifecycle;
- one lossless OpenAI HTTP/SSE execution kernel.

### Requirements

#### Runtime planning and tool lifecycle

- R1. `Agent` remains a thin facade over `ToolLoop`; ordinary, structured, Agent, and durable paths share one completed-step classification and history-projection path rather than maintaining parallel orchestration.
- R2. A completed model response is fully classified before any semantic projection is committed: messages, steps, approvals, tool journal, provider-deferred ledger, and continuation state update atomically. Irreversible model-attempt, elapsed-time, and provider-reported usage accounting is settled exactly once even when classification later fails; semantic rollback must not erase already consumed provider work.
- R3. `LanguageRequest.tools` remains the model-visible catalog and `ToolSet` remains the trusted local execution binding map. Provider-owned tools, opaque replay items, and provider-deferred work never gain local execution authority.
- R4. A crate-private durable tool journal is the sole writer for prepared, approval-relevant, dispatched, completed, indeterminate, recovered, and retried execution transitions.
- R5. The journal owns event sequence, event time assignment, legal predecessor state, attempt progression, frozen request restoration, stable idempotency identity, recovery policy, and replay disposition.
- R6. The durable dispatch order remains: persist `Dispatched`, perform the side effect, then persist `Completed`. A lost result after dispatch becomes indeterminate unless stable idempotency and explicit recovery policy permit a new attempt.
- R7. Approval remains a host/human authority. Runtime may request, verify, consume, and record bounded approval identity, but it cannot manufacture a trusted approval or persist the original authority-bearing envelope as executable state.
- R8. Public journal inspection remains available through read-only events, status, attempt, and replay disposition. Public event constructors, append/recovery mutation, and `RunReport::execution_log_mut` are removed.

#### Provider-deferred state

- R9. A crate-private `ProviderDeferredLedger` is the only owner of provider-deferred identity, first-seen order, last-observation-wins replacement, resolution, pending projection, serialization validation, and successor progression. Observations remain call-local and staged until an authoritative provider terminal commits the completed step; a failed, cancelled, or unexpected-EOF stream never turns partial observations into resumable provider work.
- R10. A ledger key contains the complete validated `ProviderScope` plus a separate bounded correlation identifier. Namespace-only identity, model ID, Registry route, or configured instance identity must not replace replay-scope identity.
- R11. Same-key observations update in place without changing order; different keys retain first-seen order; deletion, reorder, scope substitution, duplicate durable keys, or reopening a resolved key fail closed. `PendingProviderStepSnapshot.provider_state` must equal the ledger's exact unresolved projection for that step, including metadata, encoding, payload provenance, and correlation identity.
- R12. Provider-deferred work and caller-owned local calls may coexist. Runtime records provider state and continues local work; it suspends in `AwaitingProvider` only when no caller-owned progress remains and unresolved provider work still exists.
- R13. `ProviderOpaque` remains replay metadata and does not by itself create provider-deferred work, suspension, approval, or local execution.
- R14. The report and public snapshot continue to expose read-only provider-deferred observations through explicit accessors; default Debug, Display, errors, and traces remain structural and redacted. Step-local vectors/maps and duplicated normalization helpers are deleted.

#### Snapshot ownership and persistence

- R15. Snapshot production is runtime-owned. `RunSnapshot`, resume state, pending state, execution journal, provider ledger projection, fingerprints, and checkpoint identity remain publicly inspectable and serializable, but internal state constructors and mutators become crate-private. Serialized snapshots are sensitive, authority-bearing replay state rather than sanitized diagnostics.
- R16. One crate-private checkpoint writer assembles initial snapshots, ordinary successors, approval successors, provider-suspension successors, recovery successors, and terminal successors.
- R17. Runtime checkpoint code validates `previous -> candidate`, enforces `RunBudget::max_snapshot_bytes`, and rejects over-limit candidates before calling `RunStore::compare_and_swap`. `RunStore` continues to own only lease fencing, run identity, revision, terminal-write rejection, and atomic replacement. External stores must bound serialized input before deserialization and provide confidentiality, integrity/authenticity, tenant/run isolation, access control, and rollback/revision protection; schema and successor validation are not snapshot authentication.
- R18. Snapshot schema moves once from v7 to v8 after the journal and ledger shapes are final. Durable execution ABI moves once from `siumai-runtime-durable-v6` to `siumai-runtime-durable-v7`.
- R19. Snapshot v8 removes the unused tool `dispatch_id` field and serializes the validated journal/ledger representation. Snapshot v7 is rejected at the version envelope before typed payload decoding; no migration shim or guessed reconstruction is added.
- R20. Public constructors and mutable fields that only runtime/tests use are deleted, including direct construction of snapshots, checkpoints, pending states, execution events/logs, and mutable fingerprint/provider-state structures. `RunStore` adapter constructors required by real external store implementations remain.
- R21. Invalid snapshot tests mutate bounded serialized wire data or exercise module-private builders. Integration tests produce valid snapshots through `DurableToolLoop`, not by hand-building internal state graphs.

#### Responses WebSocket lifecycle

- R22. One crate-private turn lifecycle owns queue, submission phase, submission certainty, response identity, terminal publication, fallback error, consumer EOF behavior, and actor-exit settlement.
- R23. `NotSubmitted`, `Indeterminate`, and `Settled` remain the public submission states. Only an authoritative provider terminal may mark a turn `Settled`; send/ack uncertainty remains `Indeterminate`.
- R24. Actor panic, abort, socket failure, unexpected EOF, protocol desynchronization, cancellation, deadline, session close, turn drop, and saturated queues produce at most one typed turn terminal followed by EOF. A well-formed provider failed/cancelled turn may settle without poisoning a reusable session; transport uncertainty and protocol loss close conservatively.
- R25. Generated and warm-up turns share lifecycle mechanics but not response projection. Warm-up remains native-only and must not be synthesized into a portable `LanguageResponse`.
- R26. The high-level experimental API remains `OpenAiResponsesWebSocketConfig`, session, turn, event, submission state, and turn kind. The public `advanced` connector/socket/sender/receiver surface and public `with_connector` injection are deleted.
- R27. Production WebSocket transport and a scripted deterministic test adapter remain private implementation adapters. Existing finite queue-capacity bounds, session/turn timeout ceilings, actor cleanup deadline, and `TransportLimits` frame/event bounds remain enforced before allocation or actor start. No provider-neutral session trait or generic actor framework is introduced.

#### OpenAI configured execution reuse

- R28. `siumai-openai-compatible` gains a narrow, versioned, stateless OpenAI HTTP/SSE execution kernel below `OpenAiCompatibleLanguageModel`. Its `extension::v2` module is a supported direct-crate provider-author API with explicit semver coverage and an external-consumer compile contract; it is not re-exported by the facade.
- R29. Provider code produces validated prepared-call inputs: target, non-credential headers, bounded body, replay safety, warnings, decoder context, and provider-owned direct/stream decoder. The kernel constructs the immutable transport `RequestPlan` and owns transport execution, bounded non-success capture, OpenAI error metadata classification, SSE byte framing, terminal-in-batch discipline, unexpected EOF handling, cancellation/deadline propagation, and contextualized setup failure.
- R30. The kernel consumes the already selected `ProviderTransport` and validated prepared-call inputs. It does not construct a second provider runtime, select credentials, own provider options, mint provider identity, infer branded support, or let decode hooks alter endpoint, authentication, signing, replay, retry, timeout, or transport policy. The facade `openai` feature may gain an internal dependency edge but must not expose or activate the facade `openai-compatible` API feature.
- R31. The kernel accepts provider-owned direct/stream decode hooks without erasing official native response resources, native stream frames, replay status, warnings, usage, or provider opaque items.
- R32. Existing `extension::v1` codec policies remain for current branded compatible providers. The new execution seam is a bounded `extension::v2` contract; it does not force unrelated providers to rewrite codec policies in the same commit.
- R33. Both `OpenAiCompatibleLanguageModel` and official OpenAI Chat/Responses portable direct/stream paths use the kernel. Official OpenAI native direct/stream paths reuse the same transport/framing kernel while retaining branded native carriers.
- R34. Official OpenAI retains typed option merge, exact configured-instance targeting, annotation resolution, native/function tool encoding, provider resources, background Responses, Realtime, Responses WebSocket, support evidence, and branded diagnostics.
- R35. Differential fixtures must prove each official and compatible route remains equivalent to its own pre-kernel request/header/replay and result baseline before duplicate code is deleted. Dated official OpenAI request, streaming, error, EOF, and usage fixtures arbitrate mismatches; cross-route byte equality is required only for an explicitly verified exact-OpenAI compatibility profile. Native carrier, tool ownership, replay metadata, diagnostics redaction, cancellation, and EOF behavior remain covered.
- R36. The migration must not wrap official OpenAI in `OpenAiCompatibleProvider`, claim custom endpoint fidelity, or reduce official Responses to the compatible portable surface.

#### Cleanup and documentation

- R37. Delete obsolete helper functions, duplicate state vectors/maps, public internal constructors, dead `dispatch_id`, duplicate execution loops, advanced WebSocket re-exports, bad-behavior tests, and stale rustdoc/migration examples in the same unit that makes them obsolete.
- R38. Update architecture documentation and durable ADRs to describe the journal, ledger, checkpoint writer, private WebSocket lifecycle, and execution kernel as current owners; update migration guidance for all public breaks and snapshot v8/ABI v7 rejection.
- R39. Do not add a capability matrix, generic provider workflow trait, generic provider polling trait, source parser, dependency-policy mirror, or provider-by-model-by-option test matrix.
- R40. Keep code comments, rustdoc, migration text, changelogs, and commit messages in English. Tests remain deterministic, offline, bounded, and secret-free.

### Product Key Decisions

- **All five deepening candidates are in scope.** Runtime snapshot, tool journal, deferred ledger, Responses WebSocket, and OpenAI execution reuse are one coordinated breaking refactor. *(session-settled: user-directed — chosen over handling only the highest-confidence candidate and leaving duplicated state owners.)*
- **Breaking cleanup is preferred to compatibility scaffolding.** Remove obsolete public construction APIs, advanced connector seams, dead wire fields, and stale aliases; document the break instead of leaving deprecated shells. *(session-settled: user-directed — chosen over preserving beta-era implementation seams.)*
- **Use one snapshot generation break.** Journal and ledger state are finalized before one schema v8 / ABI v7 transition; v7 snapshots fail closed and are recreated. *(session-settled: user-directed — chosen over multiple incremental snapshot generations or a guessing migration shim.)*
- **Implement autonomously under a goal and commit during execution.** The plan is authoritative, units run serially in dependency order, and the orchestrator may create scoped Conventional Commits after green gates. *(session-settled: user-directed — chosen over pausing for per-unit confirmation.)*
- **AI SDK is secondary prior art, not an architecture target.** Borrow its hidden workflow persistence and focused helper principles, but do not copy its TypeScript types, dynamic maps, capability booleans, or separate provider module layout mechanically. *(session-settled: user-directed — chosen over source-layout parity with `repo-ref/ai`.)*

### Key Flows

- F1. **Completed model step**
  - Trigger: a direct or streamed model call settles successfully.
  - Steps: completed-step planner validates the entire response; projects assistant history; classifies local calls, provider-deferred observations, opaque replay data, usage, and stop policy; only then commits journal/ledger/report changes.
  - Outcome: ordinary, Agent, structured, and durable paths observe the same step record and no partial mutation survives a planning failure.
  - Covered by: R1-R8, R12-R14.

- F2. **Durable local tool execution**
  - Trigger: a validated caller-owned tool call is locally bound.
  - Steps: journal prepares frozen work; approval is resolved; durable checkpoint records dispatch; executor runs; journal records completed or indeterminate; checkpoint writer projects the authoritative state.
  - Outcome: local authority, retry certainty, budget, approvals, report, and snapshot cannot diverge.
  - Covered by: R3-R8, R15-R21.

- F3. **Mixed local and provider-deferred work**
  - Trigger: one provider step emits both local function calls and provider-deferred observations.
  - Steps: ledger records provider state; planner prepares all local work; runtime executes local calls; resolved provider results advance ledger; suspension occurs only when unresolved provider work remains without local progress.
  - Outcome: provider-owned work never enters local execution and local work is not blocked merely because provider work also exists.
  - Covered by: R9-R14.

- F4. **Snapshot resume and CAS**
  - Trigger: a durable run starts, suspends, receives approval, recovers, or reaches terminal state.
  - Steps: checkpoint writer creates one valid candidate from journal, ledger, report, and resume state; successor validation runs; only then does the store perform CAS.
  - Outcome: external and built-in stores observe the same runtime contract and callers cannot manufacture semi-valid durable state through public constructors.
  - Covered by: R15-R21.

- F5. **Responses WebSocket turn**
  - Trigger: a generated or warm-up turn is queued.
  - Steps: lifecycle tracks queue/send/ack certainty, actor ownership, response identity, provider terminal, cancellation/deadline, fallback error, and consumer EOF from one state source.
  - Outcome: one terminal at most, accurate retry certainty, bounded actor cleanup, and no clean EOF before settlement.
  - Covered by: R22-R27.

- F6. **Official OpenAI stateless call**
  - Trigger: official or compatible Chat/Responses direct or SSE execution begins.
  - Steps: provider-owned options/codecs produce a request; shared kernel builds/executes the plan, captures bounded errors, frames SSE, and hands frames/results to the provider-owned decoder.
  - Outcome: shared mechanics are reused while official native responses, frames, replay status, resources, identity, and diagnostics remain intact.
  - Covered by: R28-R36.

### Acceptance Examples

- AE1. Covers F1. A response contains two valid local calls followed by one malformed local call. Planning fails before journal, messages, approvals, provider-deferred state, or step records change; the consumed model attempt and provider-reported usage are still settled exactly once.
- AE2. Covers F1/F2. The same scripted response through `Agent`, `ToolLoop`, and `DurableToolLoop` yields the same assistant history, tool order/results, usage, budget, and final terminal.
- AE3. Covers F2. A side-effecting call checkpoints `Dispatched`, executes, then loses the result checkpoint. Resume exposes indeterminate state and does not replay without stable idempotency and explicit retry policy.
- AE4. Covers R7. Stale, replayed, unknown-call, annotation-drifted, argument-drifted, or catalog-drifted approvals fail before dispatch; provider-owned items never reach the approval decider.
- AE5. Covers F3. `queued -> in_progress -> resolved` for one exact deferred key updates in place across model steps and resume; a second key keeps its original relative order.
- AE6. Covers F3. One step emits a provider-deferred item and a caller-owned local call. The local call executes; the run suspends only after local progress completes and the provider key remains unresolved.
- AE7. Covers R10-R13. Same correlation with different platform, API mode, protocol, or replay domain remains distinct; `ProviderOpaque` alone never suspends.
- AE7a. Covers R9. A stream emits provider-deferred observations and then fails, is cancelled, or reaches unexpected EOF before an authoritative terminal. The staged observations do not enter the resumable ledger, pending snapshot, or `AwaitingProvider`; bounded native/partial failure diagnostics may retain them only through explicit sensitive accessors.
- AE8. Covers F4. Snapshot v8 round-trips every resume kind and terminal; a v7 envelope is rejected before typed payload decode; a removed `dispatch_id` cannot reappear through unknown-field acceptance.
- AE9. Covers F4. An invalid recovery candidate is rejected before a fake external store records any CAS call; valid built-in and external store flows remain behaviorally identical.
- AE9a. Covers R15-R17. Exact-limit snapshots persist, over-limit candidates produce zero CAS calls on every checkpoint path, and an external store rejects oversized serialized input before typed deserialization. Snapshot Debug/errors remain redacted while explicit serialization retains replay bytes.
- AE10. Covers R15-R21. A facade/runtime consumer can inspect lineage, resume kind, approvals, journal, deferred observations, usage, budget, and terminal but cannot name a public constructor that creates or mutates internal snapshot state.
- AE11. Covers F5. Cancellation before queue reports `NotSubmitted`; timeout after sender acceptance reports `Indeterminate`; an authoritative provider terminal reports `Settled`.
- AE12. Covers F5. Actor abort, panic, socket EOF, protocol failure, turn drop, and queue saturation each publish one typed failure at most and then EOF; warm-up remains native-only.
- AE12a. Covers F5. A provider-proven terminal that cannot enter a saturated consumer queue still reports `Settled` through the fallback path; a nonterminal saturation reports `Indeterminate` and closes the session.
- AE13. Covers R26. Facade and provider compile contracts no longer expose `experimental::responses_websocket::advanced`, while high-level config/session/turn code still compiles.
- AE14. Covers F6. Every official and compatible fixture remains byte-equivalent to its own pre-kernel request/header/replay baseline and preserves its own direct/stream result contract. Cross-route byte equality is asserted only for a profile explicitly verified as exact OpenAI compatibility, and dated official fixtures decide whether an observed difference is a preserved bug or a legitimate dialect distinction.
- AE15. Covers F6. Official native direct and native stream APIs retain one-request native+portable pairing, failed resource retention, native frames, replay status, usage-only terminal behavior, and bounded diagnostics.
- AE16. Covers R34-R36. Background Responses, resources, Realtime, WebSocket, support manifests, exact instance targeting, raw-option authority, and official/custom fidelity claims remain owned by `siumai-provider-openai`.
- AE17. Covers R37-R40. Repository search finds no removed constructors, `dispatch_id`, advanced Responses WebSocket connector exports, duplicated stream pump, or stale v7/v6 migration claim except explicitly historical text.

### Success Criteria

- Runtime has one authoritative completed-step planner, tool journal, provider-deferred ledger, and checkpoint writer; malformed completed responses cannot partially commit semantic state or erase consumed call accounting.
- Public snapshot APIs are inspection/store surfaces rather than state-machine construction surfaces.
- Snapshot v8 and durable ABI v7 are the only new durable generation introduced by this plan.
- Responses WebSocket turn settlement has one state owner and no public low-level connector seam.
- Official OpenAI and the compatible engine share stateless HTTP/SSE mechanics without losing any official native behavior.
- High-level runtime/OpenAI journeys and external `RunStore` adapters remain usable through supported public APIs, while removed construction/advanced seams have explicit migration guidance.
- Direct compatible-provider authors can implement the documented `extension::v2` contract from outside the workspace.
- The refactor deletes more duplicated mutation/execution code than it adds in adapters and public API.
- All affected deterministic contract suites, feature checks, Clippy, formatting, metadata, workspace tests, and packaging checks are green.

### Scope Boundaries

#### Included

- Runtime completed-step planning, tool journal, provider-deferred ledger, checkpoint writer, snapshot public-surface cleanup, schema v8, and ABI v7.
- Responses WebSocket private module split and public advanced-seam removal.
- Lossless configured OpenAI HTTP/SSE execution kernel and official OpenAI migration.
- Public rustdoc, facade exports, changelogs, architecture, migration guidance, contract tests, and deletion cleanup directly affected by the breaks.

#### Deferred

- Cross-process persistence/resume of provider-owned WebSocket or Realtime sessions.
- Generic remote provider polling or a provider-neutral deferred-work API.
- New provider/model capability work, live provider tests, or support-evidence refresh unrelated to changed contracts.
- Refactoring MCP transport, support manifests, media planners, Registry, or option targeting solely because files are large.
- A generic cross-provider execution kernel or session actor framework.

### Dependencies and Sources

- `AGENTS.md`
- `docs/architecture/overview.md`
- `docs/adr/0011-protocol-projection-ownership.md`
- `docs/adr/0014-canonical-language-history-and-replay.md`
- `docs/adr/0015-validation-ownership-and-forward-compatibility.md`
- `docs/migration/siumai-next.md`
- `docs/plans/2026-08-14-0903-refactor-post-beta-contract-and-lifecycle-hardening-plan.md`
- Current baseline commits `e2944cea`, `9812b128`, `88eaf8f4`, and `dcf87936`.
- Local `repo-ref/ai` commit `3bc0d4f40df7` (2026-08-01), used only for workflow/tool-state concepts and fixture ideas.
- OpenAI Chat Completions API reference, verified 2026-08-15: `https://developers.openai.com/api/reference/resources/chat`
- OpenAI Responses API reference, verified 2026-08-15: `https://developers.openai.com/api/reference/resources/responses`
- OpenAI streaming Responses guide, verified 2026-08-15: `https://developers.openai.com/api/docs/guides/streaming-responses`
- OpenAI API error guide, verified 2026-08-15: `https://developers.openai.com/api/docs/guides/error-codes`

---

## Planning Contract

### Key Technical Decisions

- KTD1. **Plan semantic state before committing it, but settle consumed calls.** Introduce a crate-private completed-step planner that classifies history projection, local calls, provider-deferred observations, usage, and terminal/continuation outcome in one pass. Engine and durable paths consume the same plan. Semantic projections commit atomically; irreversible model-attempt and provider-usage accounting settles exactly once even when planning rejects the response. Governs R1-R3, R12-R14.
- KTD2. **Use a semantic tool journal, not public raw event append.** Move execution state ownership from snapshot/event constructors into `siumai-runtime/src/tool/journal.rs` (or an equivalent private module) with `prepare`, `dispatch`, `complete`, `mark_indeterminate`, `recover`, and query operations. Governs R4-R8.
- KTD3. **Keep journal and ledger separate.** Tool journal owns caller-executed local effects; provider-deferred ledger owns provider-executed pending state. They may be committed in one completed-step transaction but must not share a generic workflow state enum. Governs R3, R9-R14.
- KTD4. **Activate runtime state and the durable break atomically.** Develop and verify U1-U3 before finalizing the wire, but do not land or expose changed durable execution semantics under snapshot v7 / ABI v6. U1-U4 activate together in one compatibility-boundary commit that performs schema v8/ABI v7, deletes `dispatch_id`, and privatizes construction. This avoids intermediate v7 snapshots with new meaning and multiple incompatible generations. Governs R15-R21. *(session-settled: user-directed breaking cleanup — chosen over compatibility shells or staged schema migrations.)*
- KTD5. **Snapshot is a deep module; store remains a replaceable port.** Keep `RunSnapshot`, read-only views, serde, `RunStore`, `StoredRun`, and lease token adapters public. Make state creation/recovery/successor mutation private and preserve successor-before-CAS ownership in runtime. Governs R15-R21.
- KTD6. **Use one actor-owned WebSocket turn lifecycle.** `TurnLifecycle` owns submission tracker, fallback error, active identity, terminal, and consumer close semantics. Actor/session/turn handles hold references or projections, not independent settlement flags. Governs R22-R27.
- KTD7. **Delete the public Responses WebSocket transport seam.** There is one production adapter and one test adapter, but no real application adapter. Keep both private; remove `advanced` exports and public injection rather than turning test machinery into product API. Governs R26-R27. *(session-settled: user-authorized public break — chosen over preserving an unproven extension surface.)*
- KTD8. **Expose one bounded provider-author execution contract.** Add a documented `extension::v2` direct-crate API around stateless prepared-call/transport/error/SSE mechanics. A prepared call supplies validated target, non-credential headers, bounded body, replay safety, warnings, and decoder context. Direct decoders map one successful buffered response to an associated output; stream decoders map framed SSE data to an associated event type, identify terminal events, and finish explicitly. Setup fails through the outer result; established streams emit `Result<Event, Error>`, settle once, and cancel their child operation on drop. Existing `extension::v1` remains supported. Governs R28-R36.
- KTD9. **Do not wrap official OpenAI in the generic provider.** Migrate official portable and native stateless calls onto the kernel while keeping official construction, typed options, annotations, resources, background work, Realtime, WebSocket, evidence, and diagnostics in the branded crate. Governs R30-R36.
- KTD10. **Use route-local characterization plus an official oracle before deletion.** Preserve each old route in fixtures, verify current official request/error/stream/EOF/usage semantics against the dated sources, switch one route at a time, and then delete duplicate loops. If the kernel cannot express a current official fixture without branded branching inside the kernel, keep that semantic in the provider hook rather than widening the kernel. Governs R31-R36.
- KTD10a. **Keep the kernel mode-agnostic and output-preserving.** Provider code prepares validated call inputs and owns decoding; the kernel alone constructs and executes the transport `RequestPlan`. It must not branch on branded `OpenAiApiMode` or collapse output to `LanguageStream` when a native frame carrier is required. Governs R28-R36.
- KTD11. **Do not refactor by file size.** MCP transports, `EngineCheckpointPort`, support manifests, and provider-specific media planners already have justified owners or real adapters. They remain out of scope. Governs R39.
- KTD12. **Borrow AI SDK principles, not structure.** Its hidden workflow persistence supports the journal/snapshot direction, but its separate OpenAI implementations and lack of a Responses WebSocket lifecycle are counterevidence against copying its module graph. Governs R28-R40.

### High-Level Technical Design

The diagrams describe ownership and flow, not exact type signatures.

```text
Model terminal
    |
    v
CompletedStepPlanner  -- validates whole step before mutation
    | local work                   | provider work
    v                              v
ToolLifecycleJournal         ProviderDeferredLedger
    |                              |
    +-------------+----------------+
                  v
            CheckpointWriter
                  |
                  v
             RunSnapshot v8
                  |
       successor validation
                  |
                  v
               RunStore CAS
```

```text
Public WS session/turn API
          |
          v
     TurnLifecycle  <----- Actor lifecycle / socket frames
     - submission certainty
     - response identity
     - exactly-once terminal
     - fallback/EOF semantics
          |
          v
  generated events OR warm-up native frames
```

```text
Provider-owned options + codec + identity
                    |
                    v
       OpenAI Execution Kernel (compatible crate)
       - RequestPlan mechanics
       - bounded HTTP errors
       - SSE framing/settlement
                    |
                    v
          existing ProviderTransport
                    |
                    v
 Provider-owned portable/native decoder and carrier
```

### System-Wide Impact

| Area | Impact |
|---|---|
| Runtime public API | Removes public snapshot/event construction and mutation; retains read-only durable inspection and store adapters. |
| Snapshot compatibility | Schema v8 and ABI v7 reject v7/v6-era runtime state. Users recreate beta snapshots. |
| Agent/tool behavior | Intended behavior remains the same, with stronger atomic planning and one local execution state owner. |
| Provider-deferred behavior | Intended replay-scope identity/order semantics remain; internal duplicate representations are removed and mixed local/deferred progression becomes explicit. |
| OpenAI experimental API | Removes Responses WebSocket `advanced` connector/socket exports and `with_connector`; high-level session/turn APIs remain. |
| Workspace dependencies | `siumai-provider-openai` adds a dependency on `siumai-openai-compatible`; no reverse edge or branded-provider dependency is introduced. |
| Compatible extension API | Adds a documented provider-author `extension::v2` with semver coverage and an external compile contract; existing `v1` remains for current branded consumers. The facade does not expose `v2`. |
| Documentation | Architecture and migration docs must describe the new owners and all public breaks. |

### Public Surface Keep/Drop Contract

| Surface | Decision |
|---|---|
| `RunSnapshot`, `RunStore`, `StoredRun`, `RunLease`, `SnapshotRevision`, durable IDs, and read-only accessors | Keep public. These are the persistence and operational inspection contract. |
| `StoredRun::new`, store-token lease adapters, and revision constructors required by external stores | Keep public and validated. They are real adapter seams. |
| Resume kind, pending approval/provider views, journal events/status, deferred observations, usage, budget, lineage, and terminal summary | Keep public as read-only inspection. |
| `RunSnapshot::new`, checkpoint/pending-state constructors, recovery mutation, event constructors, raw log append/recover, and mutable fingerprint/provider-state fields | Make crate-private or delete. They expose runtime state-machine construction. |
| `RunSnapshotSuccessorError`, `RunSnapshotError`, and `ToolExecutionTransitionError` used in durable public error source chains | Keep matchable and sanitized even when mutation APIs become private. |
| Responses WebSocket config/session/turn/event/submission state/kind | Keep in the experimental provider/facade surface. |
| Responses WebSocket connect request, connector, socket, sender, receiver, transport connector, `with_connector`, and `advanced` modules | Delete from public API; retain only private production/test adapters. |
| Compatible `extension::v1` codec policies | Keep for existing branded consumers. |
| New compatible stateless execution kernel | Add as a bounded, versioned direct-crate provider-author `extension::v2`; document its audience and stability, compile it from an external-consumer test, and do not re-export it from the facade. |

### Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Planner/journal/report become competing sources of truth | Planner is ephemeral; journal/ledger are authoritative durable state; report/snapshot are read-only projections produced through one commit path. |
| Journal extraction changes exactly-once effect boundaries | Preserve and extend crash-matrix tests around dispatch checkpoint, side effect, result checkpoint, retry, and indeterminate recovery before moving code. |
| Ledger resolves the wrong provider state | Key by complete `ProviderScope` plus correlation; test API mode/platform/protocol/replay-domain negatives and monotonic successor rules. |
| Snapshot cleanup removes needed operational visibility | Keep public getters for lineage, resume kind, pending approvals, journal, deferred observations, usage, budget, and terminal; delete only construction/mutation. |
| Serialized snapshot is mistaken for sanitized diagnostics | Document snapshots as sensitive authority-bearing replay state; require external stores to enforce confidentiality, integrity, isolation, access control, rollback protection, and pre-deserialization bounds. Keep only Debug/Display/errors/traces redacted. |
| WebSocket module split introduces race regressions | Characterize queue/send/ack/terminal races first; one lifecycle state owns terminal and fallback; retain deterministic scripted adapter tests privately. |
| Shared OpenAI kernel erases native fidelity or preserves a latent bug | Kernel transports bytes/frames and delegates decode; each route compares with its own baseline, then dated official OpenAI fixtures arbitrate request/error/stream/EOF/usage semantics before deletion. |
| New dependency creates a cycle or feature leak | Inspect `cargo metadata`; compatible depends only on core/protocol/transport, so official provider may depend downward without a reverse edge. Keep features additive and no-default checks green. |
| Large refactor makes failures hard to localize | Implement U1-U7 in the preferred serial order and run narrow gates after each unit. Land U1-U4 as one durable compatibility boundary; later independent units may use scoped commits after green gates. |

### Open Questions

No product blockers remain. Exact private type names and file splits are implementation-owned, provided they preserve the ownership, deletion, and verification contracts above. Any fixture showing that an official OpenAI semantic cannot pass through the bounded kernel is handled by a provider-owned decode/plan hook, not by widening scope or abandoning the kernel.

---

## Implementation Units

### U1. Add one completed-step planner and parity contracts

**Requirements:** R1-R3, R12-R14

**Dependencies:** None
**Primary paths:**

- `siumai-runtime/src/engine.rs`
- new private module under `siumai-runtime/src/engine/` or `siumai-runtime/src/run/`
- `siumai-runtime/src/agent.rs`
- `siumai-runtime/src/tool_loop.rs`
- `siumai-runtime/src/structured_run.rs`
- `siumai-runtime/tests/tool_loop_contract.rs`
- `siumai-runtime/tests/structured_run_contract.rs`
- `siumai-runtime/tests/durable_tool_loop_contract.rs`

**Approach:**

- Add characterization tests proving Agent/ToolLoop/Durable parity before moving mutation code.
- Extract completed-response classification into a crate-private planner that returns a fully validated immutable plan.
- Classify assistant history projection, local executable calls, provider-deferred observations, provider opaque content, tool results, usage settlement, stop policy, and next-phase decision before applying changes.
- Separate reversible semantic state from irreversible call accounting: semantic state commits only after full validation, while model-attempt, elapsed-time, and provider-reported usage settlement occurs exactly once even when the response is rejected.
- Preserve visible-tool/local-binding separation and hosted program caller relations.
- Apply the plan through one engine path; keep Agent a thin wrapper and structured output on the same history projection contract.
- Delete duplicated completed-step branching and tests that assert internal mutation order instead of public behavior.

**Test scenarios:**

- Same scripted response through Agent, ToolLoop, structured run repair, and DurableToolLoop produces equivalent history/steps/usage/budget.
- Multiple local tools followed by one preparation failure causes zero semantic report/journal/approval/message mutation while consumed attempt and provider usage accounting remains settled exactly once.
- Provider-owned hosted/MCP/opaque items never reach `ToolSet` or approval; caller-owned local function calls with provider caller metadata still execute.
- Visible unbound tools stay visible across subsequent steps but are never executable.
- Cancellation, model timeout, usage-only terminal, and history omission keep current terminal semantics.

**Gate:**

- `cargo nextest run -p siumai-runtime --test tool_loop_contract --test structured_run_contract --test durable_tool_loop_contract --test-threads 1`
- `cargo clippy -p siumai-runtime --all-targets --all-features -j 1 -- -D warnings`

### U2. Make the durable tool journal the only execution-state writer

**Requirements:** R4-R8

**Dependencies:** U1
**Primary paths:**

- new `siumai-runtime/src/tool/journal.rs`
- `siumai-runtime/src/tool/mod.rs`
- `siumai-runtime/src/engine.rs`
- `siumai-runtime/src/durable.rs`
- `siumai-runtime/src/run.rs`
- `siumai-runtime/src/snapshot/model.rs` (temporary wire/read model until U4)
- `siumai-runtime/tests/tool_contract.rs`
- `siumai-runtime/tests/tool_loop_contract.rs`
- `siumai-runtime/tests/durable_tool_loop_contract.rs`

**Approach:**

- Move tool execution transition logic out of snapshot into a crate-private journal owner.
- Give the journal semantic mutation methods for prepare, approval transition, dispatch, complete, indeterminate, recovery, and stable retry.
- Move frozen request restoration and attempt/policy/idempotency checks into the journal or its private collaborator; delete duplicated engine/durable restore implementations.
- Make event sequence/time and legal predecessor state internal details.
- Replace `RunReport::execution_log_mut` and direct event append with journal transactions/projections.
- Keep read-only event/status/attempt/replay inspection; move public inspection types to the logical `tool` surface if needed, with migration guidance.
- Keep snapshot v7 / ABI v6 serialization and resume meaning unchanged while U2 is under development. Do not land U2 independently; its activation is part of the U1-U4 compatibility-boundary commit.

**Test scenarios:**

- Table tests cover every legal/illegal state transition and attempt boundary.
- Crash matrix covers pre-dispatch checkpoint, dispatched checkpoint before effect, effect before result checkpoint, completed checkpoint, and parallel active subset.
- Stable retry reuses the idempotency key and increments attempt; non-stable or side-effecting indeterminate calls halt.
- Completed work is never replayed; direct success before dispatch remains invalid.
- Stale/replayed approvals and frozen binding/catalog/argument drift fail before dispatch.

**Gate:**

- `cargo nextest run -p siumai-runtime --test tool_contract --test tool_loop_contract --test durable_tool_loop_contract --test-threads 1`
- `cargo clippy -p siumai-runtime --all-targets --all-features -j 1 -- -D warnings`

### U3. Consolidate provider-deferred state into one ledger

**Requirements:** R9-R14

**Dependencies:** U1, U2
**Primary paths:**

- new `siumai-runtime/src/provider_deferred.rs`
- `siumai-runtime/src/run.rs`
- `siumai-runtime/src/engine.rs`
- `siumai-runtime/src/snapshot/model.rs`
- `siumai-runtime/src/snapshot/model/successor.rs`
- `siumai-server/src/event.rs`
- `siumai-runtime/tests/durable_tool_loop_contract.rs`
- `siumai-runtime/tests/tool_loop_contract.rs`

**Approach:**

- Move `ProviderDeferredKey`, observation validation, ordered upsert, resolution, pending projection, uniqueness, and successor progression into `ProviderDeferredLedger`.
- Store the ledger privately in `RunReport`; keep the public accessor as a sanitized observation slice/view.
- Replace `StepStream.provider_states`, `provider_state_positions`, and result-ID filtering with a ledger cursor/step projection.
- Make exact-scope resolution explicit and monotonic; keep correlation separate from codec namespace.
- Implement mixed local/deferred progression from the completed-step plan.
- Stage call-local observations until an authoritative provider terminal. On failed, cancelled, or unexpected-EOF streams, discard them from resumable ledger/pending state and expose them only through existing bounded sensitive native/partial diagnostics where available.
- Delete normalization, duplicate-key, and vector-prefix helpers that reimplement ledger logic.
- Keep snapshot v7 / ABI v6 serialization and resume meaning unchanged while U3 is under development. Do not land U3 independently; its activation is part of the U1-U4 compatibility-boundary commit.

**Test scenarios:**

- Same key updates in place across model steps and resume; distinct keys preserve first-seen order.
- Same correlation under different platform/protocol/API mode/replay domain stays distinct.
- Duplicate serialized keys, invalid correlation, deletion, reorder, scope substitution, or resolved-key reopening fails.
- Pending provider state is exactly the ledger's unresolved step projection; report/pending metadata, encoding, payload provenance, and correlation mismatch fails.
- Mixed deferred plus local call executes local work before suspension; pure unresolved deferred state suspends.
- Deferred observations emitted before failed/cancelled/unexpected-EOF settlement never enter the resumable ledger or pending snapshot.
- `ProviderOpaque` without deferred event never suspends.
- Server projection remains count/structure-only and redacted.

**Gate:**

- `cargo nextest run -p siumai-runtime --test tool_loop_contract --test durable_tool_loop_contract --test-threads 1`
- `cargo nextest run -p siumai-server --all-features --test-threads 1`
- `cargo clippy -p siumai-runtime -p siumai-server --all-targets --all-features -j 1 -- -D warnings`

### U4. Deepen snapshot construction and perform the single durable break

**Requirements:** R15-R21

**Dependencies:** U2, U3
**Primary paths:**

- `siumai-runtime/src/snapshot/mod.rs`
- replace/split `siumai-runtime/src/snapshot/model.rs` by ownership
- `siumai-runtime/src/snapshot/checkpoint.rs`
- `siumai-runtime/src/snapshot/model/successor.rs`
- `siumai-runtime/src/snapshot/model/wire.rs`
- `siumai-runtime/src/snapshot/store.rs`
- `siumai-runtime/src/durable.rs`
- `siumai-runtime/src/engine/checkpoint.rs`
- `siumai-runtime/src/lib.rs`
- `siumai-runtime/tests/durable_tool_loop_contract.rs`
- `docs/architecture/overview.md`
- new `docs/adr/0017-runtime-journal-ledger-and-snapshot-ownership.md`
- `docs/migration/siumai-next.md`
- `siumai-runtime/CHANGELOG.md`

**Approach:**

- Deepen the existing crate-private checkpoint assembly module into the single writer used by initial, ordinary, approval, provider-suspension, recovery, and terminal candidates; do not create a parallel writer abstraction.
- Split snapshot code by stable ownership such as identity/read model, wire/version, checkpoint assembly, successor validation, and store port; remove the old catch-all file if empty.
- Privatize internal constructors, fields, recovery mutation, and event/log mutation while retaining public read-only inspection and store adapter seams.
- Remove dead `dispatch_id`; encode finalized journal and ledger representation.
- Bump schema to v8 and durable ABI to v7 once; reject v7 at the envelope and do not add a migration shim.
- Enforce `RunBudget::max_snapshot_bytes` on every candidate before CAS, with exact-limit and over-limit coverage for initial, approval, provider-suspension, recovery, and terminal paths. External stores reject oversized serialized input before typed deserialization.
- Rewrite integration tests to obtain snapshots through runtime. Keep a small set of module-private builders for invariant tests; mutate JSON for malformed wire cases.
- Land U1-U4 together as the single durable compatibility-boundary commit; no released/intermediate commit may write changed semantics under snapshot v7 / ABI v6.
- Update migration, architecture, rustdoc, and runtime changelog in the same unit.

**Test scenarios:**

- v8 round-trip matrix covers every resume kind and completed/stopped/suspended/failed/cancelled/indeterminate terminal.
- v7 and future versions fail at the envelope before typed decode; missing required v8 fields and reintroduced `dispatch_id` fail.
- Built-in and external stores share the same CAS contract; invalid ordinary and recovery successors cause zero store CAS calls.
- Exact-limit candidates persist; every over-limit path produces zero store CAS calls; external input bounds reject before typed decode.
- Positive public compile contracts prove read-only inspection and external store adapters still work. Exact deletion searches verify removed constructors/mutators are absent.
- Usage, budget, tool journal, deferred ledger, approvals, lineage, and terminal remain inspectable and redacted.
- Snapshot Debug/errors redact sensitive contents; explicit serialization round-trips authority-bearing replay data. Store-contract tests document confidentiality/integrity/isolation responsibilities and reject rollback or oversized input before typed decode.

**Gate:**

- `cargo nextest run -p siumai-runtime --all-features --test-threads 1`
- `cargo nextest run -p siumai --no-default-features --features runtime --test facade_contract --test-threads 1`
- `cargo clippy -p siumai-runtime -p siumai --all-targets --all-features -j 1 -- -D warnings`

### U5. Give Responses WebSocket one private turn lifecycle

**Requirements:** R22-R27

**Dependencies:** None

**Preferred serial position:** After U4, so the runtime durable compatibility boundary is completed before editing the OpenAI provider crate.
**Primary paths:**

- replace `siumai-provider-openai/src/configured/responses_websocket.rs` with private ownership modules such as `responses_websocket/{mod,actor,lifecycle,turn,transport}.rs`
- `siumai-provider-openai/src/configured/mod.rs`
- `siumai/src/providers/openai.rs`
- `siumai/tests/facade_contract.rs`
- `siumai-provider-openai/CHANGELOG.md`
- `docs/migration/siumai-next.md`

**Approach:**

- Freeze current race/settlement fixtures before moving state.
- Replace `SubmissionTracker`, `TurnShared`, `ActiveTurn`, actor active-turn copy, and turn-handle `settled` flag with one `TurnLifecycle`/settlement owner and read-only projections.
- Make queue/send/ack control, fallback error, response identity, terminal publication, actor exit, and consumer EOF use that owner.
- Keep production transport and scripted test adapter private.
- Preserve the existing finite queue-capacity range, session/turn timeout ceilings, actor cleanup deadline, and `TransportLimits` frame/event bounds. Reject zero/oversized capacities and invalid timeouts before allocating channels or starting the actor.
- Remove public connector/socket/sender/receiver types, `with_connector`, provider `advanced` exports, and facade `advanced` re-exports.
- Preserve high-level API and warm-up native semantics.

**Test scenarios:**

- Queue/send/ack/terminal phase matrix proves `NotSubmitted`, `Indeterminate`, and `Settled` exactly.
- Queue saturation, caller cancellation, session cancellation, deadline, actor abort/panic, socket EOF, protocol error, close, and turn drop each settle once then EOF.
- Terminal response identity must match the turn; event after terminal fails.
- A well-formed provider failed/cancelled turn settles and leaves the session reusable; transport/protocol uncertainty closes it.
- Provider-proven terminal under a saturated event queue remains `Settled`; nonterminal saturation remains `Indeterminate`.
- Generated and warm-up turns share lifecycle but expose distinct event projection.
- Oversized inbound frames/events take the canonical typed failure path, settle once, close conservatively, and release actor resources within the cleanup deadline.
- Positive facade compile tests prove the high-level API remains. Exact deletion searches verify the advanced seam and connector symbols are absent.

**Gate:**

- `cargo nextest run -p siumai-provider-openai --all-features --test-threads 1`
- `cargo nextest run -p siumai --no-default-features --features openai,openai-responses-websocket --test facade_contract --test-threads 1`
- `cargo clippy -p siumai-provider-openai -p siumai --all-targets --all-features -j 1 -- -D warnings`

### U6. Extract a lossless OpenAI HTTP/SSE execution kernel

**Requirements:** R28-R32, R35

**Dependencies:** None

**Preferred serial position:** After U5, because both later units touch OpenAI configuration/module roots even though the stateless kernel has no technical dependency on the WebSocket lifecycle.
**Primary paths:**

- new `siumai-openai-compatible/src/configured/execution.rs`
- `siumai-openai-compatible/src/configured/model.rs`
- `siumai-openai-compatible/src/configured/mod.rs`
- `siumai-openai-compatible/src/lib.rs`
- `siumai-openai-compatible/src/configured/codec_policy.rs`
- `siumai-openai-compatible/Cargo.toml`
- compatible engine direct/stream contract tests

**Approach:**

- Add documented provider-author `extension::v2` for the stateless execution kernel while preserving `v1` codec policies for existing branded consumers. Add an external-consumer compile contract and rustdoc that states audience, semver stability, setup/stream error semantics, cancellation/drop behavior, and protected authority boundaries.
- Define prepared-call inputs as validated target, non-credential headers, bounded body, replay safety, warnings, and decoder context. The kernel constructs `RequestPlan`; provider hooks cannot mutate endpoint, credentials, signing, retry, timeout, or transport policy.
- Define a direct decoder with an associated output over successful bounded response headers/body, and a stream decoder with an associated event type, framed-data decode, explicit terminal predicate, response diagnostics, and `finish`. The kernel's established stream emits `Result<Event, Error>`, settles once, treats unexpected EOF as failure, and cancels its child operation on drop.
- Extract bounded direct error capture, OpenAI error classification, transport execute/execute-stream, SSE framing, terminal-in-batch, and unexpected EOF handling from the compatible model.
- Accept the existing transport instance, provider error context, warnings, and provider-owned direct/stream decoders.
- Keep options, identity, credentials, endpoint, replay domain, and branded semantics outside the kernel.
- Migrate `OpenAiCompatibleLanguageModel` onto the kernel and delete its duplicate execution loops.
- Add characterization fixtures for direct failure, incomplete EOF, usage-only terminal, duplicate terminal/event-after-terminal, cancellation/deadline, redaction, and raw-option authority.

**Test scenarios:**

- Chat and Responses direct/stream preserve request target, headers, replay safety, warnings, usage, terminal, and typed errors.
- Non-success bodies are bounded and sensitive; diagnostics expose only structural metadata.
- SSE chunk fragmentation, malformed UTF-8, usage-only terminal, terminal plus `[DONE]`, terminal-in-batch discipline, event-after-terminal, cancellation, and unexpected EOF remain strict.
- Provider-owned codec hooks can retain additional native data without granting endpoint/auth/transport authority.
- An external provider-author fixture implements `extension::v2` for both portable events and a custom native event carrier, proving the contract is usable without privileged workspace access.
- Existing `extension::v1` branded provider compile contracts remain green.

**Gate:**

- `cargo nextest run -p siumai-openai-compatible --all-features --test-threads 1`
- `cargo clippy -p siumai-openai-compatible --all-targets --all-features -j 1 -- -D warnings`
- focused nextest for Alibaba, DeepSeek, Groq, MiniMax, Moonshot AI, Volcengine, and xAI language contracts that consume `extension::v1`.

### U7. Migrate official OpenAI, delete duplicate plumbing, and close documentation

**Requirements:** R33-R40

**Dependencies:** U6
**Primary paths:**

- `siumai-provider-openai/Cargo.toml`
- root `Cargo.toml` only if the shared-dependency convention requires the new edge
- `siumai-provider-openai/src/configured/model.rs`
- `siumai-provider-openai/src/configured/provider.rs`
- `siumai-provider-openai/src/configured/http_error.rs`
- `siumai-provider-openai/src/configured/responses_native.rs`
- `siumai-provider-openai/src/configured/responses_resource.rs` only where shared helpers change, without moving resource ownership
- `siumai-provider-openai/src/configured/mod.rs`
- `siumai-provider-openai/src/lib.rs`
- `siumai-provider-openai/CHANGELOG.md`
- `siumai-openai-compatible/CHANGELOG.md`
- `siumai/tests/facade_contract.rs`
- `docs/architecture/overview.md`
- new `docs/adr/0018-openai-configured-execution-kernel.md`
- `docs/migration/siumai-next.md`

**Approach:**

- Add the downward dependency from official provider to the compatible execution engine and verify there is no cycle/feature leak.
- Migrate Chat direct/stream, Responses direct/portable stream, and official native direct/stream transport/framing onto the kernel one route at a time.
- Keep request preparation, option merge, annotations, native/function tools, response/native frame decoding, replay status, background/native resources, Realtime, WebSocket, support evidence, and branded error context in the official provider.
- Run route-local differential fixtures and dated official OpenAI oracle fixtures after each route. Delete the old route's request-plan/HTTP/SSE pump only after both continuity and current wire correctness are proven.
- Consolidate remaining resource error helpers only where ownership is truly shared; do not force resource APIs through the language kernel.
- Remove stale tests, docs, aliases, and imports; update architecture and migration guidance; run repository-wide deletion searches.

**Test scenarios:**

- Official request JSON, headers, replay safety, raw/typed option order, exact instance targeting, and annotation projection are unchanged.
- Native generation returns native and portable views from one request; failed native resources remain inspectable with portable typed error.
- Native stream retains raw/native frames, replay status, provider opaque items, usage-only terminal, exactly-once terminal, and no second request.
- Background Responses, resources, Realtime, and Responses WebSocket compile and retain provider-owned behavior.
- Custom endpoint remains explicit compatible configuration and never inherits official support evidence.
- Official request/error/stream/EOF/usage fixtures match the 2026-08-15 sources; exact-compatible profiles additionally prove cross-route equality, while other dialects retain their documented differences.
- Sensitive response bodies, IDs, credentials, and options remain redacted in Debug/Display/source chains.

**Gate:**

- `cargo nextest run -p siumai-provider-openai --all-features --test-threads 1`
- `cargo nextest run -p siumai-openai-compatible --all-features --test-threads 1`
- `cargo nextest run -p siumai --no-default-features --features openai,openai-compatible,openai-realtime,openai-responses-websocket --test facade_contract --test-threads 1`
- `cargo clippy -p siumai-provider-openai -p siumai-openai-compatible -p siumai --all-targets --all-features -j 1 -- -D warnings`

---

## Verification Contract

### Per-Unit Discipline

- Run Cargo commands serially and reuse the workspace `target` directory.
- Before each unit, confirm `git status --short` and preserve unrelated changes.
- Add or strengthen a failing characterization/contract test before changing a state owner or deleting a public seam.
- Format only files in scope during dirty work; use `cargo fmt --all -- --check` as the final workspace gate.
- Finish each unit with `git diff --check`; review the exact staged diff before an optional scoped commit.

### Contract Test Matrix

| Contract | Required evidence |
|---|---|
| Completed-step atomicity | Multi-tool preparation failure leaves semantic report/journal/approvals/messages/deferred state unchanged while model-attempt and provider usage accounting settles exactly once. |
| Agent/runtime parity | Agent, ToolLoop, structured, and durable scenarios produce equivalent public outcomes. |
| Tool effect safety | Crash/retry matrix proves persisted dispatch precedes effects and completed/indeterminate work is not replayed incorrectly. |
| Approval safety | Stale/replayed/unknown/drifted approvals fail before dispatch; provider-owned work bypasses local approval. |
| Deferred ledger | Exact-scope identity, stable order, last-wins, resolution, mixed progression, serialization, successor monotonicity, and failed/cancelled/EOF staged-observation discard. |
| Snapshot/store | v8 round trips, v7 envelope rejection, invalid/oversized candidate zero-CAS, pre-deserialization store bounds, external/built-in store parity. |
| WebSocket lifecycle | Queue/send/ack/terminal certainty matrix, one terminal, EOF discipline, finite queue/timeout/frame/event bounds, actor cleanup, generated/warm-up distinction. |
| OpenAI kernel | Compatible route-local behavior preserved; official direct/stream/native routes satisfy route-local parity and dated official wire oracles; native resources remain provider-owned; external `extension::v2` provider-author compile contract passes. |
| Diagnostics and persistence | Debug, Display, ordinary errors, and traces reveal no secrets/private payloads. Explicit snapshot serialization and sensitive accessors retain bounded replay bytes and are documented as sensitive authority-bearing data. |
| Public breaks | Positive compile contracts prove retained APIs; exact deletion searches and migration docs cover removed constructors, advanced exports, schema/ABI, and moved read-only types. |

### Expanded Gates

Run after U4 and again after U7:

```bash
cargo fmt --all -- --check
cargo metadata --format-version 1 --locked --no-deps
cargo nextest run -p siumai-runtime --all-features --test-threads 1
cargo nextest run -p siumai-server --all-features --test-threads 1
cargo nextest run -p siumai-provider-openai --all-features --test-threads 1
cargo nextest run -p siumai-openai-compatible --all-features --test-threads 1
cargo nextest run -p siumai --all-features --test-threads 1
cargo clippy -p siumai-runtime -p siumai-server -p siumai-provider-openai -p siumai-openai-compatible -p siumai --all-targets --all-features -j 1 -- -D warnings
```

Run final workspace/release confidence gates:

```bash
cargo nextest run --workspace --all-features --test-threads 1
cargo clippy --workspace --all-targets --all-features -j 1 -- -D warnings
cargo test --workspace --doc
cargo package --workspace --list --locked
git diff --check
git diff --cached --check
```

Also run focused no-default/facade checks:

```bash
cargo check -p siumai-runtime --no-default-features
cargo check -p siumai-provider-openai --no-default-features
cargo check -p siumai-openai-compatible --no-default-features
cargo check -p siumai --no-default-features --features runtime
cargo check -p siumai --no-default-features --features openai
cargo check -p siumai --no-default-features --features openai,openai-responses-websocket
cargo check -p siumai --no-default-features --features openai,openai-realtime
cargo check -p siumai --no-default-features --features openai-compatible
```

### Deletion Searches

Final repository searches must show no active references to:

- public internal snapshot/event/log constructors removed by U4;
- `dispatch_id` in runtime snapshot wire;
- `RunReport::execution_log_mut` and direct engine/durable event append;
- duplicated provider-deferred position maps/normalization helpers;
- `experimental::responses_websocket::advanced` and public connector/socket/sender/receiver types;
- official OpenAI duplicate HTTP/SSE execution loops superseded by the kernel;
- stale snapshot v7 / durable ABI v6 current-contract documentation.

---

## Definition of Done

- [ ] U1-U7 are implemented in dependency order with no skipped requirement.
- [ ] Agent, ToolLoop, structured, and DurableToolLoop share one completed-step planning path.
- [ ] Completed-step semantic state commits atomically, while consumed model attempts and provider-reported usage settle exactly once on later classification failure.
- [ ] The durable tool journal is the only execution-transition writer; duplicate restore and raw append paths are deleted.
- [ ] Provider-deferred state has one ledger owner with exact replay-scope identity, stable order, monotonic resolution, and mixed local/deferred progression.
- [ ] Failed, cancelled, and unexpected-EOF streams cannot commit staged provider-deferred observations as resumable work.
- [ ] Snapshot v8 and durable ABI v7 are implemented once; v7 fails closed; `dispatch_id` and public internal constructors/mutators are removed.
- [ ] `RunStore` remains a storage/CAS port and does not own successor semantics.
- [ ] Every checkpoint path enforces snapshot size before CAS; external stores document and test sensitive-state confidentiality/integrity/isolation and pre-deserialization bounds.
- [ ] Responses WebSocket uses one private turn lifecycle and exposes no public advanced connector/socket seam.
- [ ] `siumai-openai-compatible` owns the stateless OpenAI HTTP/SSE kernel; compatible and official portable/native language paths reuse it without fidelity loss.
- [ ] `extension::v2` has a documented provider-author stability contract and an external-consumer compile fixture; it cannot override endpoint/auth/signing/replay/retry/timeout/transport authority.
- [ ] Official OpenAI retains all branded options, identity, resources, background, native carriers, Realtime, WebSocket, evidence, and diagnostics ownership.
- [ ] Architecture, migration, changelogs, rustdoc, facade exports, and examples match the new public contracts.
- [ ] High-level runtime, external `RunStore`, official OpenAI, and direct compatible-provider adopter journeys remain covered after the breaking cleanup.
- [ ] Obsolete helpers, aliases, imports, tests, and documentation are deleted rather than deprecated.
- [ ] All per-unit, expanded, no-default, workspace, doctest, Clippy, metadata, formatting, packaging, and deletion gates pass serially.
- [ ] U1-U4 land as one reviewable durable compatibility-boundary commit; any later scoped commit uses an English Conventional Commit message and excludes unrelated user/agent changes.
- [ ] No publish, tag, push, release, PR creation, credentialed live test, or destructive external action occurs without separate authorization.
