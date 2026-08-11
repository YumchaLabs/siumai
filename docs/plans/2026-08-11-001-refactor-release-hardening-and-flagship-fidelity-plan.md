---
title: Release Hardening and Flagship Provider Fidelity - Plan
type: refactor
date: 2026-08-11
deepened: 2026-08-11
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Release Hardening and Flagship Provider Fidelity - Plan

## Goal Capsule

| Field | Contract |
|---|---|
| Objective | Turn the completed semantic and ownership refactors into a release-hardened `0.11.0-beta.9` workspace by strengthening deterministic gates, reducing the worst-case cost of OpenAI Responses reconciliation, deleting over-modelled OpenAI request validation, and closing the highest-value OpenAI and Anthropic provider-native gaps. |
| Product boundary | Preserve the six provider-neutral model-family traits and the complementary provider-owned native APIs. Do not add a universal client, a provider capability matrix, another model-policy layer, or generic resource/session traits. |
| Release contract | Keep every workspace package at `0.11.0-beta.9`. Breaking source changes and deletion of obsolete validation or helper code are allowed. Do not move the workspace to `0.12`. This plan does not publish or attempt to overwrite a registry version; any later public-version decision is a separate maintainer action. |
| Evidence contract | Current official provider documentation is primary evidence. The local `repo-ref/ai` checkout is secondary prior art. Deterministic offline fixtures and Cargo gates are release evidence; credentialed live calls remain opt-in diagnostics. |
| Execution posture | Prefer deletion, provider-owned typed APIs, small conformance fixtures, and direct Cargo/CI gates. Do not grow repository scripts into parsers for Rust semantics, call graphs, capability matrices, or provider-by-model test generators. |
| Tail ownership | Update CI, release guidance, public rustdoc/README examples, provider support evidence, and migration/status wording with the implementation. Commit reviewable units with English Conventional Commit messages. Do not push, publish, tag, or open a pull request. |
| Stop conditions | Stop only for an official wire contradiction that changes the product contract, a security or replay ownership gap that cannot be bounded locally, or unrelated overlapping worktree changes that cannot be isolated safely. |

---

## Product Contract

### Summary

Siumai's macro architecture is now settled. The unified API remains valuable because it gives Rust callers a small portable contract, while provider-owned APIs preserve product fidelity for resources, sessions, hosted tools, and protocol-specific lifecycle operations.

The next release risk is no longer missing architecture. It is proving that the broad refactor is consistently buildable, packageable, documented, and faithful at the two flagship providers most likely to exercise the deepest request, response, streaming, caching, tool, and resource paths.

This plan therefore makes five focused improvements:

1. PR and release gates exercise the flagship protocol/provider crates instead of relying on core/facade-only fast coverage.
2. OpenAI Responses terminal alignment becomes bounded and index-driven rather than repeatedly scanning large output collections.
3. Remaining local validation is classified by ownership: stable wire, security, replay, and resource rules remain strict; invented marker roles, resource-ID grammars, mutable product eligibility, and model-name knowledge are removed or advisory.
4. OpenAI completes the remaining high-value Conversations/reasoning/tool ergonomics, while Anthropic closes compaction, Files-in-Messages, hosted-tool replay, Batches, and Skills lifecycle gaps without changing core.
5. OpenAI and Anthropic examples and support wording show the actual provider-owned and portable paths, without claiming complete platform coverage.

### Problem Frame

The completed `0.11.0-beta.9` line already includes canonical tool input, typed in-band failures, exactly-once stream settlement, typed direct failure context, bounded replay data, exact-target provider options, OpenAI Conversations/Files/Vector Stores/Skills, Responses WebSocket, OpenAI media families, and Anthropic Files/Batches/Token Counting/Skills plus current Messages options.

Five release-quality gaps remain:

- the PR fast lane tests core, runtime, transport, Registry, and the facade, but does not directly run the OpenAI protocol, OpenAI provider, OpenAI-compatible engine, Anthropic-compatible engine, or Anthropic provider suites that own the highest-risk wire behavior;
- the OpenAI Responses reconciler has correct semantic alignment, but its positional fallback still performs repeated scans across terminal and streamed output, which can approach quadratic work inside a deliberately large bounded turn;
- OpenAI Prompt Caching currently assigns client-side `Historical` and `WriteCandidate` roles even though the wire encodes one breakpoint and the service decides read/write behavior. Additional OpenAI validation invents unsupported minimums, non-empty text requirements, resource-ID character grammars, and required input-token fields;
- OpenAI Conversations lacks item retrieve/delete, reasoning mode is unnecessarily closed, and the hosted-tool request surface omits several current typed variants despite retaining raw fidelity;
- Anthropic compaction is not fully decoded, streamed, replayed, or conservatively terminated; Files cannot be referenced from Messages; hosted-tool results lack a usable typed/replay surface; Batches do not aggregate request beta requirements or stream JSONL results; and Skills lifecycle/upload constraints remain partial.

The user-facing surface also understates the flagship journeys. The root README lists features but does not teach a compact OpenAI Responses call with exact typed options, an Anthropic Messages call with current request-level options, or the relationship between portable models and provider-owned lifecycle resources.

### Requirements

#### Stable product and version

- R1. Keep every workspace package at `0.11.0-beta.9`; no `0.12` version migration is part of this plan.
- R2. Preserve the six provider-neutral family traits and the provider-owned native resource/session APIs.
- R3. Do not introduce a universal client, generic resource trait, capability boolean matrix, runtime model policy, or second neutral request/response type family.
- R4. Delete obsolete helpers, aliases, tests, and validation branches when their ownership or product purpose no longer exists.

#### Release gates

- R5. The ordinary PR lane must directly exercise representative OpenAI and Anthropic protocol/provider contracts, including direct response, stream settlement, typed options, and provider construction, without enabling live network calls.
- R6. The release lane must retain the workspace/all-features test and Clippy gates, formatting, docs, doctests, MSRV, architecture boundaries, and package inspection guidance.
- R7. Feature checks must cover the facade's default-free OpenAI, Anthropic, Responses WebSocket, and Realtime feature ownership without creating a provider-by-feature Cartesian matrix.
- R8. Release tooling must orchestrate authoritative Cargo and repository checks only. It must not duplicate Cargo metadata semantics or infer Rust callability.

#### OpenAI Responses reconciliation

- R9. Terminal alignment must remain semantically strict for portable content, executable identity, caller-owned tool name/call ID/canonical JSON, duplicate terminal, event-after-terminal, malformed lanes, and unexpected EOF.
- R10. Provider-native bookkeeping drift remains native or marks replay unavailable; it does not become a portable semantic failure unless the field is explicitly replay-critical. Portable success and replay eligibility are separate outcomes.
- R11. Alignment and merge work must be bounded near `O(streamed_items log n + terminal_items log n)` or equivalent indexed behavior. No terminal item may repeatedly rescan the full streamed item collection, and no missing-item merge may repeatedly insert into the middle of a growing vector.
- R12. Item-count, turn-byte, tool-input, metadata, and diagnostic bounds remain enforced before large clones or user-visible publication.
- R13. Official OpenAI and explicitly maintained compatible dialect fixtures must retain their current omission and identity rules. A caller-controlled custom endpoint does not automatically inherit a relaxed dialect.

#### Validation ownership and future compatibility

- R14. Unknown and future model IDs remain callable whenever the selected protocol/family can encode the request structurally.
- R15. An explicit typed option reaches final wire or fails with a typed structural, security, resource, or stable wire error. It is never silently removed by a model-name prediction.
- R16. Retain exact-model branches only when current official evidence proves a stable incompatibility in the selected wire or resource operation. Each retained branch has a source date and a focused known-model plus unknown-model-baseline fixture.
- R17. Model lifecycle, rolling aliases, commercial availability, account eligibility, region availability, and documented product recommendation remain advisory evidence and cannot gate construction or execution.
- R18. Bounded raw provider-body options may carry future fields and future enum strings after path-aware canonical/protected validation. They never gain endpoint, credential, header, signing, retry, or transport authority.

#### Flagship provider fidelity and ergonomics

- R19. OpenAI documentation and tests must distinguish the portable language/media families from provider-owned Responses resources, Conversations, Files, Vector Stores, Skills, Realtime, and Responses WebSocket sessions.
- R20. Anthropic documentation and tests must expose the typed Messages controls already implemented, including automatic cache control, service tier, speed, container/skills, MCP servers, context management, inference geography, task budget, and feature-derived beta headers.
- R21. OpenAI Chat and Responses retain direct/stream usage parity, trailing usage-only handling, bounded metadata, prompt-cache intent, input-token counting, and typed failure settlement.
- R22. Anthropic direct and stream paths retain request/response parity, feature-derived beta headers, explicit node cache annotations, automatic request caching, typed resources, and open future model IDs.
- R23. Public examples use provider-owned typed options and resources directly, then show how to obtain the portable model adapter. They do not imply that a portable trait exposes the provider's entire product surface.
- R24. Support documents use the exact labels `claimed slice complete`, `provider platform complete`, and `intentionally deferred`; no wording may claim that every provider's complete product surface is aligned.

#### Live diagnostics

- R25. Credentialed sub2api/OpenAI-compatible canaries remain operator-authorized, temporary, secret-free diagnostics after offline gates pass.
- R26. Live relay capacity, unsupported continuation, quota, timeout, or provider availability is reported as operational evidence unless an offline reproduction proves a Siumai defect.
- R27. No credential, base URL, response body, tool argument, raw provider payload, response ID, or provider close reason enters source control, default logs, public diagnostics, or the plan artifact.

#### OpenAI provider-native completion

- R28. Replace `OpenAiPromptCacheMarker::{Historical, WriteCandidate}` with one node-scoped explicit breakpoint annotation. Do not locally classify service-side read/write outcome, enforce marker-role order, or impose a semantic 3/4 write-candidate budget that the wire cannot prove.
- R29. Preserve ordinary request/body/node bounds for prompt caching and keep TTL, cache mode, and deprecated retention as provider-owned typed controls; document retention as deprecated/advisory rather than deleting wire fidelity.
- R30. Remove undocumented OpenAI minimum/non-empty validation for `compact_threshold`, `max_tool_calls`, optional input-token fields, instructions, `user`, prompt-cache key, and raw duplicate include values. Keep JSON kind, maximum size, control-character, canonical-field, relationship, and transport/security validation when the official wire or local resource boundary proves it.
- R31. Treat provider resource IDs as opaque bounded path segments. Pass raw segments to a trusted typed segment builder that performs one encoding step and rejects path/query/fragment escape; do not relax generic target validation or invent an ID alphabet.
- R32. Add Conversations item retrieve and delete to complete the declared item lifecycle slice.
- R33. Make OpenAI reasoning mode an open provider-owned value with known `standard` and `pro` constants plus future/custom string support.
- R34. Add typed request ergonomics for the current high-value hosted-tool variants that are already representable on wire, while retaining `OpenAiRawTool` and lossless unknown native output/events. Do not copy the entire volatile provider union into one release.
- R35. Defer Vector Store search/file-batch breadth and Skills ZIP upload unless they are required by another unit; keep their support-policy exclusions explicit.

#### Anthropic provider-native completion

- R36. Map `compaction` and unknown future stop reasons conservatively to incomplete outcomes. Decode `compaction_delta`, preserve/replay compaction blocks only for the same complete provider scope/domain/caller scope, and retain the current compaction beta header.
- R37. Enforce the current documented compaction trigger minimum of 50,000 for the versioned compaction edit, remove the unsupported global 20,000 task-budget minimum, and remove the obsolete mid-conversation system beta header while retaining feature requirements still documented for tool changes, compaction, MCP, and other beta surfaces.
- R38. Add Anthropic file references and container uploads to Messages, derive the Files beta requirement from actual file use, bind untrusted file IDs to the configured provider scope, and correct filename/size validation to current official limits without promoting Files into core.
- R39. Add provider-owned typed inspection and exact-scope replay for maintained hosted-tool use/result blocks. Provider-executed tools remain non-executable portable opaque data; they never become local `ToolCall`.
- R40. Make Message Batches aggregate per-item feature requirements, reject only current documented exclusions, expose bounded unordered JSONL result decoding, and preserve unknown status/result values.
- R41. Complete the declared Anthropic Skills lifecycle with list/delete skill and create/delete version operations, enforce current aggregate upload bounds, and validate per-file upload structure without implementing a ZIP parser.
- R42. Update Anthropic resource source URLs, verification dates, and claims; remove or narrow any upload field whose current official schema cannot prove it.

#### Cross-cutting resource, replay, and diagnostic safety

- R43. Opaque provider IDs are encoded through a trusted typed path-segment builder owned by the provider/transport boundary. Do not relax the generic `RequestTarget` validator, concatenate IDs into URLs, or decode caller-provided percent escapes before encoding.
- R44. Durable replay identity remains `ProviderScope` plus `ReplayDomain` and its caller scope where required; `ProviderInstanceId` is an execution capability, not a serialized replay key. Same scope/domain instances may replay when the existing ADR permits it; different audience, caller scope, API mode, protocol, or missing domain fails closed.
- R45. OpenAI and Anthropic hosted/server tools remain provider-owned request/output data. They never become caller-owned `ToolCall`, local approvals, or runtime-executable tools; remote URLs, headers, and MCP data remain provider wire fields with redacted diagnostics.
- R46. Multipart Files/Skills paths reject absolute paths, `..`, backslashes, NUL/control characters, empty or duplicate normalized paths, and Unicode/case collisions before cloning or streaming large content. Aggregate, per-file, decoded, and filename bounds are enforced at the owning resource boundary.
- R47. Batch JSONL decoding is incremental end to end: it never first downloads an unbounded `Bytes` body, bounds encoded/decoded bytes, line length, depth, strings, records, and diagnostics, and returns one typed error for a malformed/truncated line after already-emitted records.
- R48. Destructive native operations (delete, cancel, update, version deletion) use `ReplaySafety::Never` or an equivalent no-automatic-retry policy; only safe reads may be retried after an uncertain submission.
- R49. Raw and typed overlays have an explicit protected-field negative matrix covering endpoint, method, authorization, API key, transport retry/timeout, canonical model/input/messages/tools, streaming controls, nested parent replacement, duplicate/case variants, and URL query/fragment escape.
- R50. Live diagnostics read only the minimum configured values in memory, never place credentials or private endpoints in argv, temporary files, shell traces, full environment dumps, or debug output, use synthetic bounded prompts, and clean up from a `finally` path outside the workspace.
- R51. Default `Debug`, `Display`, tracing, and serialized diagnostics redact prompt-cache keys, identifiers, file/skill IDs, hosted-tool/MCP URLs, malformed batch lines, WebSocket close reasons, and provider response bodies; explicit typed accessors may expose values only by caller choice.

### Key Flows

- F1. Flagship PR validation
  - **Trigger:** A pull request changes core, protocol, provider, facade, feature, or release behavior.
  - **Steps:** The fast repository checks run the core/runtime baseline and one grouped flagship suite containing exactly `siumai-protocol-openai`, `siumai-provider-openai`, `siumai-openai-compatible`, `siumai-protocol-anthropic`, `siumai-anthropic-compatible`, and `siumai-provider-anthropic`; facade default-free feature checks compile the two flagship providers and the independent WebSocket/Realtime features.
  - **Outcome:** High-risk wire regressions fail before merge without requiring the full workspace lane or live credentials.
  - **Covered by:** R5-R8

- F2. Large Responses terminal reconciliation
  - **Trigger:** A bounded Responses stream observes many incremental output items and receives a terminal resource with explicit identities, documented omissions, or terminal-only native bookkeeping.
  - **Steps:** The reconciler builds stable ID/call/position indexes once, aligns explicit identities, resolves only uniquely legal positional fallbacks, validates portable/replay-critical parity, and rebuilds the final output in one merge pass.
  - **Outcome:** The same semantic result and errors are preserved without repeated full-collection scans or middle-vector insertions.
  - **Covered by:** R9-R13

- F3. Future model with explicit options
  - **Trigger:** A caller uses an unlisted future model and selects a typed OpenAI or Anthropic option.
  - **Steps:** The provider applies protocol-shape, security, resource, and relationship validation; feature-derived wire requirements are added; no lifecycle or model-name capability table rewrites the request.
  - **Outcome:** The option reaches final wire or a typed stable error explains why that protocol cannot represent it.
  - **Covered by:** R14-R18, R21-R22

- F4. Provider-owned plus portable usage
  - **Trigger:** A caller needs both a common agent loop and provider-native lifecycle/resource features.
  - **Steps:** The configured provider exposes typed native resources/sessions and constructs the appropriate portable family model. Provider options target the concrete model; native resources stay on the provider.
  - **Outcome:** The caller does not choose between portability and fidelity, and no provider-specific lifecycle is forced into core.
  - **Covered by:** R2-R3, R19-R23

- F5. OpenAI caller-first request and resource lifecycle
  - **Trigger:** A caller uses a future model, explicit prompt-cache breakpoint, custom reasoning mode, or provider resource ID unknown to this release.
  - **Steps:** The provider validates stable shape/security/resource bounds, encodes opaque IDs as path segments, preserves open values, and invokes the typed Conversations/tool/resource operation without model-name or invented ID/marker roles.
  - **Outcome:** Valid future provider input reaches wire, while canonical-field, path-escape, endpoint, auth, and resource-bound violations still fail before transport.
  - **Covered by:** R28-R35

- F6. Anthropic compaction and native replay
  - **Trigger:** A Messages response compacts context or returns hosted-tool blocks, and the caller continues the conversation.
  - **Steps:** Direct and stream decoders preserve the provider-owned block, portable termination remains conservative, replay checks the exact scope/domain, and request policy adds only feature-derived beta requirements.
  - **Outcome:** Context continuity remains replayable without promoting provider-executed tools into the local tool loop.
  - **Covered by:** R36-R39

- F7. Anthropic native resource lifecycle
  - **Trigger:** A caller references a File in Messages, creates a feature-bearing Batch, consumes unordered JSONL results, or manages a Skill/version.
  - **Steps:** Provider-owned typed resources validate current request constraints, aggregate required beta features, stream bounded results, preserve unknown values, and keep IDs scoped to the configured provider.
  - **Outcome:** The declared Anthropic resource slice is usable end to end without core resource abstractions or stale exclusion tables.
  - **Covered by:** R38-R42

- F8. Opt-in live canary
  - **Trigger:** Offline gates pass and an operator explicitly authorizes current credentials and endpoint use.
  - **Steps:** The operator checks upstream health, runs a temporary bounded probe for representative direct/stream/cache/tool/WebSocket behavior, records only categories and timestamps, and deletes the temporary harness.
  - **Outcome:** Live behavior informs maintenance without becoming a flaky or secret-bearing release gate.
  - **Covered by:** R25-R27

### Acceptance Examples

- AE1. Covers F1. A change that breaks Chat trailing usage, Responses settlement, Anthropic beta-header derivation, or provider construction fails the PR flagship suite without running a network call.
- AE2. Covers R7. The facade compiles with `openai`, `anthropic`, `openai-responses-websocket`, and `openai-realtime` in the documented default-free combinations; the WebSocket feature still owns and enables its OpenAI dependency intentionally.
- AE3. Covers F2. A terminal resource with hundreds or thousands of mixed message/reasoning/function/native items aligns with the same output and replay status as the small fixture while the implementation performs indexed lookup and one final merge.
- AE4. Covers R9-R13. Explicit item-ID conflict, caller tool name/call-ID/JSON mismatch, duplicate terminal, event after terminal, or incomplete EOF remains a typed protocol failure after the performance refactor.
- AE5. Covers R10. A provider-native bookkeeping mismatch may return portable success when portable semantics agree, marks replay unavailable, and never synthesizes merged replay state; a field classified replay-critical remains a typed failure.
- AE6. Covers F3. A future OpenAI model with explicit reasoning effort and a future Anthropic model with current feature-driven options reach their respective final request bodies without a model catalog gate.
- AE7. Covers U3 and R14-R16. A future OpenAI embedding model with explicit dimensions reaches final wire; any retained exact-model dimension restriction has dated official evidence and a separate unknown-model baseline, otherwise the branch is deleted.
- AE8. Covers R18. A future provider enum string passes through the reviewed raw body overlay, while canonical message/model/input fields and any transport/header authority remain rejected.
- AE9. Covers R19-R23. One OpenAI example shows portable Responses plus provider-owned input-token or conversation lifecycle access; one Anthropic example shows typed request options and explicit provider-owned resources without adding core APIs.
- AE10. Covers R24. The support matrix uses the exact labels `claimed slice complete`, `provider platform complete`, and `intentionally deferred`, says every current public claim has dated official evidence, and separately states that not every provider's complete product platform has been audited or implemented.
- AE11. Covers F8. A live canary can fail because a relay rejects continuation or WebSocket service is unavailable without changing deterministic protocol claims or causing a release failure.
- AE12. Covers R4 and R15-R17. The final diff contains no obsolete policy alias, warning-only omission branch, or duplicated validation helper retained solely for compatibility.
- AE13. Covers R28-R30. A request with several explicit OpenAI breakpoint annotations is encoded with one wire marker per annotated node; no client-side historical/write role, ordering error, or invented write-candidate count rejects it.
- AE14. Covers R30-R31. A resource ID containing legal non-control punctuation is percent-encoded as one path segment, while embedded path/query/fragment escape cannot alter the target URL.
- AE15. Covers R32-R34. Conversations item retrieve/delete work through typed methods, a custom reasoning mode reaches wire, and representative hosted-tool variants have typed constructors while unknown variants retain raw fidelity.
- AE16. Covers R36-R37. Direct and stream compaction produce an incomplete portable outcome plus replayable native state; an unknown stop reason is incomplete, the obsolete system beta is absent, and a future model can use a positive task budget below 20,000.
- AE17. Covers R38-R39. A same-scope Anthropic file reference and hosted-tool result can be replayed, while a foreign-scope ID/block is rejected and no provider-owned tool is exposed as a local executable call.
- AE18. Covers R40. A Batch combining Skills, MCP, context editing, and task budget emits the union of required beta headers, rejects only current documented exclusions, and decodes unordered result lines incrementally within bounds.
- AE19. Covers R41-R42. Skills list/delete and version create/delete complete the typed lifecycle; multi-file upload requires the documented top-level structure and aggregate limit, while ZIP content remains server-validated opaque data.
- AE20. Covers R43-R44. Resource IDs containing `/`, `\\`, `.`, `..`, `?`, `#`, `%2F`, `%252F`, Unicode, controls, or empty input are tested through the trusted segment builder; the final origin/base path/query remain unchanged, and replay fixtures distinguish same complete scope/domain from different caller scope, API mode, protocol, or missing domain without serializing `ProviderInstanceId`.
- AE21. Covers R45-R49. Hosted tools cannot enter the local runtime loop; multipart paths reject traversal/collision inputs before assembly; Batch JSONL is incrementally bounded and fails once on malformed/truncated input; destructive operations never automatically retry after uncertain submission; protected-field overlays fail across nested/case/duplicate variants.
- AE22. Covers R50-R51. A live-canary sentinel test and final scan prove credentials/private endpoints/provider bodies/IDs/cache keys/close reasons stay out of argv, files, logs, diagnostics, snapshots, and source control.

### Success Criteria

- The PR fast lane directly runs representative OpenAI and Anthropic protocol/provider suites and remains deterministic, offline, and serial at Cargo level.
- The full release gates, facade feature checks, docs/doctests, MSRV, architecture checks, and package inspection instructions are current and internally consistent.
- OpenAI Responses terminal alignment no longer performs repeated full streamed-item scans per terminal item or repeated middle insertions, while all semantic and settlement fixtures remain unchanged or stronger.
- OpenAI prompt-cache annotations, optional request controls, and resource IDs no longer encode client-invented roles or grammars; Conversations item lifecycle and open reasoning mode are available through typed APIs.
- Anthropic compaction, Files-in-Messages, hosted-tool replay, Batches, and Skills lifecycle satisfy the declared provider-owned slice with conservative future-value handling.
- Every surviving exact-model/product validation branch in the audited OpenAI/Anthropic paths has current official wire/resource evidence and a focused future-model baseline; unsupported branches are deleted.
- OpenAI and Anthropic current public surfaces are accurately documented and demonstrated without claiming complete provider-platform alignment.
- No new core trait, universal resource abstraction, model policy, capability matrix, credentialed CI test, or provider-by-model-by-option matrix is introduced.
- Focused crates, dependent facade/runtime crates, workspace tests, Clippy, formatting, docs, doctests, MSRV, architecture boundaries, and diff checks pass serially.

### Scope Boundaries

#### Included

- PR and release CI lanes, maintained test orchestration, release guidance, and facade feature checks.
- OpenAI Responses terminal alignment and merge performance without changing portable semantics.
- OpenAI and Anthropic request-policy/model-name validation audit and deletion of unsupported product heuristics.
- OpenAI prompt-cache annotation simplification, resource-ID path-segment handling, Conversations item retrieve/delete, open reasoning mode, and a bounded hosted-tool ergonomics increment.
- Anthropic compaction settlement/replay, Files references in Messages, hosted-tool inspection/replay, Message Batches result streaming/feature aggregation, and Skills lifecycle closure.
- Focused OpenAI/Anthropic direct, stream, option, resource, dialect, and future-model fixtures where the audit finds a real gap.
- Root README, facade rustdoc/examples, provider support policy, migration/status, and release guidance updates.
- One final opt-in live diagnostic only if the configured endpoint is healthy and the user authorization remains applicable.

#### Deferred

- New provider product families or long-tail resources not already implemented.
  - OpenAI Vector Store search/file-batch breadth and Skills ZIP upload, unless a selected U4/U9 implementation dependency makes one of them mechanically necessary; any such exception requires a new scoped decision before code changes.
- A cross-provider resource/session abstraction.
- Broad performance rewrites outside the OpenAI Responses decoder.
- A maintained credentialed smoke-test script or live CI gate.
- Provider-by-model-by-option test generation.
- Release publishing, tagging, pushing, or pull-request creation.

---

## Planning Contract

### Context and Research

- The workspace delivery summary records that semantic trust, provider ownership, flagship breadth, OpenAI conversation conformance, and validation ownership are complete through 2026-08-10. This plan treats those contracts as baseline rather than reopening them on 2026-08-11.
- The local PR fast suite currently runs `siumai-core`, `siumai-runtime`, `siumai-transport`, `siumai-registry`, and `siumai`, while the highest-risk OpenAI/Anthropic protocol/provider suites run only in the full non-PR workspace lane.
- The OpenAI Responses decoder already owns dialect normalization, whole-item parity, replay status, resource bounds, and exactly-once settlement. Its remaining design debt is collection alignment cost, not another semantic type model.
- OpenAI official documentation currently recommends Responses for new tool/reasoning workflows while retaining Chat Completions, documents Responses WebSocket as a provider-native persistent mode, and exposes Conversations, Files, Vector Stores, input-token counting, and prompt caching as product-specific surfaces.
- Anthropic official Messages documentation continues to expand request-level controls and provider-owned Files, Message Batches, token counting, Skills, tools, caching, and MCP behavior. The current Siumai typed surface already contains the main request fields; this plan verifies that they remain feature-driven and future-model-safe.
- The repository contains no `docs/solutions/` or `CONCEPTS.md` durable learning artifact. Relevant standing guidance is in `AGENTS.md`, `docs/architecture/`, ADR 0015, provider support evidence, release guidance, and the two completed 2026-08-09/10 plans.

### Sources and References

- `AGENTS.md`
- `Cargo.toml`
- `.github/workflows/ci.yml`
- `.github/workflows/release-plz.yml`
- `scripts/test-workspace.py`
- `scripts/tests/test_test_workspace.py`
- `docs/releasing.md`
- `docs/migration/siumai-next-status.md`
- `docs/migration/siumai-next.md`
- `docs/providers/support-policy.md`
- `docs/adr/0015-validation-ownership-and-forward-compatibility.md`
- `docs/plans/2026-08-09-001-refactor-openai-conversation-conformance-plan.md`
- `docs/plans/2026-08-10-001-refactor-validation-ownership-and-forward-compatibility-plan.md`
- OpenAI API documentation, verified 2026-08-11: `https://developers.openai.com/api/docs/guides/migrate-to-responses`
- OpenAI prompt caching documentation, verified 2026-08-11: `https://developers.openai.com/api/docs/guides/prompt-caching`
- OpenAI Responses WebSocket documentation, verified 2026-08-11: `https://developers.openai.com/api/docs/guides/websocket-mode`
- OpenAI Chat Completions reference, verified 2026-08-11: `https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create`
- OpenAI Responses create reference, verified 2026-08-11: `https://developers.openai.com/api/reference/python/resources/responses/methods/create/`
- OpenAI Conversations reference, verified 2026-08-11: `https://developers.openai.com/api/reference/python/resources/conversations`
- OpenAI deprecations, verified 2026-08-11: `https://developers.openai.com/api/docs/deprecations`
- Anthropic Messages reference, verified 2026-08-11: `https://platform.claude.com/docs/en/api/messages`
- Anthropic prompt caching documentation, verified 2026-08-11: `https://platform.claude.com/docs/en/build-with-claude/prompt-caching`
- Anthropic Files upload reference, verified 2026-08-11: `https://platform.claude.com/docs/en/api/beta/files/upload`
- Anthropic Message Batches create reference, verified 2026-08-11: `https://platform.claude.com/docs/en/api/messages/batches/create`
- Anthropic token counting reference, verified 2026-08-11: `https://platform.claude.com/docs/en/api/messages/count_tokens`
- Anthropic Skills create reference, verified 2026-08-11: `https://platform.claude.com/docs/en/api/beta/skills/create`
- `repo-ref/ai` at commit `3bc0d4f40df7a77af4b181bc97dc1c54843545ab` (2026-08-01); secondary implementation prior art only.

### Requirement Evidence Map

| Requirements | Primary official source | Verified | Stability classification |
|---|---|---|---|
| R28-R30 | `https://developers.openai.com/api/docs/guides/prompt-caching`; `https://developers.openai.com/api/reference/python/resources/responses/methods/create/` | 2026-08-11 | Wire field and current documented request contract; service lookback counts and cache outcome are volatile and non-gating |
| R31-R32 | `https://developers.openai.com/api/reference/python/resources/conversations/items/methods/retrieve/`; `https://developers.openai.com/api/reference/python/resources/conversations/items/methods/delete/` | 2026-08-11 | Resource lifecycle and HTTP path shape; ID alphabet remains opaque |
| R33-R34 | `https://developers.openai.com/api/reference/python/resources/responses/methods/create/` | 2026-08-11 | Open provider enum/tool union; known values are conveniences, not exhaustive validity gates |
| R35 | `https://developers.openai.com/api/reference/python/resources/vector_stores`; `https://developers.openai.com/api/reference/python/resources/skills` | 2026-08-11 | Current product breadth evidence used only to document deferral |
| R36-R37 | `https://platform.claude.com/docs/en/build-with-claude/context-management`; `https://platform.claude.com/docs/en/api/messages` | 2026-08-11 | Versioned compaction wire/beta contract and current request relationship rules |
| R38 | `https://platform.claude.com/docs/en/build-with-claude/files`; `https://platform.claude.com/docs/en/api/beta/files/upload` | 2026-08-11 | Provider-owned Files/Message source contract and current upload limits |
| R39 | `https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview`; `https://platform.claude.com/docs/en/api/messages` | 2026-08-11 | Provider-owned hosted-tool block semantics; replay scope remains Siumai's security invariant |
| R40 | `https://platform.claude.com/docs/en/api/messages/batches/create`; `https://platform.claude.com/docs/en/api/messages/batches/results` | 2026-08-11 | Batch request/result lifecycle, unordered JSONL shape, and current exclusions |
| R41-R42 | `https://platform.claude.com/docs/en/api/beta/skills/list`; `https://platform.claude.com/docs/en/api/beta/skills/delete`; `https://platform.claude.com/docs/en/api/beta/skills/versions/create`; `https://platform.claude.com/docs/en/api/beta/skills/versions/delete` | 2026-08-11 | Provider-owned Skills lifecycle and upload structure; local safety bounds must be labeled separately |
| R43-R51 | `docs/adr/0014-canonical-language-history-and-replay.md`; `siumai-transport` request-target/replay-safety contracts; `AGENTS.md` diagnostics and secret-boundary rules | 2026-08-11 | Siumai-owned security/replay/resource invariants; not provider product claims |

If an implementation detail in R28-R42 cannot be traced to the listed provider-owned source or a more specific current official page, narrow or defer it. `repo-ref/ai` may supply fixture ideas but cannot establish support or validity by itself.

### Key Technical Decisions

- KTD1. **Keep `0.11.0-beta.9` and harden it rather than moving to `0.12`.** The workspace is already unified on this pre-release version, and the purpose of this round is to prove and polish the current breaking line rather than create another migration boundary. (session-settled: user-directed — chosen over a `0.12` version reset.) Governs R1, R4.
- KTD2. **Preserve the current core architecture and deepen provider modules.** The six family traits, provider-owned native APIs, Registry, runtime, and transport/protocol/provider ownership split remain the product architecture. (session-settled: user-approved — chosen over another core reset because the current boundary now encodes the desired unified-plus-native model.) Governs R2-R4, R19-R23.
- KTD3. **Offline conformance and Cargo gates are release authority; live calls are diagnostic.** The sub2api relay can reveal current interoperability or operational behavior, but it cannot replace deterministic fixtures or make upstream capacity a parser verdict. (session-settled: user-approved — chosen over credentialed live CI and flaky release gates.) Governs R5-R8, R25-R27.
- KTD4. **Strictness follows ownership, not fear of provider change.** Stable semantic, security, replay, resource, and wire-shape invariants remain strict; mutable model/product facts are advisory, and explicit typed intent is wire-or-typed-error. Governs R14-R18.
- KTD5. **Optimize the existing Responses reconciler instead of introducing another event or response model.** Build one alignment index and one merge result while preserving the current dialect and semantic matrix. Governs R9-R13.
- KTD6. **Keep provider resources native and add portable adapters only for proven family subsets.** OpenAI Conversations/Files/Vector Stores/Skills/Realtime/WebSocket and Anthropic Files/Batches/Skills remain provider-owned; examples teach both paths instead of abstracting them. Governs R2-R3, R19-R23.
- KTD7. **Expand CI with a small flagship slice, not a Cartesian matrix.** PRs get one grouped OpenAI/Anthropic suite and selected facade feature checks; the full workspace remains the release-level backstop. Governs R5-R8.
- KTD8. **Use repository scripts only as bounded orchestrators.** Extend the current test runner where command grouping adds value; use Cargo and workflow commands directly for metadata, package, docs, MSRV, and feature checks. Do not add a partial Rust or Cargo semantic analyzer. Governs R6-R8.
- KTD9. **Model OpenAI prompt caching as one explicit breakpoint, not caller-guessed service state.** The service owns cache lookup and write selection; Siumai owns only the node annotation, stable bounds, and typed request controls. This supersedes the `Historical`/`WriteCandidate` decision in the 2026-08-09 plan because current official evidence shows one wire marker and conflicting service lookback counts. (session-settled: user-approved breaking simplification — chosen over preserving an unverifiable role model.) Governs R28-R30.
- KTD10. **Close Anthropic product gaps in provider and protocol crates, not core.** Compaction, Files references, hosted-tool replay, Message Batches, and Skills lifecycle remain Anthropic-owned even when portable termination, usage, or opaque replay participates. Governs R36-R42.
- KTD11. **Defer breadth that does not close a declared slice.** OpenAI Vector Store search/file batches, Skills ZIP upload, and exhaustive hosted-tool variants remain explicit follow-up work unless implementation dependencies prove otherwise. Governs R34-R35.
- KTD12. **Add a narrow path-segment seam instead of weakening transport validation.** Provider resource codecs pass raw bounded IDs to a trusted segment builder that performs one encoding step; generic URL/request-target validation remains strict for callers and raw JSON. Governs R31, R43, R49.
- KTD13. **Honor the accepted replay identity model.** Do not serialize or compare `ProviderInstanceId` as durable replay identity. Use complete `ProviderScope`/`ReplayDomain`/caller scope semantics from ADR 0014, with explicit sensitive-resource caller scopes and fail-closed missing domains. Governs R38-R39, R44.
- KTD14. **Treat hosted tools as opaque provider execution.** Typed inspection improves ergonomics and replay, but neither OpenAI nor Anthropic hosted/server tools enter the portable local tool loop. Governs R34, R39, R45, R51.

### High-Level Technical Design

```text
official provider docs + existing ADRs
                 |
                 v
      provider/protocol ownership audit
                 |
     +-----------+-------------+
     |                         |
stable local invariant   mutable product fact
     |                         |
typed validation/test      delete or advisory
     |
final wire / decoder / native resource
                 |
                 v
deterministic flagship fixtures -> PR flagship lane
                 |
                 v
full workspace + docs + MSRV + package release gates
                 |
                 v
optional secret-free live diagnostic
```

The Responses performance change remains internal to `siumai-protocol-openai`: decode state produces the same canonical events and terminal resource, but explicit ID/call/position indexes are built once and shared by normalization and typed reconciliation. The final output is reconstructed in one ordered merge rather than mutated through repeated insertion.

### Assumptions and Constraints

- A changed official provider rule must be verified from the owning provider documentation before code changes.
- The local AI SDK reference may suggest fixtures or ergonomics but does not own Siumai's public Rust architecture.
- `0.11.0-beta.9` is the user-directed workspace baseline for this refactor. This plan validates and commits that source state but performs no registry publication; a later publishable version, if any, requires a separate explicit release decision.
- Cargo commands run serially with the shared workspace target directory.
- Default tests remain offline and secret-free.
- Public source, rustdoc, examples, and technical documentation are written in English; maintainer communication remains Chinese.
- Other user or agent edits in the shared worktree are preserved and never reset, restored, stashed, or deleted.

### Sequencing

1. Establish the new flagship PR gate first so later provider/protocol changes are continuously covered.
2. Refactor Responses alignment next, before adding or changing provider assertions that depend on terminal reconciliation.
3. Simplify OpenAI request validation and prompt-cache annotations before extending Conversations/reasoning/tool ergonomics.
4. Repair Anthropic compaction and request-policy semantics before adding Files/tool replay, then close Batches and Skills lifecycle gaps.
5. Add facade feature/package gates and update examples, ADRs, support evidence, migration/status, and release documentation from the final public shape.
6. Run full release gates, then perform the optional live diagnostic only after deterministic success.

---

## Implementation Units

### U1. Add a deterministic flagship PR suite

- **Goal:** Give OpenAI and Anthropic protocol/provider regressions a normal PR gate without making every PR run the whole workspace.
- **Requirements:** R5-R8, R21-R22; AE1-AE2.
- **Files:**
  - `scripts/test-workspace.py`
  - `scripts/tests/test_test_workspace.py`
  - `.github/workflows/ci.yml`
  - `scripts/README.md`
  - `docs/releasing.md`
- **Approach:**
  1. Add one named flagship suite to the maintained test orchestrator.
  2. Include exactly `siumai-protocol-openai`, `siumai-provider-openai`, `siumai-openai-compatible`, `siumai-protocol-anthropic`, `siumai-anthropic-compatible`, and `siumai-provider-anthropic`; these packages own the direct/stream/options behavior under test.
  3. Keep Cargo serial, offline, no-fail-fast, and independent from credentialed tests.
  4. Invoke the flagship suite in PR CI after the existing fast baseline.
  5. Add selected default-free facade checks for OpenAI, Anthropic, Responses WebSocket, and Realtime ownership; do not enumerate every feature combination.
- **Execution note:** Characterize the existing `fast` and `full` command lists first, then add the new suite and observe the command-construction test failure before changing the implementation.
- **Test files:**
  - `scripts/tests/test_test_workspace.py`
- **Test scenarios:**
  - `flagship` with nextest emits one serial package command containing the intended protocol/provider crates.
  - `flagship` with cargo-test preserves single-threaded test execution.
  - `fast` and `full` retain their existing command sets and behavior.
  - CI invokes no live or credentialed target.
- **Verification:** The Python contract tests pass, workflow syntax remains valid, and dry-run output contains the expected bounded commands.

### U2. Make Responses terminal alignment index-driven

- **Goal:** Preserve current OpenAI Responses semantics while removing repeated full-collection scans and middle-vector insertions from terminal reconciliation.
- **Requirements:** R9-R13; AE3-AE5.
- **Files:**
  - `siumai-protocol-openai/src/responses/stream.rs`
  - `siumai-protocol-openai/src/responses/tests.rs`
  - any existing protocol-local test helper directly owning Responses fixtures
- **Approach:**
  1. Introduce one internal alignment workspace containing explicit item-ID indexes, stable call-identity indexes, ordered streamed positions, terminal anchors, and consumed mappings.
  2. Resolve explicit identities first, then bounded unique positional fallback between anchors according to `ResponsesWireDialect`.
  3. Reuse the same alignment result for raw terminal normalization and typed item reconciliation.
  4. Build missing portable-item insertions by gap and reconstruct the final output in one pass.
  5. Keep replay-unavailable decisions and portable/executable errors identical to the current field matrix.
  6. Delete redundant scans, temporary clones, or legacy helper branches superseded by the indexed workspace.
- **Execution note:** Use characterization-first tests. Record current outputs/errors for the representative strict, compatible, replay-conflict, and incomplete-item fixtures before refactoring.
- **Test files:**
  - `siumai-protocol-openai/src/responses/tests.rs`
- **Test scenarios:**
  - Mixed message, reasoning, function call, program, and terminal-only native items preserve exact output order.
  - Official strict and explicit compatible dialect omissions produce their existing outcomes.
  - Explicit ID conflict, call identity conflict, canonical JSON mismatch, duplicate terminal, event-after-terminal, and unexpected EOF still fail.
  - Replay-critical disagreement preserves portable semantics only with replay unavailable.
  - A large bounded synthetic output aligns successfully without changing item order or exceeding turn bounds.
  - Ambiguous positional fallback remains a typed protocol error.
- **Verification:** OpenAI protocol tests pass with equivalent semantic fixtures, the large-output regression passes, and the changed alignment functions contain no per-terminal full scan or repeated `Vec::insert` pattern.

### U3. Simplify OpenAI prompt caching and request validation

- **Goal:** Delete client-invented prompt-cache state and undocumented request restrictions while preserving stable wire, security, replay, and resource bounds.
- **Requirements:** R14-R18, R21, R28-R31; AE6, AE8, AE12-AE14.
- **Files:**
  - `siumai-provider-openai/src/configured/annotations.rs`
  - `siumai-provider-openai/src/configured/options.rs`
  - `siumai-provider-openai/src/configured/provider.rs`
  - `siumai-provider-openai/src/configured/embedding.rs`
  - `siumai-provider-openai/src/configured/responses_resource.rs`
  - `siumai-provider-openai/src/configured/resources/common.rs`
  - OpenAI Chat/Responses request encoders and module-local tests that project these options
  - `siumai-provider-openai/src/configured/mod.rs`
  - `siumai-provider-openai/src/lib.rs`
- **Approach:**
  1. Replace `OpenAiPromptCacheMarker::{Historical, WriteCandidate}` with one explicit node breakpoint annotation and delete role-order/count selection code.
  2. Keep TTL, cache mode, deprecated retention fidelity, annotation/resource bounds, and node-coordinate uniqueness where they remain structurally meaningful.
  3. Remove undocumented minimum/non-empty rules for `compact_threshold`, `max_tool_calls`, optional instructions/user/cache-key fields, and input-token count request composition.
  4. Stop rejecting duplicate values in raw future `include` arrays; typed builders may normalize their own input without imposing a raw-wire validity rule.
  5. Replace resource-ID character allowlists with bounded opaque-ID validation plus a trusted provider/transport path-segment builder; keep generic `RequestTarget` validation strict and perform exactly one encoding step.
  6. Retain path-aware canonical/protected validation, endpoint/auth/header isolation, maximum size/control-character checks, and proven cross-field relationships. The negative matrix must cover nested parent replacement, duplicate/case-variant keys, `model`, `input/messages`, `tools`, `stream`, `stream_options`, HTTP method/target/auth, and transport retry/timeout fields; legal hosted-tool-internal URLs/headers stay body data and cannot alter the transport control plane.
- **Execution note:** Add failing characterization fixtures before deletion. Treat official prompt-cache lookback-count disagreement as evidence against a client validity gate, not as a reason to pick one number.
- **Test scenarios:**
  - Multiple annotated nodes encode the same explicit breakpoint marker without historical/write roles, ordering rejection, or candidate-count rejection.
  - `compact_threshold = 0`, `max_tool_calls = 0`, empty optional strings, and a model-plus-instructions input-token request reach the final body when structurally valid.
  - Future raw include strings and duplicates remain bounded and pass through.
  - A future embedding model with explicit dimensions reaches final wire; any retained exact-model dimension rule has dated official evidence and a separate unknown-model baseline.
  - Opaque punctuation in conversation/file/vector-store/skill IDs is encoded as one path segment; slash, backslash, dot segments, query, fragment, percent-escape, Unicode, and control inputs cannot alter the endpoint or bypass generic target validation.
  - Canonical body fields, endpoint/auth/header changes, nested replacement and duplicate/case-variant protected keys, oversized values, controls, invalid JSON kinds, URL query/fragment escape, and real relationship conflicts still fail before transport.
- **Verification:** OpenAI protocol/provider focused nextest and Clippy pass; old marker-role symbols and unexplained exact-model/product gates are absent from production code.

### U4. Complete the bounded OpenAI native ergonomics slice

- **Goal:** Close the small, high-value OpenAI lifecycle and typed ergonomics gaps without copying the full volatile product union.
- **Requirements:** R19, R23, R31-R35, R43-R45, R48, R51; AE9, AE14-AE15.
- **Files:**
  - `siumai-protocol-openai/src/resources/conversations.rs`
  - `siumai-provider-openai/src/configured/resources/conversations.rs`
  - `siumai-provider-openai/src/configured/resources/mod.rs`
  - `siumai-provider-openai/src/configured/options.rs`
  - `siumai-provider-openai/src/configured/tools.rs`
  - `siumai-provider-openai/src/configured/resources/common.rs`
  - `siumai-provider-openai/Cargo.toml`
  - `siumai-provider-openai/src/lib.rs`
- **Approach:**
  1. Add typed Conversations item retrieve and delete operations using the common opaque path-segment encoder and shared resource error mapping.
  2. Replace the closed reasoning-mode enum with an open validated newtype exposing `STANDARD` and `PRO` constants.
  3. Add exactly five typed request ergonomics increments: `computer_use_preview`, `local_shell`, `namespace`, versioned web-search preview, and the missing computer configuration fields. All other hosted-tool variants remain raw or existing typed forms.
  4. Keep `OpenAiRawTool` and lossless unknown native output/stream-event access as the forward-compatibility seam. Mark every hosted/server tool as provider-owned; it must never project into portable caller `ToolCall` or the runtime approval/execution loop.
  5. Add one shared resource-executor fixture proving typed HTTP error classification and sensitive response-body isolation, plus a hosted-tool-to-local-tool negative fixture and nested URL/header redaction fixture.
  6. Mark destructive Conversations item deletion as `ReplaySafety::Never`; audit any existing delete/cancel/update resource operation touched by this unit before adding a new retry policy.
  7. Add package docs.rs metadata and a concise crate-root provider-owned/portable surface map, including independent Realtime and Responses WebSocket features.
- **Execution note:** Do not add Vector Store search/file batches, Skills ZIP upload, or every current tool union member unless a selected typed variant requires shared support.
- **Test scenarios:**
  - Conversation item GET/DELETE paths, success payloads, missing resources, sanitized errors, and opaque IDs.
  - Known and future reasoning-mode values round-trip through typed options; controls/oversize fail locally.
  - Each of the five selected hosted tools produces its exact final request object and unknown tools still use the raw escape hatch.
  - Hosted/server tool output cannot become a local executable `ToolCall`; remote URLs, nested headers, credentials, and opaque output are redacted from diagnostics.
  - Resource debug/display never includes a sentinel response body, credential, signed URL, cache key, safety identifier, or opaque ID payload.
- **Verification:** OpenAI protocol/provider suites, docs.rs feature check, package metadata inspection, and focused Clippy pass.

### U5. Repair Anthropic compaction and caller-first request policy

- **Goal:** Make compaction and future stop reasons conservative and replayable, while deleting obsolete model/product gates.
- **Requirements:** R14-R18, R20, R22, R36-R37; AE6, AE8, AE12, AE16.
- **Files:**
  - `siumai-protocol-anthropic/src/messages/options.rs`
  - `siumai-protocol-anthropic/src/messages/request.rs`
  - `siumai-protocol-anthropic/src/messages/response.rs`
  - `siumai-protocol-anthropic/src/messages/stream.rs`
  - message request/response/stream contract tests
  - `siumai-provider-anthropic/src/options.rs`
  - `siumai-provider-anthropic/src/request_policy.rs`
  - `siumai-provider-anthropic/src/tests.rs`
  - `siumai-anthropic-compatible/src/model.rs`
- **Approach:**
  1. Map `compaction` and unknown future stop reasons to `LanguageTermination::Incomplete(Other(...))`, never implicit success.
  2. Decode `compaction_delta`, retain the provider-native block, and allow exact same-scope/domain opaque replay through the Messages encoder.
  3. Validate the versioned compaction trigger against the current 50,000 minimum and retain the compaction beta header.
  4. Delete the obsolete mid-conversation system beta requirement; retain independently documented mid-conversation tool-change, compaction, MCP, cache, and other feature requirements.
  5. Delete the global 20,000 task-budget minimum while keeping `total > 0`, `remaining <= total`, typed bounds, and feature-derived headers.
  6. Add direct/stream parity fixtures for an in-band error after text/reasoning/usage and for sanitized HTTP error metadata.
- **Execution note:** Do not introduce task-budget response accounting that the provider does not return, and do not select compaction/tool versions from model-name patterns.
- **Test scenarios:**
  - Direct and stream compaction settle incomplete with equivalent usage and replayable native state.
  - A same-scope compaction block re-encodes exactly; a foreign-scope block fails before transport.
  - `compaction_delta` closes exactly once; duplicate terminal, event-after-terminal, and EOF behavior remain strict.
  - Unknown stop reasons are incomplete, not completed.
  - A future model accepts a valid positive task budget below 20,000, while invalid total/remaining relationships fail.
  - Mid-conversation system content no longer emits its retired beta header; still-required feature headers remain deduplicated.
- **Verification:** Anthropic protocol, compatibility, and provider nextest plus Clippy pass.

### U6. Add Anthropic Files-in-Messages and hosted-tool replay

- **Goal:** Make current Anthropic file and hosted-tool results usable in continued Messages turns without exposing provider execution as local tools.
- **Requirements:** R20, R22-R23, R38-R39, R43-R46, R49, R51; AE9, AE17.
- **Files:**
  - `siumai-provider-anthropic/src/annotations.rs`
  - `siumai-provider-anthropic/src/resources/files.rs`
  - `siumai-provider-anthropic/src/request_policy.rs`
  - a narrow provider-owned hosted-tool inspection module if existing metadata ownership is insufficient
  - `siumai-protocol-anthropic/src/messages/request.rs`
  - `siumai-protocol-anthropic/src/messages/response.rs`
  - `siumai-protocol-anthropic/src/messages/stream.rs`
  - related provider/protocol tests and exports
- **Approach:**
  1. Add typed message sources for `file_id` references and `container_upload`, preserving request direction and provider scope.
  2. Derive the Files beta requirement from actual message content and reject foreign-scope or malformed IDs before transport. Use complete `ProviderScope`/`ReplayDomain`/caller-scope semantics; do not serialize or compare `ProviderInstanceId`.
  3. Correct upload filename and single-file size validation to the current official contract; remove or narrow unproven `purpose` input. Reject absolute paths, `..`, backslashes, empty/control segments, normalized duplicates, case/Unicode collisions, and multiple top-level roots before multipart assembly; document that the API accepts bounded in-memory bytes rather than following arbitrary local paths.
  4. Add typed provider-owned inspection for maintained `server_tool_use` and hosted-tool result blocks while preserving bounded raw unknown fields.
  5. Allow exact same provider/protocol/mode/domain/caller-scope replay of maintained hosted-tool blocks; different audience, caller scope, API mode, protocol, missing domain, mismatched identity, or unsupported replay fails closed. Same complete scope/domain across configured instances follows ADR 0014 and is not rejected solely by instance token.
  6. Keep provider-executed calls/results outside portable local `ToolCall`; the portable response may retain only bounded opaque provider state.
- **Execution note:** Reuse existing provider-opaque provenance and replay-domain contracts. Do not add a core file/tool-result type to accommodate Anthropic.
- **Test scenarios:**
  - Image/document file references and container uploads encode correctly and add the Files beta once.
  - Workspace-scoped untrusted file IDs require the selected provider scope/caller scope; same complete scope/domain across instances follows the durable replay contract, while different or missing caller scope is rejected.
  - Filename boundary, character, and 500 MB size checks match the documented operation.
  - Hosted web/MCP/code/tool-search use/results expose typed inspection, preserve bounded unknown fields, and replay only in the same complete scope/domain.
  - No hosted tool block becomes a caller-owned executable tool call or enters the local runtime tool loop; nested URLs, headers, tokens, and provider bodies stay redacted by default.
- **Verification:** Anthropic protocol/provider fixtures and dependent facade compile contracts pass with sanitized diagnostics.

### U7. Close Anthropic Message Batches and Skills lifecycle slices

- **Goal:** Complete the currently declared provider-owned Batches and Skills operations with bounded, future-compatible result handling.
- **Requirements:** R20, R22-R24, R40-R42, R46-R49, R51; AE10, AE18-AE19.
- **Files:**
  - `siumai-provider-anthropic/src/resources/message_batches.rs`
  - `siumai-provider-anthropic/src/resources/skills.rs`
  - `siumai-provider-anthropic/src/resources/files.rs`
  - `siumai-provider-anthropic/src/resources/mod.rs`
  - `siumai-provider-anthropic/src/provider.rs`
  - related provider tests
- **Approach:**
  1. Run each batch item through the same feature/request-policy derivation used by ordinary Messages and emit the union of required beta headers.
  2. Reject only current documented batch exclusions: request-level automatic cache, `max_tokens: 0`, and unsupported server-side fallbacks; preserve explicit cache breakpoints.
  3. Add a bounded incremental JSONL result decoder/stream whose records retain request identity and do not assume input order. Bound encoded and decoded bytes, line length, record count, JSON depth/node count, string lengths, and diagnostic excerpts; reject malformed UTF-8/JSON, truncated final lines, and over-limit records with one typed error after already-emitted records, never skip or log the raw line.
  4. Use open provider-owned wrappers for processing/result status so future values remain inspectable within the same string/control-character bounds.
  5. Add list/delete skill and create/delete version operations, marking delete/version-delete as `ReplaySafety::Never` and auditing existing cancel/update/delete methods for the same post-submission no-retry policy.
  6. Enforce the current aggregate upload bound and per-file top-level directory/`SKILL.md` structure for multipart uploads; reject absolute paths, `..`, backslashes, empty/control segments, normalized duplicates, case/Unicode collisions, and multiple roots before copying content. Accept ZIP as opaque archive data for server validation, not as a locally parsed path source.
- **Execution note:** Existing local file-count/path/title limits may remain as explicit Siumai safety bounds, but documentation and error text must not mislabel them as official provider limits.
- **Test scenarios:**
  - A mixed-feature batch derives one deduplicated beta-header union and preserves per-request typed options.
  - Current exclusions fail before transport; no stale unsupported-feature matrix remains.
  - Unordered JSONL results decode incrementally without an initial full-body `Bytes` cache, enforce encoded/decoded line/aggregate/depth/node/string/record limits, preserve unknown status/results, and sanitize malformed-line diagnostics.
  - Skills list/delete and version create/delete cover success/error paths.
  - Multipart files share one top-level directory with a top-level `SKILL.md`; aggregate overflow and path collisions fail before transport; destructive resource calls do not retry after an uncertain submission; ZIP is not parsed locally.
- **Verification:** Anthropic provider nextest and Clippy pass; resource construction remains network-free and provider-owned.

### U8. Strengthen facade feature and package release gates

- **Goal:** Make the feature graph and changed-package contents independently verifiable without adding a custom Cargo analyzer.
- **Requirements:** R5-R8, R19-R20; AE1-AE2.
- **Files:**
  - `.github/workflows/ci.yml`
  - `.github/workflows/release-plz.yml` only if a safe non-publishing preflight belongs there
  - `siumai/Cargo.toml`
  - `siumai-provider-openai/Cargo.toml`
  - `scripts/README.md`
  - `docs/releasing.md`
- **Approach:**
  1. Add independent compile gates for `siumai --no-default-features --lib`, `all-providers`, and the OpenAI Realtime/Responses WebSocket pair.
  2. Keep provider features additive and verify docs.rs metadata enables the intended provider-owned optional modules.
  3. Add an explicit Cargo-native package file-list/dry-run preflight for changed publishable crates in release guidance or a safe release preflight job.
  4. Use a short maintained package list or release-plz/Cargo output; do not infer workspace publishing order or parse Rust source.
  5. Ensure no release preflight publishes, tags, opens release PRs, pushes, or changes remote state; if `release-plz.yml` changes, inspect job permissions and command arguments explicitly.
- **Test scenarios:**
  - Bare facade, all providers, and Realtime/WebSocket combinations compile independently.
  - OpenAI provider docs build with optional Realtime and Responses WebSocket APIs visible.
  - Package list/dry-run excludes credentials, local paths, build output, `repo-ref`, and temporary canary artifacts.
- **Verification:** Selected `cargo check`/`cargo doc` commands pass serially and the release workflow/guidance contains only non-mutating preflight steps.

### U9. Teach the flagship journeys and record superseding decisions

- **Goal:** Make the final OpenAI and Anthropic surfaces discoverable, accurately evidenced, and explicit about what remains deferred.
- **Requirements:** R19-R24, R27-R35, R42-R51; AE9-AE10, AE13-AE15, AE19.
- **Files:**
  - `README.md`
  - `siumai/src/lib.rs`
  - `siumai/src/prelude.rs`
  - focused facade examples or compiled rustdoc modules under `siumai/`
  - `docs/adr/` for the prompt-cache validation ownership decision
  - `docs/providers/support-policy.md`
  - `docs/migration/siumai-next.md`
  - `docs/migration/siumai-next-status.md`
  - `docs/architecture/public-api.md`
  - `docs/releasing.md`
  - relevant crate changelogs where current claims are factually wrong
- **Approach:**
  1. Add one compact OpenAI journey combining an exact-target typed Responses option, the portable language model, and a provider-owned Conversations/resource call.
  2. Add one compact Anthropic journey combining current Messages options, a file/tool replay example, and a provider-owned Batch or Skill operation.
  3. Record an ADR that supersedes the previous OpenAI historical/write-candidate prompt-cache model; do not rewrite historical plan rationale as though it never existed.
  4. Update current support sources and verification dates, including the Anthropic resource URLs in this plan.
  5. Use the exact three labels `claimed slice complete`, `provider platform complete`, and `intentionally deferred` throughout current support and migration material.
  6. Mark deprecated-but-wire-faithful fields as advisory and label local safety bounds as Siumai bounds.
- **Execution note:** Prefer examples compiled by existing facade/doc gates; do not create a tutorial framework or duplicate provider reference documentation.
- **Test scenarios:**
  - Curated imports compile for provider-owned and portable paths with only documented features.
  - Examples remain offline by default and contain no credentials or private endpoints.
  - Current docs expose the single OpenAI breakpoint API and no longer recommend historical/write roles.
  - Support claims list current evidence without equating slice completion to provider-platform completeness.
- **Verification:** Facade contracts, examples, rustdoc, doctests, link/source review, and migration consistency checks pass.

### U10. Run release-hardening verification and the optional live diagnostic

- **Goal:** Prove the completed plan with deterministic release gates, then collect bounded live evidence without making it release authority.
- **Requirements:** R1-R8, R25-R27, R43-R51; AE1-AE2, AE11-AE12, AE20-AE22.
- **Files:**
  - no required production file; update `docs/migration/siumai-next-status.md` only when final verified counts or an authorized live diagnostic add durable evidence
- **Approach:**
  1. Run focused nextest and Clippy after each preceding implementation unit, serially against the shared target directory.
  2. Run formatting, Python script tests, architecture boundaries, fast and flagship suites, full workspace/all-features nextest, workspace/all-targets Clippy, docs, doctests, MSRV, metadata, facade feature checks, and package inspection in release order.
  3. Resolve implementation-caused failures; deterministic release gates are not waivable. Only a non-code package-publication limitation may be documented separately from the source gates.
  4. Perform simplification and independent code review on the complete diff; fix eligible findings and rerun affected gates.
  5. After deterministic success, check the upstream status endpoint and run one temporary sub2api canary only when the lane is healthy. Read only the selected provider fields from `~/.codex-helper/config.toml` in memory; never put credentials/base URLs in argv, shell traces, temporary files, full environment dumps, or debug output.
  6. Exercise representative direct Chat/Responses, streams, cache telemetry, caller-owned tool/history continuation, and Responses WebSocket settlement with synthetic bounded prompts/tool data, fixed timeouts/iterations/frame limits, and wire tracing disabled; record only category, retryability, protocol, and timestamp.
  7. Place any temporary harness outside the repository and clean it in a `finally`/equivalent path. Verify final `git status`, diff checks, and a sentinel secret scan; no credentials, endpoint, response body, IDs, tool arguments, close reasons, or generated build artifacts may enter source control.
- **Execution note:** No maintained live script is added. The one-off command is optional, non-gating, and must not change endpoint/credential routing based on the status check. Live failure never relaxes an offline invariant without a deterministic reproduction.
- **Verification:** Every required deterministic gate passes. Only package-publication inspection that depends on an unpublished workspace state may be documented separately as an environmental limitation; it never waives source, test, Clippy, docs, feature, metadata, or security gates. Live results remain explicitly non-gating and secret-free.

---

## System-Wide Impact

- **Core:** No planned public core type changes. Existing exact-target options, terminal semantics, usage updates, and replay contracts remain unchanged.
- **Protocol:** OpenAI Responses internal alignment changes; OpenAI Conversations item wire methods and Anthropic compaction/file/tool replay codecs expand. Portable semantics remain stable except for intentionally conservative unknown/compaction termination.
- **Providers:** OpenAI deletes unverifiable prompt-cache roles and request restrictions, opens reasoning mode, and completes a bounded native slice. Anthropic completes provider-owned compaction, Files, hosted tools, Batches, and Skills behavior.
- **Compatibility engines:** Anthropic-compatible request/stream handling may change only to carry provider-owned compaction, feature headers, and typed failures. Generic endpoints remain generic and cannot inherit branded claims or relaxed dialects.
- **Facade:** Documentation, curated imports, examples, and feature checks change; provider-native types may be newly re-exported behind existing provider features, but no universal facade client is introduced.
- **Runtime/Registry/server:** No planned behavior change. They participate in dependent verification because provider option and terminal regressions would surface there.
- **CI/release:** PRs gain a bounded flagship suite; default-free, docs.rs, and package preflights become explicit and Cargo-native.
- **Security/privacy:** No live secret enters the repository. Validation deletion cannot weaken endpoint/auth/header/replay/resource boundaries.
- **Performance:** Responses alignment should reduce worst-case CPU and allocation pressure while preserving the existing turn bounds; Anthropic batch results become bounded incremental JSONL instead of whole-body-only consumption.

---

## Verification Contract

### Per-Unit Gates

| Unit | Required evidence |
|---|---|
| U1 | Python command-construction tests, dry-run command inspection, workflow diff review |
| U2 | OpenAI protocol characterization and regression fixtures, focused Clippy, large bounded alignment case |
| U3 | OpenAI prompt-cache/request/resource validation fixtures, provider/protocol nextest and Clippy, removed-symbol scan |
| U4 | OpenAI Conversations/reasoning/tool/resource fixtures, docs.rs feature build, provider Clippy |
| U5 | Anthropic compaction/termination/request-policy direct and stream fixtures, protocol/compat/provider Clippy |
| U6 | Anthropic Files/message/tool replay fixtures, provider/protocol suites, facade compile contract |
| U7 | Anthropic Batch JSONL/beta-union and Skills lifecycle fixtures, provider nextest and Clippy |
| U8 | Bare/all-provider/Realtime-WebSocket facade checks, OpenAI docs build, Cargo-native package inspection |
| U9 | Facade contracts, examples/rustdoc/doctests, ADR/support/migration consistency review |
| U10 | Full deterministic release gates, independent review, optional categorized live diagnostic, clean secret scan |

### Release Gates

- Workspace formatting check.
- Python repository-script unit tests.
- Architecture/dependency boundary validation.
- New PR flagship suite and existing fast suite.
- Full workspace/all-features nextest, serial Cargo execution.
- Workspace/all-targets/all-features Clippy with warnings denied.
- Selected default-free facade feature checks, including OpenAI/Anthropic and independent WebSocket/Realtime ownership.
- Cargo metadata assertion that every workspace package remains exactly `0.11.0-beta.9`.
- Workspace documentation and doctests.
- Rust 1.88 MSRV workspace check.
- Locked Cargo metadata inspection after any feature/dependency change.
- Package file-list and dry-run inspection for changed crates where dependency publication state permits it.
- If release automation changes, a read-only workflow/configuration review proves there is no `cargo publish`, release-plz publish/release-PR, `git push`, tag creation, or other remote write in this task.
- `git diff --check` and staged diff check before each commit.

### Non-Gating Operational Evidence

- Upstream status check for the selected live lane.
- Authorized sub2api/OpenAI-compatible direct and stream smoke results.
- Cache telemetry classification without attributing a cache hit unless the sequence proves it.
- WebSocket close/error classification with sanitized public diagnostics.
- No live success requirement for release completion.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| PR runtime grows too much | Contributors wait for a near-full workspace suite | Keep one grouped flagship package set and selected facade checks; retain full workspace only as release/non-PR backstop |
| Reconciler optimization changes semantics | Tool/history/replay behavior regresses while performance improves | Characterize existing strict/compatible/error fixtures first and reuse one alignment result for both raw and typed paths |
| Validation deletion becomes permissive | Security or stable wire constraints are lost with product heuristics | Classify every branch by owner; keep endpoint/auth/header/replay/resource/canonical-field checks and add focused negative fixtures |
| Prompt-cache simplification loses useful intent | Callers depended on client-guessed historical/write roles | Replace both roles with the exact wire-level breakpoint concept, document the breaking migration in repository guidance, and preserve TTL/mode/retention controls |
| Native replay crosses provider scope | Files or hosted-tool state is replayed against another account or endpoint | Reuse complete ProviderScope/ReplayDomain/caller-scope provenance without serializing ProviderInstanceId; add same-scope/different-instance, different-caller-scope, different-mode, and missing-domain fixtures |
| Batch result streaming becomes unbounded | Malformed JSONL consumes memory or leaks diagnostics | Bound line, aggregate, node, and diagnostic sizes; decode incrementally and redact raw failing lines |
| Resource lifecycle breadth expands too far | This round becomes another provider-platform rewrite | Close only declared OpenAI/Anthropic slices; keep Vector Store breadth, Skills ZIP, and exhaustive tool unions deferred |
| Official docs change during implementation | A product fact becomes stale before release | Record source and verification date, keep model IDs open, and avoid turning mutable facts into generic core contracts |
| Docs imply complete platform support | Users assume unimplemented native products exist | Use the exact labels `claimed slice complete`, `provider platform complete`, and `intentionally deferred`, and enumerate only current declared slices |
| Live relay behavior is misread as provider truth | Operational outage causes code churn | Run offline fixtures first and report live results only as categorized diagnostics |
| Release scripts overgrow | Repository maintains a second Cargo/Rust analyzer | Extend only bounded command orchestration and use Cargo/workflow commands directly for authoritative checks |

---

## Definition of Done

| ID | Done when |
|---|---|
| D1 | A Cargo-metadata check proves every workspace package still reports version `0.11.0-beta.9`; no registry publication is attempted. |
| D2 | The PR CI configuration runs the existing fast suite plus one deterministic flagship OpenAI/Anthropic suite and selected facade feature checks. |
| D3 | Responses terminal reconciliation is index-driven, preserves all current semantic failures and replay outcomes, and passes a large bounded-output regression. |
| D4 | OpenAI exposes one explicit prompt-cache breakpoint annotation; historical/write roles, ordering gates, and invented candidate counts are removed with a documented migration. |
| D5 | OpenAI undocumented request minima/text rules and resource-ID grammar are removed, while opaque path encoding and stable security/resource bounds have focused negative tests. |
| D6 | Conversations item retrieve/delete, open reasoning mode, and the selected hosted-tool typed variants pass provider-owned offline fixtures. |
| D7 | Anthropic compaction and unknown stops settle conservatively; compaction deltas/replay, current beta headers, and caller-first task budgets pass direct/stream fixtures. |
| D8 | Anthropic Files-in-Messages and hosted-tool inspection/replay are exact-scope, bounded, and never become caller-owned executable tool calls. |
| D9 | Anthropic Message Batches aggregate current feature requirements and decode bounded unordered JSONL; Skills declared lifecycle operations and upload constraints pass offline fixtures. |
| D10 | Every audited OpenAI/Anthropic exact-model or product-policy branch is retained with current official stable evidence and focused baseline coverage, or deleted. |
| D11 | Future model IDs with representative explicit typed options reach final wire or fail only for a documented stable structural rule. |
| D12 | OpenAI and Anthropic public examples compile and demonstrate both provider-owned and portable paths. |
| D13 | Support, ADR, and migration documentation uses `claimed slice complete`, `provider platform complete`, and `intentionally deferred`, records superseding prompt-cache reasoning, and lists the deferred work explicitly. |
| D14 | Focused and full deterministic release gates pass serially, including workspace tests, Clippy, docs, doctests, MSRV, architecture, feature, metadata, package, formatting, and diff checks as applicable. |
| D15 | Any live diagnostic is separately categorized, secret-free, non-gating, and leaves no maintained temporary harness or credential artifact. |
| D16 | Simplification and independent code review have been performed on the final diff; eligible findings are fixed or explicitly surfaced. |
| D17 | Changes are committed in reviewable English Conventional Commit units, with no push, tag, publish, release, or pull request. |
