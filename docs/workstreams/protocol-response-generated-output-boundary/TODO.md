# Protocol Response Generated-Output Boundary — TODO

Status: Active
Last updated: 2026-05-19

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## M0 — Scope And Response-Lossiness Baseline

- [x] PRG-010 [owner=planner] [deps=none] [scope=docs/workstreams/protocol-response-generated-output-boundary]
  Goal: Open the workstream, record current parser hotspots, freeze non-goals, and choose the first
  executable slice.
  Validation: workstream docs exist; `WORKSTREAM.json` parses; `next_task` points to PRG-020.
  Evidence: `DESIGN.md`, `EVIDENCE_AND_GATES.md`, `JOURNAL/2026-05-19-prg-010.md`.
  Handoff: PRG-020 is the first code task; do not start broad parser rewrites before its guard is
  in place.

## M1 — OpenAI Responses Proof Slice

- [x] PRG-020 [owner=codex] [deps=PRG-010] [scope=siumai-protocol-openai/src/standards/openai/transformers/response]
  Goal: Extract OpenAI Responses response-owned legacy `ContentPart` construction behind a named
  response compatibility adapter module without changing serialized output.
  Validation:
  `cargo fmt --check -p siumai-protocol-openai`;
  `cargo nextest run -p siumai-protocol-openai --no-default-features --features openai-standard,openai-responses responses_response_transformer_source_does_not_emit_request_provider_options --no-fail-fast`;
  plus targeted OpenAI Responses transformer fixture tests touched by the extraction.
  Review: completed; no blocking findings from task-local self-review. A formal
  `review-workstream` pass can be run before accepting the lane, but PRG-020's own gates pass.
  Evidence: recorded in `EVIDENCE_AND_GATES.md`; source guards prove broad parser code delegates
  legacy construction to the adapter, and adapter-owned defaults keep request provider options
  empty.
  Handoff: PRG-030 should classify OpenAI Responses output shapes by generated-output projection
  lossiness; do not force ambiguous hosted tools/files/approval parts through
  `GenerateTextContentPart`.

- [x] PRG-030 [owner=codex] [deps=PRG-020] [scope=siumai-protocol-openai/src/standards/openai/transformers/response,siumai-spec/tests]
  Goal: Classify OpenAI Responses output shapes by generated-output projection lossiness and add
  tests/docs for the shapes that must remain legacy compatibility payloads.
  Validation:
  `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast`;
  targeted OpenAI Responses fixture tests.
  Review: completed; source guard verifies production OpenAI Responses parsing does not directly
  call generated-output projection helpers, and behavior tests cover both lossless and lossy
  representative shapes.
  Evidence: `EVIDENCE_AND_GATES.md` lossiness matrix and fresh PRG-030 gates.
  Handoff: Only lossless subsets may call spec-owned response projection helpers. PRG-040 should
  apply or deliberately adapt the response-adapter pattern for Anthropic response parsing.

## M2 — Second Provider Proof

- [x] PRG-040 [owner=codex] [deps=PRG-020] [scope=siumai-protocol-anthropic/src/standards/anthropic/utils/parse.rs]
  Goal: Apply the response-adapter pattern to Anthropic response parsing or document why Anthropic
  citations/tool-use require a different adapter shape.
  Validation:
  `cargo fmt --check -p siumai-protocol-anthropic`;
  `cargo nextest run -p siumai-protocol-anthropic --no-default-features --features anthropic-standard anthropic_parse_response_content_source_does_not_emit_request_provider_options --no-fail-fast`.
  Review: completed; parser-local `response_content` adapter owns legacy response constructors,
  source guards cover delegation, request-provider-options hygiene, and no forced generated-output
  projection.
  Evidence: `EVIDENCE_AND_GATES.md`; targeted Anthropic parse tests preserve text citations,
  document sources, web-search sources, MCP/server tool metadata, and provider-executed tool parts.
  Handoff: PRG-050 should evaluate Gemini response parsing against the same named-adapter/source
  guard pattern; a documented no-op is acceptable if Gemini is already sufficiently guarded.

- [x] PRG-050 [owner=codex] [deps=PRG-020] [scope=siumai-protocol-gemini/src/standards/gemini/transformers/response.rs]
  Goal: Evaluate Gemini response parsing against the adapter pattern and either migrate a narrow
  helper or record it as already sufficiently guarded.
  Validation:
  `cargo fmt --check -p siumai-protocol-gemini`;
  `cargo nextest run -p siumai-protocol-gemini --no-default-features --features google gemini_response_content_source_does_not_emit_request_provider_options --no-fail-fast`.
  Review: completed; Gemini was not a no-op, so response-side legacy constructors moved behind a
  parser-local `response_content` adapter and source guards now cover delegation, request-option
  hygiene, and no forced generated-output projection.
  Evidence: `EVIDENCE_AND_GATES.md`; targeted Gemini response tests preserve grounding, URL
  context, safety/logprobs metadata, thought-signature metadata, reasoning files, media, and
  tool-call finish semantics.
  Handoff: PRG-060 should decide bridge response/stream ownership after all three protocol parser
  proofs are complete.

## M3 — Bridge, Stream, And Provider-Owned Response Paths

- [ ] PRG-060 [owner=unassigned] [deps=PRG-020,PRG-040] [scope=siumai-bridge/src/response,siumai-bridge/src/stream]
  Goal: Decide whether bridge response/stream paths should use protocol response adapters, keep
  primitive-only serialization, or split a narrower bridge follow-on.
  Validation:
  `cargo nextest run -p siumai-bridge --features openai,anthropic,google response --no-fail-fast`.
  Review: bridge code must not become the canonical owner of provider response semantics.
  Evidence: bridge decision note in `EVIDENCE_AND_GATES.md`.
  Handoff: Prefer split follow-ons for gateway/proxy JSON encoders if the scope grows.

- [ ] PRG-070 [owner=unassigned] [deps=PRG-040,PRG-050] [scope=siumai-provider-gemini,siumai-provider-amazon-bedrock]
  Goal: Audit provider-owned response parsers that are not pure protocol modules and decide whether
  they should adopt local response adapters or remain provider-owned exceptions.
  Validation: provider-specific nextest gates chosen from the touched crate and feature.
  Review: avoid mixed request/response file rewrites unless the changed function is narrowly
  isolated.
  Evidence: provider-owned parser audit table in `EVIDENCE_AND_GATES.md`.
  Handoff: Bedrock is a mixed file; split before broad edits.

## M4 — Integration, Docs, And Closeout

- [ ] PRG-080 [owner=planner] [deps=PRG-020,PRG-040,PRG-050] [scope=docs/architecture,docs/migration,docs/workstreams]
  Goal: Update public architecture/migration docs so response parser adapters, legacy
  compatibility payloads, and generated-output projection are clearly distinguished.
  Validation: docs grep/source guard evidence plus touched crate tests.
  Review: migration docs must not teach legacy `ContentPart` as canonical.
  Evidence: `EVIDENCE_AND_GATES.md`.
  Handoff: Split an ADR if a new public generated-output response model is proposed.

- [ ] PRG-090 [owner=planner] [deps=PRG-080] [scope=docs/workstreams/protocol-response-generated-output-boundary]
  Goal: Close this lane or split remaining parser-wide generated-output migration into narrower
  follow-ons.
  Validation: `verify-rust-workstream` records fresh final gate evidence.
  Review: no blocking findings from `review-workstream`.
  Evidence: `WORKSTREAM.json`, `HANDOFF.md`, `EVIDENCE_AND_GATES.md`.
  Handoff: Final status must name retained compatibility paths and their removal/narrowing criteria.
