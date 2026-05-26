# ElevenLabs Pronunciation Dictionary Mutations — TODO

Status: Active
Last updated: 2026-05-26

## EPDM-010 — Scope And Endpoint Contract Freeze

- [x] EPDM-010 [owner=planner] [deps=none] [scope=docs/workstreams/elevenlabs-pronunciation-dictionary-mutations,official-docs]
  Goal: Freeze the endpoint inventory, first mutation slice, non-goals, and validation gates.
  Validation: DESIGN.md, TODO.md, MILESTONES.md, EVIDENCE_AND_GATES.md, WORKSTREAM.json, and HANDOFF.md exist and agree.
  Review: planner self-review for mutation/download boundaries.
  Evidence: `docs/workstreams/elevenlabs-pronunciation-dictionary-mutations/EVIDENCE_AND_GATES.md`
  Handoff: DONE. First executable slice is JSON create-from-rules. Create-from-file, metadata update,
  rule mutation, and download-by-version are follow-on tasks inside this workstream unless they grow
  beyond focused no-network validation.

## EPDM-020 — Create From Rules

- [x] EPDM-020 [owner=worker] [deps=EPDM-010] [scope=siumai-provider-elevenlabs,siumai,siumai/tests]
  Goal: Add typed `create_from_rules` support to `ElevenLabsPronunciationDictionaries` for
  `POST /v1/pronunciation-dictionaries/add-from-rules`, including alias/phoneme rule request types,
  create response mapping, configured auth/header/base URL reuse, request header merge, and facade
  exports.
  Validation: `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs pronunciation --no-fail-fast`; `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast`; `cargo fmt --check -p siumai-provider-elevenlabs -p siumai`.
  Review: review-workstream for request shape, response mapping, rule type naming, and locator fit.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: DONE. Implemented `create_from_rules` with typed request/response structs, shared JSON
  POST resource wiring, alias/phoneme rule request builders, no-network provider coverage for
  auth/header/body/response behavior, and facade public-surface exports. Next executable task is
  EPDM-030 create-from-file multipart support.

## EPDM-030 — Create From File

- [ ] EPDM-030 [owner=worker] [deps=EPDM-020] [scope=siumai-provider-elevenlabs,siumai,siumai/tests]
  Goal: Add typed multipart `create_from_file` support for
  `POST /v1/pronunciation-dictionaries/add-from-file`.
  Validation: focused provider nextest filter for multipart dictionary creation; facade compile test if public types are added; `cargo fmt --check`.
  Review: review-workstream for filename/MIME behavior, content-length/body capture, and request
  header merge.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Keep PLS parsing out of scope; send caller-provided bytes and metadata.

## EPDM-040 — Metadata Update

- [ ] EPDM-040 [owner=worker] [deps=EPDM-020] [scope=siumai-provider-elevenlabs,siumai,siumai/tests]
  Goal: Add `PATCH /v1/pronunciation-dictionaries/{id}` metadata update for `archived` and/or
  `name`.
  Validation: focused provider/facade nextest filter for update path encoding, JSON body, and
  metadata response mapping.
  Review: review-workstream for partial update semantics and empty-body validation.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Reject empty update requests locally.

## EPDM-050 — Rule Mutation

- [ ] EPDM-050 [owner=worker/planner] [deps=EPDM-020] [scope=siumai-provider-elevenlabs,siumai,siumai/tests,docs/workstreams/elevenlabs-pronunciation-dictionary-mutations]
  Goal: Decide whether add/remove/set rules stay in this lane, then implement the smallest accepted
  slice using shared rule structs.
  Validation: focused provider/facade nextest filter for accepted rule mutation endpoints.
  Review: review-workstream for version semantics and whether add/set/remove should split.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Add/set use alias/phoneme rule arrays; remove uses `rule_strings`.

## EPDM-060 — Download By Version Decision

- [ ] EPDM-060 [owner=planner/worker] [deps=EPDM-020] [scope=docs/workstreams/elevenlabs-pronunciation-dictionary-mutations,siumai-provider-elevenlabs,siumai]
  Goal: Re-audit the official download-by-version docs and decide whether to implement binary PLS
  download in this lane or split it.
  Validation: official docs accessible or fallback source is recorded; if implemented, focused binary
  GET test proves path/query encoding and bytes mapping.
  Review: review-workstream for docs stability and binary response shape.
  Evidence: `EVIDENCE_AND_GATES.md`
  Handoff: Opening audit saw `download.mdx` listed in `llms.txt` but direct page fetch returned HTTP
  500, so do not implement from guesswork without a stable source.

## EPDM-070 — Closeout

- [ ] EPDM-070 [owner=planner] [deps=EPDM-020] [scope=docs/workstreams/elevenlabs-pronunciation-dictionary-mutations,CHANGELOG.md]
  Goal: Close the lane or split residual mutation/download gaps into narrow follow-ons.
  Validation: verify-rust-workstream records fresh final gate evidence.
  Review: review-workstream has no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`, `CHANGELOG.md`
  Handoff: Summarize shipped mutation behavior and deferred endpoints.
