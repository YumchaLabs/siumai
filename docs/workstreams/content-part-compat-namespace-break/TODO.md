# ContentPart Compatibility Namespace Break — TODO

Status: Closed
Last updated: 2026-05-19

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## M0 — Scope And Compatibility Contract

- [x] CPN-010 [owner=planner] [deps=none] [scope=docs/workstreams/content-part-compat-namespace-break]
  Goal: Open the breaking-slice workstream, freeze target state, and choose the first executable
  slice.
  Validation: workstream docs exist; `WORKSTREAM.json` parses; active task points to CPN-020.
  Review: planner self-review; no code changes in this task.
  Evidence: pending in `EVIDENCE_AND_GATES.md`.
  Handoff: CPN-020 is the first executable code task.

## M1 — Canonical Compatibility Namespace

- [x] CPN-020 [owner=codex] [deps=CPN-010] [scope=siumai-spec,siumai-core,siumai,docs/architecture,docs/migration]
  Goal: Add the canonical explicit compatibility namespace for legacy chat content carriers and
  document it as the only recommended import path for `ContentPart`.
  Validation:
  `cargo fmt --check -p siumai-spec -p siumai-core -p siumai`;
  `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast`;
  `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast`;
  `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast`.
  Review: completed; no blocking findings. The only follow-up is the intentionally scoped CPN-030
  public-prelude removal/retention decision.
  Evidence: recorded in `EVIDENCE_AND_GATES.md`.
  Handoff: old-path aliases stay available for this slice; CPN-030 owns removing legacy
  `ContentPart` from the recommended stable unified prelude or proving it remains migration-only.

- [x] CPN-030 [owner=codex] [deps=CPN-020] [scope=siumai/src/lib.rs,siumai/tests,docs/architecture]
  Goal: Remove legacy `ContentPart` from recommended `prelude::unified` exports or prove it remains
  only as a deprecated migration alias outside the stable recommended prelude.
  Validation:
  `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast`;
  `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast`.
  Review: completed; no blocking findings.
  Evidence: recorded in `EVIDENCE_AND_GATES.md`; public surface tests prove the final import rule.
  Handoff: Stable examples should compile without `ContentPart`; migration-only examples should use
  the explicit compatibility namespace.

## M2 — Response Adapter Deepening

- [x] CPN-040 [owner=codex] [deps=CPN-020] [scope=siumai-spec/src/types/ai_sdk,siumai-spec/tests,siumai/src/text.rs,siumai/tests]
  Goal: Extract response-side legacy `ContentPart` -> generated-output projection into a named
  response compatibility adapter module without changing behavior.
  Validation:
  `cargo fmt --check -p siumai-spec -p siumai`;
  `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast`;
  `cargo nextest run -p siumai --features openai,anthropic,google,xai,groq --test public_surface_imports_test --no-fail-fast`;
  targeted generate-text/OpenAI response fixture gates if `siumai/src/text.rs` behavior changes.
  Review: completed; no blocking findings.
  Evidence: recorded in `EVIDENCE_AND_GATES.md`; source guards prove the response compatibility
  adapter owns legacy mapping and does not emit request `providerOptions`.
  Handoff: If this exposes protocol-specific parser seams, split provider/parser rewrites into
  follow-on tasks rather than broadening CPN-040.

- [-] CPN-050 [owner=codex] [deps=CPN-040] [scope=siumai-protocol-openai,siumai-protocol-anthropic,siumai-protocol-gemini,siumai-bridge]
  Goal: Select and migrate one high-value protocol response parser or bridge response path to the
  response adapter seam, proving the adapter is usable outside facade/spec projection.
  Validation:
  `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`;
  `cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast`;
  `cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast`;
  `cargo nextest run -p siumai-bridge --features openai,anthropic,google response --no-fail-fast`.
  Review: completed; intentionally deferred/split after code inspection.
  Evidence: recorded in `EVIDENCE_AND_GATES.md`.
  Handoff: no narrow protocol/bridge response path can adopt the generated-output adapter without
  broad parser/encoder reshaping. Keep this as a follow-on after protocol response models grow an
  output-part lane; do not rewrite all protocols in this workstream.

## M3 — Migration And Closeout

- [x] CPN-060 [owner=codex] [deps=CPN-020,CPN-030,CPN-040] [scope=docs/architecture,docs/migration,siumai/tests,siumai-spec/tests]
  Goal: Update migration and architecture docs, plus source guards, so new work cannot present
  legacy `ContentPart` as canonical.
  Validation:
  `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast`;
  `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast`;
  docs grep/source guard evidence recorded in `EVIDENCE_AND_GATES.md`.
  Review: completed; no blocking findings.
  Evidence: recorded in `EVIDENCE_AND_GATES.md`; docs show before/after import examples,
  replacement directional types, and named response adapter behavior.
  Handoff: CPN-070 closeout is next.

- [x] CPN-070 [owner=planner] [deps=CPN-020,CPN-030,CPN-040,CPN-050,CPN-060] [scope=docs/workstreams/content-part-compat-namespace-break]
  Goal: Close the lane or split any remaining parser-wide rewrites into narrower follow-ons.
  Validation: `verify-rust-workstream` records fresh final gate evidence.
  Review: completed; no blocking findings. Protocol/parser generated-output migration is split as
  a future lane.
  Evidence: recorded in `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`,
  `MILESTONES.md`, and `JOURNAL/2026-05-19-cpn-070.md`.
  Handoff: shipped namespace break, retained migration aliases, named response adapter, and
  parser follow-on split are summarized.
