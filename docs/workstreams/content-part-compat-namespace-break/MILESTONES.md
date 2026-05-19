# ContentPart Compatibility Namespace Break — Milestones

Status: Closed
Last updated: 2026-05-19

## M0 — Scope And Compatibility Contract

Status: Complete.

Exit criteria:

- A dedicated workstream exists for the ContentPart compatibility namespace break.
- ADR-0008 and the two prior ContentPart lanes are linked.
- The first executable slice is selected.

Primary evidence:

- `DESIGN.md`
- `TODO.md`
- `WORKSTREAM.json`

## M1 — Compatibility Namespace Established

Status: Complete.

Exit criteria:

- Legacy chat content carriers have an explicit canonical compatibility namespace.
- Public/migration docs teach that namespace for legacy `ContentPart`.
- Recommended stable examples use directional prompt/output parts.
- Any retained old-path aliases are explicitly deprecated or migration-only.

Primary gates:

- `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast`
- `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast`

## M2 — Response Adapter Deepened

Status: Complete with CPN-050 split.

Exit criteria:

- Response-side legacy `ContentPart` projection is physically adapter-owned.
- Generated-output projection preserves response `providerMetadata` and does not emit request
  `providerOptions`.
- A high-value response path is inspected. CPN-050 records why protocol/bridge parser migration is
  split instead of forced through a lossy generated-output detour.

Primary gates:

- `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast`
- Protocol all-features gates for OpenAI, Anthropic, and Gemini when response parser paths move.
- `cargo nextest run -p siumai-bridge --features openai,anthropic,google response --no-fail-fast`

## M3 — Migration And Closeout

Status: Complete.

Exit criteria:

- Migration notes include before/after imports.
- Architecture docs and source guards prevent regressions.
- Workstream evidence is fresh and complete.
- Any retained aliases or broad parser rewrites are split into explicit follow-ons.

Closeout evidence:

- `cargo fmt --check -p siumai-spec -p siumai-core -p siumai` — PASS.
- `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast` — PASS.
- `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast` — PASS.
- `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast` — PASS.
- `cargo nextest run -p siumai-bridge --features openai,anthropic,google response --no-fail-fast` — PASS.
- `cargo nextest run -p siumai --features openai,anthropic,google,xai,groq --test public_surface_imports_test --no-fail-fast` — PASS.
