# Fearless Language Extension Handle Isolation - TODO

Last updated: 2026-05-16

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[d]` deferred with rationale

## LEH-010 - Workstream Framing

- [x] Create design, TODO, milestones, evidence, handoff, and machine-readable workstream docs.
- [x] Link the workstream from `docs/README.md`.

Validation:

- Documentation review and `git diff --check`.

## LEH-020 - Registry Extension Adapters

- [ ] Add registry-owned adapters for client-backed file, skill, and music extension traits.
- [ ] Add explicit `ProviderFactory` extension methods with compatibility defaults.

Validation:

- `cargo check -p siumai-registry --no-default-features`

## LEH-030 - Handle Routing

- [ ] Route `LanguageModelHandle` file, skill, and music implementations through factory extension methods.
- [ ] Remove handle-local compatibility-client construction for extension methods.

Validation:

- `cargo nextest run -p siumai-registry registry::entry::file_tests registry::entry::skills_tests registry::entry::music_tests --no-default-features --no-fail-fast`

## LEH-040 - Guards And Audit

- [ ] Add or tighten source guards proving language-handle extension methods do not downcast.
- [ ] Update compatibility audit notes from "keep temporarily" to "isolated behind adapters".

Validation:

- `cargo nextest run -p siumai-registry registry::entry::boundary_tests --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --features openai,google --no-default-features --no-fail-fast`

## LEH-050 - Closeout

- [ ] Run final focused gates.
- [ ] Update evidence, milestones, handoff, and journal notes.
- [ ] Close or split any residual follow-up.
