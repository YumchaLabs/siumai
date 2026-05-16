# Fearless Provider Composite Client Isolation - TODO

Last updated: 2026-05-16

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[d]` deferred with rationale

## PCI-010 - Workstream Framing

- [x] Create design, TODO, milestones, evidence, handoff, and machine-readable workstream docs.
- [x] Link the workstream from `docs/README.md`.

Validation:

- Documentation review and `git diff --check`.

## PCI-020 - Compat Composite Naming

- [x] Rename DeepInfra, Fireworks, and TogetherAI private composite clients so their compatibility
  role is visible in source and debug output.
- [x] Keep public provider IDs and behavior unchanged.

Validation:

- `cargo check -p siumai-registry --features openai,togetherai,deepinfra --no-default-features`

## PCI-030 - Source Guards

- [x] Add architecture guards proving composite clients are only constructed in
  `compat_language_client_with_ctx(...)`.
- [x] Guard native family methods against composite-client construction, compat-client self-calls,
  and `LlmClient` capability downcasts.

Validation:

- `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --features openai,togetherai,deepinfra --no-default-features --no-fail-fast`

## PCI-040 - Audit And Migration Notes

- [x] Update the compatibility audit from "keep temporarily" to "compat-only isolated".
- [x] Add a migration/changelog note explaining that hybrid provider composite clients are internal
  compatibility adapters and new factory code should use native family methods.

Validation:

- `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --features openai,togetherai,deepinfra --no-default-features --no-fail-fast`

## PCI-050 - Closeout

- [x] Run final focused gates.
- [x] Update evidence, milestones, handoff, and journal notes.
- [x] Decide whether the residual compatibility wrapper deletion should be a future workstream.
