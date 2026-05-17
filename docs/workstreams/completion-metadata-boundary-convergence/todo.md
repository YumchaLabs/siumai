# Completion Metadata Boundary Convergence - Todo

Last updated: 2026-05-17

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Tasks

- [x] CMBC-010 - Open the workstream and document the target boundary.
  - Scope: `docs/workstreams/completion-metadata-boundary-convergence/*`
  - Validation: workstream docs describe problem, target state, scope, gates, and handoff state.

- [x] CMBC-020 - Extract shared completion metadata helpers into the OpenAI protocol layer.
  - Scope: `siumai-protocol-openai/src/standards/openai/*`
  - Validation: protocol-level unit tests cover logprobs, sources, namespacing, merge behavior, and
    empty metadata elision.

- [x] CMBC-030 - Replace native OpenAI completion local helpers with the shared seam.
  - Scope: `siumai-provider-openai/src/providers/openai/client/completion.rs`
  - Validation: native completion response and stream tests still preserve `providerMetadata.openai.logprobs`.

- [x] CMBC-040 - Replace OpenAI-compatible completion local helpers with the shared seam.
  - Scope: `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client.rs`
  - Validation: compatible completion response and stream tests preserve provider-keyed `logprobs`
    and `sources`.

- [x] CMBC-050 - Run gates and close the lane.
  - Scope: formatting, nextest, workstream evidence, conventional commit.
  - Validation:
    - `cargo fmt -p siumai-protocol-openai -p siumai-provider-openai -p siumai-provider-openai-compatible`
    - `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`
    - `cargo nextest run -p siumai-provider-openai --features openai --no-fail-fast`
    - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
