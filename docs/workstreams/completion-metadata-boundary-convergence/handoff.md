# Completion Metadata Boundary Convergence - Handoff

Last updated: 2026-05-17

## Current State

The lane is closed. Shared completion metadata helpers now live in
`siumai-protocol-openai`, and native OpenAI plus OpenAI-compatible completion clients call that
protocol seam.

## Next Task

No in-lane implementation task remains.

## Watch Points

- Completion `logprobs` are raw completion logprobs objects, not chat `logprobs.content` arrays.
- OpenAI-compatible completion currently preserves top-level `sources`; native OpenAI does not have
  a separate local path for sources, but the shared helper may safely preserve the field when
  present.
- Keep provider namespace selection at the caller so compat aliases and requested public metadata
  keys remain stable.

No follow-on was split from this lane.
