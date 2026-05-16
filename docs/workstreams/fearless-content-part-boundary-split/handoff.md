# Fearless ContentPart Boundary Split - Handoff

Last updated: 2026-05-16

## Current State

The workstream is open. No production code has changed yet.

The first executable task is `CPB-020`: refresh the direct `ContentPart` construction scan, compare
it with the existing spec/core convergence audit, and tighten source guards so new production hits
must be classified or moved to directional adapters.

## Continuation Notes

- Prefer adapter-first migrations over introducing a new broad public enum too early.
- Keep legacy `ContentPart` available until migration docs and fixture parity prove replacement
  paths.
- Run focused tests before updating milestone status.
