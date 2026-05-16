# Fearless Language Extension Handle Isolation - Handoff

Last updated: 2026-05-16

## Current State

The workstream is open. `LEH-010` is complete.

`LanguageModelHandle` still owns extension-only compatibility downcasts for file, skill, and music.
The next task is `LEH-020`: add registry-owned adapters and explicit provider-factory extension
methods.

## Continuation Notes

- Keep file, skill, and music as extension surfaces in this lane.
- Do not remove compatibility facade downcasts from `Siumai`; this lane only cleans the registry
  language-handle implementation.
- The expected implementation shape is adapter isolation, not a new stable model family.
