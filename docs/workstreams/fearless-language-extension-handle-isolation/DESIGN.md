# Fearless Language Extension Handle Isolation - Design

Opened: 2026-05-16

## Problem

`LanguageModelHandle` is now the stable registry language-family handle, but its extension-only
file, skill, and music implementations still build a compatibility `LlmClient` and downcast inside
the handle implementation.

That keeps a shallow interface in the wrong place:

- The handle must know both family execution and compatibility downcast details.
- Source guards can only say stable chat execution is clean, while extension methods remain
  coupled to `LlmClient`.
- Future extension handles would need to rediscover the same downcast pattern.

## Target State

`LanguageModelHandle` should call explicit provider-factory extension methods:

- `file_management_capability_with_ctx(...)`
- `skills_capability_with_ctx(...)`
- `music_generation_capability_with_ctx(...)`

The default implementation of those methods may still adapt a legacy generic client, but that
compatibility adapter belongs behind the registry factory seam, not inside the handle.

## Scope

- Add registry-owned client-backed adapters for file, skill, and music extension traits.
- Add default provider-factory extension methods that isolate legacy `LlmClient` downcasts.
- Route `LanguageModelHandle` extension implementations through those methods.
- Add source guards proving the handle no longer calls `compat_language_client_with_ctx(...)` or
  `as_*_capability()` in extension implementations.
- Update architecture audit notes.

## Non-Goals

- Do not make file, skill, or music first-class stable model families in this slice.
- Do not remove `LlmClient` extension downcasts from the explicit compatibility facade.
- Do not change provider-owned extension trait names.

## Architecture Direction

This is a depth improvement: the handle interface stays small, while adapter complexity moves
behind the provider-factory seam. The default adapter is compatibility-only, and providers can later
override these extension methods with native extension objects without changing handle code.
