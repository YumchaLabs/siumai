# Fearless Provider Composite Client Isolation - Design

Opened: 2026-05-16
Closed: 2026-05-16

## Problem

DeepInfra, Fireworks, and TogetherAI are hybrid providers. Their stable provider surface combines
shared OpenAI-compatible text/audio families with provider-owned image or rerank runtimes.

The current implementation already exposes native family factory methods, but each provider also
keeps a private composite `LlmClient` wrapper for historical method-style compatibility. That is a
reasonable migration bridge, but the old `*UnifiedClient` naming and weak source guards leave a
shallow interface in the wrong place:

- The private wrapper name reads like the target architecture instead of a compatibility adapter.
- Future factory edits could accidentally route stable family methods through the composite client.
- The compatibility audit still describes the wrappers as "keep temporarily" rather than
  "compat-only, guarded".

## Target State

Provider composite clients remain only as explicit compatibility adapters for
`compat_language_client_with_ctx(...)`.

Stable family methods must construct provider-owned or shared family models directly:

- `language_model_text_with_ctx(...)`
- `completion_model_family_with_ctx(...)`
- `embedding_model_family_with_ctx(...)`
- `image_model_family_with_ctx(...)`
- `speech_model_family_with_ctx(...)`
- `transcription_model_family_with_ctx(...)`
- `reranking_model_family_with_ctx(...)`

The source should make this role obvious by naming the private wrappers as compat composite
clients, and tests should fail if a stable family method reuses those wrappers or calls a compat
client path.

## Scope

- Rename the private DeepInfra, Fireworks, and TogetherAI composite clients to make the
  compatibility role explicit.
- Add source guards proving the composite clients are constructed only by the language compat path.
- Guard the stable family methods in those providers against composite-client construction,
  compat-client self-calls, and `LlmClient` capability downcasts.
- Update the compatibility audit, migration guide, changelog, and workstream notes.

## Non-Goals

- Do not remove the compatibility `LlmClient` wrappers in this slice.
- Do not rewrite the provider-owned image or rerank runtimes.
- Do not move hybrid provider wrappers into provider crates in this slice.
- Do not change public provider IDs or public user-facing capability behavior.

## Architecture Direction

The deep module here is the provider factory seam: stable family callers ask for a family model and
do not need to know whether the implementation uses an OpenAI-compatible runtime, a provider-owned
runtime, or both.

The composite `LlmClient` is a migration adapter, not the central execution path. Keeping it
private, compat-named, and source-guarded improves locality: compatibility behavior stays in one
place while family execution stays direct and testable.

## Closeout

This lane closes with the compatibility wrappers kept deliberately. The next deletion step should
be a separate compatibility-removal lane because historical method-style `Siumai` and generic
`LlmClient` callers still depend on composite capability views.
