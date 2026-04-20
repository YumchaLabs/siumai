# OpenAI-Compatible Package Surface Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Public type parity

- [x] Audit `repo-ref/ai/packages/openai-compatible/src/index.ts`.
- [x] Add the missing generic typed option structs for chat, completion, and embedding on the
  provider-owned/public compat facade.
- [x] Add the missing generic `OpenAiCompatible{Chat,Completion,Embedding}ModelId` aliases on the
  provider-owned/public compat facade.
- [x] Add the missing generic `OpenAiCompatibleImageModelId` alias on the provider-owned/public
  compat facade.
- [x] Add the missing generic `OpenAiCompatibleErrorData` and `ProviderErrorStructure<T>` names on
  the provider-owned/public compat facade.
- [x] Re-export the AI SDK-style `MetadataExtractor` alias on the provider-owned/public compat
  facade.
- [x] Add exact-case `OpenAICompatible*` compatibility aliases for the main audited package names
  while keeping the existing Rust-style `OpenAiCompatible*` names available.

## Track B - Shared helper parity

- [x] Add typed generic request helpers for chat, completion, and embedding requests under
  `providerOptions.openaiCompatible`.
- [x] Make those helpers merge onto existing compat provider-option objects instead of overwriting
  sibling raw fields.
- [x] Centralize the helper merge logic so chat/completion/embedding cannot silently drift in merge
  behavior.

## Track C - Docs and changelog

- [x] Create a dedicated `docs/workstreams/openai-compatible-package-surface-alignment/` folder.
- [x] Record the shared compat package-surface alignment slice in `CHANGELOG.md` `Unreleased`.
- [x] Fold future shared compat package audits into this workstream instead of scattering them
  across generic refactor notes.

## Track D - Intentional deferrals

- [-] Do not fabricate TypeScript-only `createOpenAICompatible`, `OpenAICompatibleProvider`, or
  `OpenAICompatibleProviderSettings` exports on the Rust facade.
- [-] Do not invent a Rust `OpenAiCompatibleImageModel` wrapper type just to mirror the
  TypeScript callable class when the runtime capability already exists through the stable image
  client/registry surface.
