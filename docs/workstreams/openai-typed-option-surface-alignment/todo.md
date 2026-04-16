# OpenAI Typed Option Surface Alignment - Todo

Last updated: 2026-04-11

## Done

- [x] Compare native OpenAI public typed exports with `repo-ref/ai/packages/openai/src/index.ts`
- [x] Add AI SDK-style OpenAI typed option names to the provider-owned module
- [x] Preserve upstream deprecated migration aliases where they still exist
- [x] Re-export the new names from `provider_ext::openai`
- [x] Add compile guards on the stable public facade
- [x] Ensure speech/transcription/file typed options are not name-only shims
- [x] Update changelog entries under `Unreleased`

## Open

- [ ] Run the same typed-option audit against Google Vertex with the same "real semantics first"
      rule
- [ ] Decide whether OpenAI builder/config typed default helpers should gain AI SDK-style wrapper
      overloads or stay Rust-first
