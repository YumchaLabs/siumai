# Upload Skill Provider Wrapper Removal Alignment - Design

Last updated: 2026-04-21

## Context

The earlier skill-upload slice already closed the main high-level AI SDK gap:

- `siumai::skills::upload(...)` existed as the public helper
- shared `SkillUploadRequest` / `SkillUploadResult` already modeled the stable contract
- OpenAI and Anthropic `skills()` resources already matched the audited runtime semantics

However, one provider-surface drift remained underneath that helper:

- `OpenAiSkills` still exposed provider-local `OpenAiSkillFile`, `OpenAiSkillFileContent`, and
  `OpenAiSkillUploadResult`
- `AnthropicSkills` still exposed the parallel `AnthropicSkill*` wrapper family
- both resources still centered a bespoke `upload(...)` method instead of one shared request
  object

Those wrappers did not exist in the upstream AI SDK package boundary. They were mostly a legacy
Rust compatibility shell around a contract that was already shared elsewhere in the workspace.

## Goal

- Remove provider-local skill-upload wrapper types that no longer encode a meaningful boundary.
- Make provider-owned `skills()` resources consume the shared `SkillUploadRequest` contract
  directly.
- Keep OpenAI and Anthropic runtime semantics unchanged while simplifying the public surface.

## Non-goals

- Do not change the high-level `siumai::skills::upload(...)` helper contract.
- Do not remove OpenAI `displayTitle -> unsupported` warning behavior.
- Do not remove Anthropic latest-version metadata fetching.
- Do not widen provider coverage beyond the currently audited OpenAI and Anthropic skill routes.

## Chosen design

### 1. Reuse the shared skill-upload request/result types end-to-end

`OpenAiSkills` and `AnthropicSkills` now accept `SkillUploadRequest` directly and return the shared
`SkillUploadResult`.

This keeps the provider resource layer structurally aligned with the rest of the workspace:

- shared file carrier: `SkillUploadFile`
- shared content carrier: `SkillFileContent`
- shared provider result: `SkillUploadResult`

### 2. Delete provider-local duplicate wrapper types

The following public compatibility shells are removed:

- `OpenAiSkillFile`
- `OpenAiSkillFileContent`
- `OpenAiSkillProviderMetadata`
- `OpenAiSkillUploadResult`
- `AnthropicSkillFile`
- `AnthropicSkillFileContent`
- `AnthropicSkillProviderMetadata`
- `AnthropicSkillUploadResult`

This makes the public Rust surface more honest relative to the upstream AI SDK: the stable shape is
shared, while provider-owned resources remain responsible only for provider-specific execution.

### 3. Keep provider-owned semantics, not provider-owned wrappers

The refactor does not flatten away real provider differences:

- OpenAI still emits the audited `unsupported { feature: "displayTitle" }` warning
- Anthropic still posts multipart uploads, injects the `skills-2025-10-02` beta header, and
  fetches latest-version metadata from `GET /v1/skills/{skill_id}/versions/{version}`
- both providers still expose their own `providerReference` and `providerMetadata` shaping

The only thing removed is the redundant wrapper layer around that behavior.

## Follow-up

If other non-chat provider resources still expose provider-local request/result wrappers where the
stable contract is already shared and audited, the same cleanup pattern should be applied there as
well instead of preserving drift for historical convenience.
