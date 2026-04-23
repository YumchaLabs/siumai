# Upload Skill Provider Wrapper Removal Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Contract audit

- [x] Confirm the stable skill-upload contract is already shared through `SkillUploadRequest` /
  `SkillUploadResult`.
- [x] Verify the remaining provider-local `OpenAiSkill*` / `AnthropicSkill*` types are wrappers,
  not distinct runtime contracts.

## Track B - Provider resource simplification

- [x] Make `OpenAiSkills` consume shared `SkillUploadRequest` and return shared
  `SkillUploadResult`.
- [x] Make `AnthropicSkills` consume shared `SkillUploadRequest` and return shared
  `SkillUploadResult`.
- [x] Remove provider-local skill-upload wrapper type exports from the provider modules and public
  façade re-exports.

## Track C - Behavioral parity preservation

- [x] Preserve OpenAI `displayTitle` unsupported warnings.
- [x] Preserve Anthropic latest-version metadata fetch behavior.
- [x] Preserve existing provider-root `providerReference` / `providerMetadata` output shaping.

## Track D - Docs and verification

- [x] Add a dedicated workstream describing the wrapper-removal slice.
- [x] Update `CHANGELOG.md` `Unreleased`.
- [x] Update the structural-alignment matrix/todo/milestones notes for the skill-upload row.
- [x] Run focused `nextest` coverage for the affected public/helper paths.
- [x] Run `cargo check --workspace`.
