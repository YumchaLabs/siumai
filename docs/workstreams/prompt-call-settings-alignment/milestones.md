# Prompt Call Settings Alignment - Milestones

Last updated: 2026-04-21

## Completed

- Audited the remaining `packages/ai/src/prompt/index.ts` compatibility exports after the earlier
  `LanguageModelCallOptions` and `RequestOptions` workstreams landed.
- Added deprecated shared `CallSettings` as a compatibility projection over
  `LanguageModelCallOptions` plus non-timeout `RequestOptions`.
- Added free timeout helper functions that mirror the AI SDK helper role on
  `TimeoutConfiguration`.
- Re-exported the new compatibility surface from the stable Rust facade and covered it with public
  compile guards plus focused shared-type unit tests.

## Next

- Start a separate prompt/message shared-contract workstream for `ModelMessage`, `Prompt`, and the
  prompt content-part naming layer.
