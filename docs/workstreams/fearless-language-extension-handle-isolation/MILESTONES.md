# Fearless Language Extension Handle Isolation - Milestones

Last updated: 2026-05-16

## M1 - Framed

- Workstream docs exist and are linked from the docs index.
- The lane has a bounded target: isolate file, skill, and music extension downcasts behind registry
  adapters.

## M2 - Adapters Exist

- Provider factories expose explicit extension construction methods.
- Default extension methods use registry-owned compatibility adapters.
- The adapter names make the compatibility role visible.

## M3 - Handle Is Clean

- `LanguageModelHandle` extension implementations no longer call `compat_language_client_with_ctx`.
- `LanguageModelHandle` extension implementations no longer call `as_file_management_capability`,
  `as_skills_capability`, or `as_music_generation_capability`.
- Existing file, skill, and music behavior stays source-compatible.

## M4 - Guarded And Closed

- Source guards lock the new seam.
- Compatibility audit documents the new state.
- Focused gates pass.
