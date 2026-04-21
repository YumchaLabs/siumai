# Language Model Call Options Alignment - Milestones

Last updated: 2026-04-21

## M1 - Audit and scope

Status: done

- audit `repo-ref/ai/packages/ai/src/prompt/language-model-call-options.ts`
- confirm that this slice is model-facing only and separate from request transport controls

## M2 - Shared projection

Status: done

- add `LanguageModelCallOptions`
- add `LanguageModelReasoning`
- expose the projection on the stable Rust facade

## M3 - Underlying local fixes

Status: done

- add `CommonParamsBuilder::max_completion_tokens(...)`
- include `max_completion_tokens` in `CommonParams::cache_hash()`
- lock behavior with unit tests

## M4 - Follow-up request controls

Status: deferred

- design `RequestOptions`
- design `TimeoutConfiguration`
- decide whether cancellation belongs on stable request structs or stays in runtime helper APIs
