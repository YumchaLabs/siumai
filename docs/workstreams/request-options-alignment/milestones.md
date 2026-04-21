# Request Options Alignment - Milestones

Last updated: 2026-04-21

## M1 - Shared contract audit

Status: done

- audit `repo-ref/ai/packages/ai/src/prompt/request-options.ts`
- separate shared contract gaps from existing helper/runtime behavior

## M2 - Shared type ownership fix

Status: done

- move `CancelHandle` ownership to `siumai-spec`
- update runtime stream internals to use the shared handle

## M3 - Shared request-facing surface

Status: done

- add `TimeoutConfiguration`
- add `TimeoutConfigurationSettings`
- add `RequestOptions`
- expose them on the stable Rust facade

## M4 - Helper/runtime adoption

Status: pending

- decide which helper option structs should expose `with_request_options(...)`
- wire supported fields consistently across text/completion/image/speech/transcription/rerank
- keep unsupported lanes explicit instead of silently dropping them
