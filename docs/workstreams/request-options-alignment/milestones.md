# Request Options Alignment - Milestones

Last updated: 2026-04-24

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

Status: done

- add `request_options: Option<RequestOptions>` to the stable facade helper option structs
- wire supported fields consistently across text/completion/embedding/image/video/speech/transcription/rerank
- map `maxRetries` to policy retry attempts with the AI SDK default when `request_options` is used
- merge request headers and total timeout into helper-owned request `HttpConfig`
- honor `abortSignal` for non-streaming calls, stream setup/consumption, stream handles, and video
  polling
- keep unsupported lanes explicit instead of silently dropping them

## M5 - Remaining runtime lanes

Status: pending

- enforce `chunkMs` in the shared stream lane
- enforce `stepMs` once the facade owns an AI SDK-style multi-step loop
- enforce `toolMs` / per-tool timeouts once generic tool execution scheduling is centralized
