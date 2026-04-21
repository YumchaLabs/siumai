# Request Options Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Shared request-facing types

- [x] Audit `repo-ref/ai/packages/ai/src/prompt/request-options.ts`.
- [x] Add shared `TimeoutConfiguration`.
- [x] Add shared `TimeoutConfigurationSettings`.
- [x] Add shared `RequestOptions`.
- [x] Re-export the new shared request-facing types on the stable Rust facade.

## Track B - Cancellation ownership

- [x] Move `CancelHandle` into `siumai-spec` so shared request types can own it honestly.
- [x] Update runtime stream internals to use the shared cancel handle type.

## Track C - Runtime/helper adoption

- [ ] Decide which facade helper option structs should accept `RequestOptions`.
- [ ] Decide how much of `max_retries` should map onto simple `RetryOptions` defaults.
- [ ] Add explicit runtime handling or explicit documented deferral for `abort_signal`.
- [ ] Add explicit runtime handling or explicit documented deferral for `stepMs` / `chunkMs` /
  per-tool timeout semantics.

## Track D - Intentional limitations

- [-] Do not pretend the full AI SDK request transport contract is already consumed uniformly by
  every Siumai helper.
