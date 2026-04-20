# Speech And Transcription Helper Parity

Last updated: 2026-04-20

## Goal

Align the public Rust speech/transcription helper semantics more closely with AI SDK
`generateSpeech()` and `transcribe()` without inventing a fake provider-agnostic audio runtime.

Upstream reference:

- `repo-ref/ai/packages/ai/src/generate-speech/generate-speech.ts`
- `repo-ref/ai/packages/ai/src/generate-speech/generate-speech-result.ts`
- `repo-ref/ai/packages/ai/src/transcribe/transcribe.ts`
- `repo-ref/ai/packages/ai/src/transcribe/transcribe-result.ts`
- `repo-ref/ai/packages/ai/src/error/no-speech-generated-error.ts`
- `repo-ref/ai/packages/ai/src/error/no-transcript-generated-error.ts`

## What Changed

- `TtsResponse` and `SttResponse` now carry best-effort final `response` metadata on the stable
  Rust surface, matching the role of AI SDK speech/transcription response metadata more closely.
- The shared `AudioExecutor` now captures response headers plus model identity for successful TTS
  and STT calls instead of dropping that information below the provider boundary.
- OpenAI, OpenAI-compatible, Azure, Groq, xAI, and MiniMaxi audio paths now all preserve that
  `response` envelope when they lower executor results into stable Rust response structs.
- The public helper lane now mirrors AI SDK empty-result semantics more closely:
  - `siumai::speech::synthesize(...)` returns `LlmError::NoSpeechGenerated` when the provider
    succeeds but returns empty audio bytes.
  - `siumai::transcription::transcribe(...)` returns `LlmError::NoTranscriptGenerated` when the
    provider succeeds but returns blank transcript text.
- Both specialized errors now carry the final response metadata array, so helper callers can debug
  provider-specific filtered/empty outputs without falling back to generic parse errors.

## Architectural Decision

The response metadata capture lives in the shared executor, not in ad hoc provider wrappers.

Reasoning:

- AI SDK treats empty-result speech/transcription failures as helper-level semantics, but those
  errors still carry provider response metadata.
- If Rust only added `response: None` at each provider boundary, the surface would compile but the
  higher-level parity would still be false because the specialized error types would lose the final
  response envelope.
- Putting the metadata capture into `AudioExecutor` keeps the shared contract honest and avoids
  repeating header-to-map conversion logic across every provider.

## Compatibility Notes

- This change does not invent a new unified audio helper beyond the current Rust-first
  `speech::synthesize(...)` and `transcription::transcribe(...)` facades.
- Providers that do not use the shared audio executor can still populate `response` manually, but
  the current audited providers now inherit the stable behavior from the executor path directly.
- Audio translation still uses the same stable `SttResponse` shape, so provider-owned translation
  paths can preserve the same response metadata contract when available.

## Remaining Follow-up

- The shared speech/transcription call-option shape is already close to AI SDK parity; the
  remaining work is now mostly provider-specific feature/runtime coverage rather than shared
  request/response structure.
- If Siumai later introduces richer batched speech/transcription helpers, the stable response shape
  may need the same multi-call metadata discussion that already exists for image/video helpers.
