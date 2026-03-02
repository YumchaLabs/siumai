# Fearless Refactor V3 — Milestones

Last updated: 2026-03-02

This workstream is tracked by milestones with explicit acceptance criteria.

## V3-M0 — Baseline safety

Acceptance criteria:

- Workspace builds for common feature sets.
- Existing integration tests still compile (`--no-run`) on the branch.
- No new crates introduced for tools/UI.

Status: ✅ done

## V3-M1 — Client foundation decoupled

Acceptance criteria:

- `LlmClient` is no longer “chat-first” (no inheritance from chat capability).
- Registry model handles still work with caching + middleware.

Status: ✅ done

## V3-M2 — V3 model-family traits exist (six families)

Acceptance criteria:

- V3 traits for: text, embedding, image, rerank, speech, transcription.
- Text supports both non-stream and stream in a single interface.
- Minimal no-network tests exist for trait adapters.

Status: ✅ done

## V3-M3 — New recommended public API in `siumai`

Acceptance criteria:

- `siumai` exposes function-style family APIs (Rust-first naming).
- Registry handles can be passed directly to these APIs.
- Compatibility layer exists for existing examples/tests (temporary).

Status: ✅ done

## V3-M4 — Tools unified (schema + execute)

Acceptance criteria:

- A single executable tool type exists.
- Typed wrapper exists but is optional.
- Adapters exist from the current tool system.

Status: ✅ done

## V3-M5 — Orchestrator migrated

Acceptance criteria:

- `siumai-extras` orchestrator uses the new text APIs.
- Stop conditions, approvals, streaming remain functionally equivalent.

Status: ✅ done

## V3-M6 — Core providers migrated

Acceptance criteria:

- OpenAI / Anthropic / Gemini providers implement the new family traits (directly or via adapters).
- No-network tests cover provider-specific mapping invariants.

Status: ✅ done

## V3-M7 — Cleanup and stabilization

Acceptance criteria:

- `compat` surface is documented and time-bounded.
- Examples/docs updated to new recommended APIs.
- Release notes prepared (breaking changes clearly listed).

Status: 🟨 in progress

## V3-M8 — Config-first construction (beta.6)

Acceptance criteria:

- Core providers support config-first construction (no global builder required for new code):
  - `OpenAiClient::from_config(...)` (or equivalent)
  - `AnthropicClient::from_config(...)` (or equivalent)
  - `GeminiClient::from_config(...)` (or equivalent)
- README + the top “quickstart” examples no longer require `Siumai::builder()`.
- Builder surface is explicitly marked as `compat` and time-bounded.

Status: ✅ done
