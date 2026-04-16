# Fearless Refactor Workstream — TODO

Note (2026-03-01): This TODO list is for the earlier phase. The active V3 workstream plan is in
`docs/workstreams/fearless-refactor-v3/todo.md`.

Last updated: 2026-04-06 (historical phase-1 record)

## Status snapshot

✅ Landed:

- `siumai-spec` extracted (types/tools/errors/telemetry config), `siumai-core` re-exports shims.
- Provider id resolver centralized in `siumai-registry`.
- Unified builder/build path delegates API key and base_url defaults to `ProviderFactory`.
- Registry handle normalizes common aliases when safe.
- Provider catalog prefers shared native metadata for built-in providers (avoids “Custom provider” mislabeling).
- This earlier phase assumed Cohere/TogetherAI were rerank-led; that historical decision has since
  been superseded by the newer unified-provider workstreams (`cohere`, `togetherai`).
- Standardize multipart MIME validation errors as `InvalidParameter` (avoid misleading `HttpError`).

## TODO (next)

### Registry routing (high value)

- [x] Reduce the `provider/build.rs` `match ProviderType` further by delegating to factories uniformly
      (aim: only build `BuildContext` + select the factory; no per-provider wiring).
- [x] Consider deprecating `SiumaiBuilder.provider_type` in favor of `provider_id` as the sole routing key,
      or enforce a strict consistency rule everywhere.
- [x] Make provider “variant routing” explicit:
  - `openai-chat` / `openai-responses`
  - `azure-chat`
  - centralized ids + parsing live in `siumai-registry/src/provider/ids.rs`

### Resolver cleanup

- [x] Prefer `provider_id`-first helpers for inference and OpenAI-compatible behaviors.

### Spec/runtime boundary hardening

- [x] Audit `siumai-spec` dependencies for “runtime creep”
      (keep heavy deps behind features; prefer `siumai-core` for runtime-only conversions/helpers).
- [x] Decide whether `siumai-spec::error` should keep `From<reqwest::Error>` at all.
      Decision (2026-02-27): remove `reqwest` from `siumai-spec` entirely and use explicit error mapping
      (`LlmError::HttpError(...)`, `TimeoutError(...)`, etc.) in runtime code.

### Factory consistency

- [ ] Ensure every built-in `ProviderFactory` applies the same BuildContext precedence:
  1) `ctx.http_client` > build from `ctx.http_config`
  2) `ctx.api_key` > env var (when required)
  3) `ctx.base_url` > provider default (when applicable)
-   - Progress: contract tests cover `openai`, OpenAI-compatible presets (`deepseek`, `openrouter`),
        `azure`, `anthropic`, `gemini`, `groq`, `xai`, `ollama`, `minimaxi`, `vertex`, `anthropic-vertex`.
- [x] Add a small set of “contract tests” that can be reused by each factory (no network),
      and expand coverage across multiple factories.

### Metadata-only providers (clarify intent)

- [x] Historical policy decision is superseded by later workstreams:
  - `cohere` is now a first-class unified native `/v2` provider surface
  - `togetherai` is now a first-class unified public provider surface
  - `bedrock` now has a first-class factory/provider path instead of metadata-only reservation

Decision (2026-02-27):

- [x] Phase-1 historical step:
  - `cohere` and `togetherai` first landed as focused rerank-led factories before later
    unified-surface refactors expanded the public story
- [x] Phase-1 historical step:
  - `bedrock` was temporarily metadata-only before later first-class factory/provider work landed

### Docs & tooling

- [ ] Add a lightweight changelog note for the refactor milestones (when ready).
- [ ] Add a migration note if any public paths change (only when unavoidable).
