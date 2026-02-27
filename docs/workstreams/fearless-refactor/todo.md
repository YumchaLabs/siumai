# Fearless Refactor Workstream — TODO

Last updated: 2026-02-27

## Status snapshot

✅ Landed:

- `siumai-spec` extracted (types/tools/errors/telemetry config), `siumai-core` re-exports shims.
- Provider id resolver centralized in `siumai-registry`.
- Unified builder/build path delegates API key and base_url defaults to `ProviderFactory`.
- Registry handle normalizes common aliases when safe.

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

- [ ] Audit `siumai-spec` dependencies for “runtime creep”
      (keep heavy deps behind features; prefer `siumai-core` for runtime-only conversions/helpers).
- [ ] Decide whether `siumai-spec::error` should keep `From<reqwest::Error>` at all,
      or move HTTP-specific conversions to runtime.

### Factory consistency

- [ ] Ensure every built-in `ProviderFactory` applies the same BuildContext precedence:
  1) `ctx.http_client` > build from `ctx.http_config`
  2) `ctx.api_key` > env var (when required)
  3) `ctx.base_url` > provider default (when applicable)
-   - Progress: contract tests cover `openai`, OpenAI-compatible presets (`deepseek`, `openrouter`),
        `azure`, `anthropic`, `gemini`, `groq`, `xai`, `ollama`, `minimaxi`, `vertex`, `anthropic-vertex`.
- [x] Add a small set of “contract tests” that can be reused by each factory (no network),
      and expand coverage across multiple factories.

### Docs & tooling

- [ ] Add a lightweight changelog note for the refactor milestones (when ready).
- [ ] Add a migration note if any public paths change (only when unavoidable).
