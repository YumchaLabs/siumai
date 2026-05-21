# FCAB-040 Review

Status: Accepted
Date: 2026-05-21
Scope: Built-in provider factory projection cleanup.

## Workstream Compliance

- No blocking findings.
- The task goal is satisfied for this slice: built-in provider factories now construct native or
  provider-owned typed clients through local family-first helpers and reuse local typed-client `Arc`
  projection helpers for both stable family methods and explicit compatibility methods.
- The change stayed inside the FCAB-040 scope:
  `siumai-registry/src/registry/factories`, factory architecture source guards, and architecture /
  seam-inventory docs.
- Compatibility adapters that still exist are explicit:
  - hybrid compatibility composites for DeepInfra, Fireworks, and TogetherAI remain private
    `compat_language_client_with_ctx` adapters;
  - extension-only or unsupported family paths still return explicit unsupported-operation errors;
  - compatibility methods are not used as the same-family construction shortcut.

## Code Quality

- No blocking findings.
- The refactor increases locality: repeated `build_*_with_ctx(...)` + `Arc::new(...)` projection
  logic now lives in one helper per relevant typed-client shape.
- The added source guards cover:
  - OpenAI and generic OpenAI-compatible projection helpers;
  - promoted OpenAI-compatible hybrid vendor text/image/rerank projection helpers;
  - the wider built-in provider factory set.
- The source guards are intentionally structural. They protect the architecture seam that unit tests
  cannot easily observe without downcasting every provider family. Representative runtime contract
  tests still exercise provider-owned/native construction paths.

## Missing Gates

- No missing FCAB-040 task-local gates.
- A broad `cargo check -p siumai-registry --features openai --tests` previously failed with a
  rustc memory/stack allocation failure. This was treated as infrastructure noise and replaced with
  low-concurrency focused checks plus representative nextest gates. The low-concurrency widened
  `--lib` compile gate passed.

## Residual Risk

- This task intentionally did not move protocol/runtime logic out of OpenAI-compatible provider
  crates. FCAB-050 owns that deeper protocol/runtime seam.
- This task intentionally did not rewrite vendor-owned presets, typed options, facade extensions,
  or metadata. FCAB-060 owns that vendor-module cleanup.
- Some factories still have compatibility methods because they are public migration or
  extension-only paths. The completed guard ensures those paths do not become the primary
  same-family construction shortcut.
