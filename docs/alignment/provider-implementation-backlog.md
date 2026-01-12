# Provider Implementation Backlog (Post-M1)

This is the **action list** derived from `docs/alignment/provider-implementation-alignment.md`.
It is intentionally short and test-driven: each item should end with a fixture/test that locks behavior.

## P0 (high ROI, low risk)

- (Done) Error mapping parity (OpenAI / Anthropic):
  - Code: `siumai-protocol-openai/src/standards/openai/errors.rs`,
    `siumai-protocol-anthropic/src/standards/anthropic/errors.rs`,
    `siumai-core/src/retry_api.rs`
  - Behavior: prefers HTTP reason phrase for non-JSON / empty bodies; Anthropic overloaded maps to synthetic `529`.

- (Done) Azure error mapping parity:
  - Code: `siumai-provider-azure/src/providers/azure_openai/spec.rs`
  - Vercel ref: `repo-ref/ai/packages/azure/src/*` (Azure provider delegates to OpenAI internals)

## P1 (gateway stability)

- (Done) OpenAI-compatible streaming quirks parity (vendor variance):
  - Code: `siumai-core/src/standards/openai/compat/streaming.rs`
  - Tests: `siumai-core/src/standards/openai/compat/streaming_tests.rs`

## P2 (breadth)

- Extend this audit doc set to non-core providers (as we add/upgrade them):
  - Bedrock, Groq, xAI, Ollama, MiniMaxi, Cohere, TogetherAI
  - Goal: at least one “request mapping” fixture and one “error mapping” fixture per provider family.
  - Progress:
    - Bedrock: chat request fixtures + HTTP error fixture ✅
    - Cohere: rerank request/response fixtures + HTTP error fixture ✅
    - TogetherAI: rerank request/response fixtures + HTTP error fixture ✅
    - Ollama: chat request fixtures + HTTP error fixture ✅

## How to validate

- Run the M1 smoke matrix: `scripts/test-m1.bat` / `bash scripts/test-m1.sh`
- Run a focused nextest suite for the provider you touched.
- Keep `scripts/audit_vercel_fixtures.py` at `Missing: 0` and `Drift: 0`.
