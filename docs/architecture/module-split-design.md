# Modular Split Design (Crates, Ownership, Dependencies)

This document describes the target modular architecture and the concrete ownership rules
that guide the fearless refactor.

It is inspired by Vercel AI SDK’s dependency direction and TanStack AI’s capability-first
composition, adapted for Rust (feature gating + compile-time ergonomics).

## Goals

- Minimize coupling in provider-agnostic code.
- Make provider-specific features evolve without changing core enums/types.
- Enable reuse for OpenAI-like (OpenAI-compatible) vendors without duplicating protocol logic.
- Preserve a stable, small public surface for users.

## Non-goals

- A UI/React/RSC style package ecosystem.
- Forcing “capability gating” that blocks operations; capability info remains advisory.

## Current state (beta.5)

Workspace members:

- `siumai` — facade crate
- `siumai-core` — provider-agnostic runtime + types (protocol mapping moved out; remaining coupling is being reduced)
- `siumai-registry` — registry + factories + handles (optional built-ins via feature)
- `siumai-extras` — orchestrator + telemetry + server + MCP
- `siumai-provider-openai` — OpenAI provider implementation (native) + OpenAI-compatible vendor wiring
- `siumai-protocol-openai` — OpenAI-like protocol standard (shared mapping + streaming/tool-call helpers; preferred name)
- `siumai-provider-openai-compatible` — OpenAI-like protocol standard (legacy crate name; compatibility alias)
- `siumai-provider-ollama` — Ollama provider + Ollama protocol standard
- `siumai-provider-anthropic` — Anthropic provider implementation (native)
- `siumai-protocol-anthropic` — Anthropic Messages protocol standard (shared mapping + streaming; preferred name)
- `siumai-provider-anthropic-compatible` — Anthropic Messages protocol standard (legacy crate name; compatibility alias)
- `siumai-provider-gemini` — Gemini provider implementation
- `siumai-protocol-gemini` — Gemini protocol standard (mapping + streaming)
- `siumai-provider-google-vertex` — Vertex provider implementation (Imagen-focused)
- `siumai-provider-groq` — Groq provider (OpenAI-like protocol via `siumai-protocol-openai`)
- `siumai-provider-xai` — xAI provider (OpenAI-like protocol via `siumai-protocol-openai`)
- `siumai-provider-minimaxi` — MiniMaxi provider (Anthropic chat + OpenAI-like media endpoints)

## Target layering (dependency direction)

The key rule is: **dependencies point inward** (high-level → low-level), never the reverse.

```text
User code
  ↓
siumai (facade)
  ↓
siumai-registry (optional)
  ↓
  ├─ siumai-provider-openai (OpenAI provider)
  ├─ siumai-protocol-openai (OpenAI-like protocol crate)  ← shared implementation layer (family crate)
  ├─ siumai-provider-openai-compatible (legacy alias)     ← compatibility only
  ├─ siumai-provider-ollama (Ollama provider + Ollama standard)
  ├─ siumai-provider-anthropic (Anthropic provider)
  ├─ siumai-protocol-anthropic (Anthropic Messages protocol crate) ← shared implementation layer (family crate)
  ├─ siumai-provider-anthropic-compatible (legacy alias)           ← compatibility only
  ├─ siumai-protocol-gemini (Gemini protocol crate)                           ← shared implementation layer (protocol crate)
  ├─ siumai-provider-gemini (Gemini provider)
  ├─ siumai-provider-google-vertex (Vertex provider)
  ├─ siumai-provider-groq (Groq provider)
  ├─ siumai-provider-xai (xAI provider)
  ├─ siumai-provider-minimaxi (MiniMaxi provider)
  └─ (future provider crates)
  ↓
siumai-core (provider-agnostic runtime + shared types)
```

Notes:

- We keep a single shared *family* crate for the OpenAI-like protocol (`siumai-protocol-openai`)
  because multiple providers reuse the OpenAI-like mapping logic (Groq/xAI/OpenAI-compatible vendors, and parts of MiniMaxi).
- We keep a shared *family* crate for the Anthropic Messages protocol (`siumai-protocol-anthropic`)
  because multiple providers reuse the Messages mapping (Anthropic native, and providers like MiniMaxi).
- New protocol crates follow the `siumai-protocol-*` naming convention. Existing `*-compatible` crates are
  treated as protocol crates but keep their names for compatibility.
- `siumai-core` must not import provider-specific protocol modules.

## Ownership rules (what belongs where)

### `siumai-core`

Owns:

- provider-agnostic types and traits for the stable surface
- streaming event normalization
- retry abstractions and HTTP configuration types
- middleware abstractions
- request/response types for the 6 model families

Must NOT own:

- provider protocol mapping modules (e.g., OpenAI/Gemini/Anthropic request/response schema mapping)
- provider-specific typed option structs and provider-specific metadata types

### Legacy: `siumai-providers` (umbrella)

Owns:

- (historical) compatibility shims that preserved old module paths

Status:

- The fearless refactor moved provider implementations into dedicated provider crates.
- The umbrella crate is no longer part of the workspace (kept only as a historical reference).

### `siumai-registry`

Owns:

- registry abstractions (factories, handles, caching)
- optional built-in provider registration behind `builtins` feature

Must NOT:

- force built-in providers by default

### `siumai`

Owns:

- stable public entry points and preludes
- feature aggregation and ergonomic APIs

Should NOT:

- re-export internal modules in a way that encourages cross-layer imports

### `siumai-extras`

Owns:

- orchestrator / agent loop utilities
- schema validation helpers and telemetry integrations
- server adapters and MCP helpers

## Provider options design (open, Vercel-aligned)

### Problem

Closed enums like `ProviderOptions::OpenAi(...)` force `siumai-core` changes whenever provider-specific
features evolve or new providers are added.

### Target shape

Provider options are a **pass-through map keyed by provider id**, where each entry is a JSON object.

```text
providerOptions: Map<provider_id, JSON-object>
```

Provider implementations:

- parse and validate their own options
- treat unknown keys as provider-owned (core does not interpret them)

Rust representation (current refactor stage):

- Requests carry `provider_options_map: ProviderOptionsMap` (re-exported from `siumai-core::types`).
- `ChatRequest` / `EmbeddingRequest` / etc. expose helpers like `with_provider_option(provider_id, json_value)`.
- There is no closed `ProviderOptions` enum transport in `siumai-core` (breaking change).

This matches Vercel’s `SharedV3ProviderOptions` concept:
`repo-ref/ai/packages/provider/src/shared/v3/shared-v3-provider-options.ts`.

### Typed provider options (where they live)

Typed option structs should live with the provider implementation (in provider crates like `siumai-provider-openai`),
not in `siumai-core`.

Ergonomic access should be provided via stable facade paths (e.g. `siumai::provider_ext::<provider>::*`) and
request/response extension traits behind provider features.

## OpenAI vs OpenAI-compatible vendors (shared adapter strategy)

### Why a shared layer exists

Many providers implement an OpenAI-like protocol surface with small deltas:

- base URL / headers / auth
- error envelope shape
- streaming chunk differences (e.g., reasoning fields)
- structured output support differences
- tool-call streaming quirks

Duplicating the protocol code across vendors creates drift and bug fixes become expensive.

### MVP approach (no new crates required)

We introduce one shared family crate (and evolve it into a provider crate as we split providers):

```text
siumai-provider-openai/
  src/standards/openai/*
  src/providers/openai/*
  src/providers/openai_compatible/*
```

Then:

- OpenAI-like providers reuse `siumai-protocol-openai` for protocol mapping and streaming/tool-call helpers.
- OpenAI-compatible vendors are treated as configuration (base URL / headers / error structure), not as separate crates.

This mirrors Vercel’s *concept* (shared adapter + vendor providers), while keeping crate count minimal.

## Where `standards/*` lives now

Provider-specific protocol mapping is protocol-crate-owned (provider crates re-export for compatibility):

- OpenAI-like (shared): `siumai-protocol-openai` (current impl: `siumai-provider-openai-compatible/src/standards/openai/*`)
- Ollama standard: `siumai-provider-ollama/src/standards/ollama/*`
- Anthropic Messages (shared): `siumai-protocol-anthropic` (current impl: `siumai-provider-anthropic-compatible/src/standards/anthropic/*`)
- Gemini standard: `siumai-protocol-gemini/src/standards/gemini/*`
- Vertex Imagen standard: `siumai-provider-google-vertex/src/standards/vertex_imagen.rs`

`siumai-core` may still contain **protocol-level shared building blocks** under `siumai-core/src/standards/*`
(e.g. OpenAI-compatible adapters / streaming converters / wire helpers). These are not part of the stable facade;
they are surfaced through `siumai::experimental::standards::*` via provider re-exports when needed.

## Standard-only features (reduce coupling)

Some provider crates expose a **standard-only** feature to allow other providers to reuse protocol
mapping logic without pulling the provider implementation modules:

- `siumai-protocol-openai/openai-standard`: OpenAI-like protocol mapping only (chat/embeddings/images/audio/rerank).
- `siumai-protocol-anthropic/anthropic-standard`: Anthropic Messages protocol mapping only (chat + streaming helpers).

## Capability composition (TanStack-inspired, Rust-friendly)

Even inside one provider, implementations should be split by capability:

- chat (+ streaming)
- embeddings
- images
- rerank
- speech
- transcription

Benefits:

- improves code locality
- enables finer-grained feature gating
- reduces cognitive load and merge conflicts

## Migration steps (practical)

1. Introduce open `providerOptions` map in request types (compat bridge).
2. Migrate one provider (OpenAI) to read from the map.
3. Move `standards/*` provider mapping modules out of `siumai-core` into provider-owned code.
4. Introduce shared `openai_like` module and refactor `openai` + `openai_compatible` to use it.
5. Tighten exports and document stable module paths.

Breaking-change details and “before/after” snippets:

- `docs/migration/migration-0.11.0-beta.5.md`

## Testing strategy

Use existing scripts as refactor safety nets:

- `./scripts/test-fast.sh` — fastest core-level checks
- `./scripts/test-smoke.sh` — minimal provider feature set compilation + unit tests
- `./scripts/test-full.sh` — CI-like matrix; uses `cargo nextest` when available

During refactors, keep `test-fast` green continuously; run `test-smoke` at each milestone boundary.
