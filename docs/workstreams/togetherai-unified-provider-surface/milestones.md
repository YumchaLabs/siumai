# TogetherAI Unified Provider Surface - Milestones

Last updated: 2026-04-20

## TUP-M0 - Scope locked

Acceptance criteria:

- the AI SDK TogetherAI reference surface is identified
- the Siumai split between `together` and `togetherai` is written down explicitly
- the workstream chooses a canonical public provider id

Status: completed

## TUP-M1 - Canonical provider id is unified

Acceptance criteria:

- `togetherai` is the canonical TogetherAI provider id
- the public default model is chat-led, not rerank-led
- compat defaults exist on canonical `togetherai` without preserving a second public TogetherAI id

Current state:

- `siumai-provider-openai-compatible` now exposes `togetherai` preset metadata and family defaults
- public OpenAI-compatible builder/discovery surfaces now use canonical `togetherai` instead of the
  old `together` alias
- the explicit compat escape hatch is now `togetherai_openai_compatible()` on both
  `Provider::openai()` and `Siumai::builder().openai()`, while the unified builder keeps the
  hidden `together` alias only internally as a migration bridge so it does not collide with the
  native `togetherai` provider id
- registry/provider metadata use `togetherai` as the canonical TogetherAI id
- the default model is now `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`

Status: completed

## TUP-M2 - Unified runtime surface exists

Acceptance criteria:

- one built `togetherai` client can expose chat/completion/embedding/image/audio plus rerank
- image and rerank remain provider-owned
- registry handles can still resolve provider-specific family paths cleanly

Current state:

- `TogetherAiProviderFactory` now aggregates compat text/audio clients, a provider-owned image
  client, and native rerank
- `TogetherAiUnifiedClient` delegates chat/completion/embedding/speech/transcription to compat,
  image generation/edit to the provider-owned image client, and rerank to native
- completion, embedding, image, speech, transcription, and rerank family-model paths are all wired,
  with canonical image generation/edit pinned to `/images/generations`
- `provider_catalog` now lists TogetherAI's unified family defaults across chat, embedding, image,
  speech, transcription, and rerank instead of only the earlier text/rerank subset
- `provider_ext::togetherai` now also exposes curated model constants plus AI SDK-style option
  aliases (`TogetherAIImageModelOptions`, `TogetherAIRerankingModelOptions`) plus the deprecated
  package-compat aliases (`TogetherAIImageProviderOptions`, `TogetherAIRerankingOptions`), so
  public interface audits no longer have to compare only raw string literals and provider-specific
  Rust names
- `provider_catalog` now also includes the audited curated TogetherAI chat/completion/embedding/
  image/rerank model subset instead of stopping at family defaults only

Status: completed

## TUP-M3 - Public facade and docs match the new story

Acceptance criteria:

- `Provider::togetherai()` exposes the unified builder path
- public-path tests pin the unified builder/runtime semantics
- changelogs and workstream docs explain the migration clearly

Current state:

- `Provider::togetherai()` now returns the unified `SiumaiBuilder`
- public import/runtime tests now lock the unified top-level builder, provider-owned image
  generation/edit semantics, typed image options, and rerank parity behavior
- changelogs and structural-alignment docs now describe `togetherai` as the canonical unified
  surface with compat text/audio lanes plus provider-owned image/rerank
- the compat `togetherai` plus legacy `together` presets now also advertise `completion`
  explicitly in static metadata so the shared compat package boundary matches the audited AI SDK
  surface without relying on inferred completion support

Status: completed

## TUP-M4 - Broader provider audit continues elsewhere

Acceptance criteria:

- TogetherAI is no longer blocking the AI SDK provider-surface audit
- the next remaining candidates are explicitly named

Current state:

- TogetherAI is no longer a structural parity blocker
- the next likely AI SDK-alignment candidates are no longer TogetherAI itself; any remaining work
  is optional cleanup such as removing the last low-level `together` compatibility lookup

Status: in progress
