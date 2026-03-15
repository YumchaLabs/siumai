# Fearless Refactor V4 - Risk Register

Last updated: 2026-03-08

## Purpose

This document records the major refactor risks, the expected signals, and the mitigation strategy.
It should be reviewed whenever milestone scope changes.

## Risk scale

- Severity: Low / Medium / High
- Likelihood: Low / Medium / High

## R1 - Streaming regression

- Severity: High
- Likelihood: High
- Area: `text::stream`, SSE parsing, chunk assembly, stream finish handling
- Failure mode: missing deltas, duplicated deltas, malformed end events, broken cancellation
- Signals:
  - fixture stream mismatches
  - hanging stream tests
  - different final text after chunk aggregation
- Mitigation:
  - preserve fixture/alignment tests throughout migration
  - keep stream adapters small and family-specific
  - validate stream start/delta/usage/end invariants early

## R2 - Tool calling regression

- Severity: High
- Likelihood: High
- Area: tool schema mapping, tool call/result loops, provider-defined tools
- Failure mode: tool names drift, argument encoding changes, tool result shape mismatch
- Signals:
  - existing tool-loop examples break
  - fixture mismatch for provider-defined tool flows
  - tool result content no longer round-trips
- Mitigation:
  - keep `siumai-spec` tool/message shapes stable
  - run provider parity tests during migration
  - explicitly test provider-defined tool paths for OpenAI and Anthropic

## R3 - Registry cache behavior drift

- Severity: Medium
- Likelihood: Medium
- Area: handle caching, TTL expiration, provider/model override caching keys
- Failure mode: stale clients reused incorrectly, cache misses explode, wrong model cached after middleware override
- Signals:
  - test flakiness around custom middleware
  - repeated client builds in hot paths
  - wrong provider/model selected after override
- Mitigation:
  - lock cache-key rules before implementation
  - keep focused no-network cache tests
  - verify override logic separately from provider execution logic

## R4 - Builder/config divergence

- Severity: High
- Likelihood: Medium
- Area: provider construction
- Failure mode: builder path and config path behave differently
- Signals:
  - one path supports options the other path misses
  - tests pass for builder but fail for config-first, or vice versa
- Mitigation:
  - define config structs as canonical construction inputs
  - require builder/config parity tests on major providers
  - reject builder-only features in review

## R5 - Audio contract confusion

- Severity: Medium
- Likelihood: High
- Area: TTS/STT split
- Failure mode: speech and transcription remain mixed behind broad audio abstractions
- Signals:
  - new code still uses broad audio entry points as the default path
  - speech/transcription handles internally depend on broad compatibility traits
- Mitigation:
  - define explicit `SpeechModel` and `TranscriptionModel` ownership early
  - keep `AudioCapability` as compatibility-only during migration
  - add migration examples and tests for both families

## R6 - Provider option mapping drift

- Severity: High
- Likelihood: Medium
- Area: `providerOptions`, request transforms, response metadata
- Failure mode: provider-specific options disappear or move incorrectly
- Signals:
  - OpenAI/Anthropic/Gemini alignment tests fail
  - options accepted before are silently ignored after refactor
- Mitigation:
  - keep provider option parsing in provider crates
  - avoid moving provider-specific interpretation into shared family traits
  - add provider parity fixtures for major options

## R7 - Public API story becomes more confusing during migration

- Severity: Medium
- Likelihood: High
- Area: docs, examples, public exports
- Failure mode: users see registry-first, builder-first, config-first, and low-level client paths all presented as equal
- Signals:
  - docs contradictions
  - examples mix preferred and compatibility paths without explanation
- Mitigation:
  - enforce compatibility-layer wording
  - rank paths explicitly in docs
  - require reviewer checks for API-layer clarity

## R8 - Over-refactoring `siumai-spec`

- Severity: Medium
- Likelihood: Medium
- Area: shared types and naming
- Failure mode: unnecessary renames create migration noise without architectural gain
- Signals:
  - patches rename many request/response/message types without changing semantics
  - downstream docs and tests churn heavily for naming-only changes
- Mitigation:
  - keep existing good names
  - require explicit justification for type renames
  - separate semantic change from naming cleanup

## R9 - Provider migration stalls after OpenAI

- Severity: High
- Likelihood: Medium
- Area: rollout sequencing
- Failure mode: architecture fits only OpenAI and leaves other providers in long-lived adapter limbo
- Signals:
  - Anthropic/Gemini migration stories remain unclear after OpenAI lands
  - provider-specific exceptions start leaking into core traits
- Mitigation:
  - validate trait design against OpenAI, Anthropic, and Gemini before locking it
  - treat the first three providers as the true architecture fit test

## R10 - Hidden performance regressions

- Severity: Medium
- Likelihood: Medium
- Area: extra indirection, trait objects, repeated adaptation
- Failure mode: more allocations, more dynamic dispatch, repeated translation work
- Signals:
  - benchmark regressions in hot generation/streaming paths
  - repeated cloning or request re-materialization in traces
- Mitigation:
  - keep adapters thin
  - avoid double-transforming requests in handles and providers
  - ensure provider-specific adapter wrapping is assembled once during config/build time rather than per request (DeepSeek's compat-default extraction is the expected wrapper pattern)
  - prefer caching final family-model objects over rebuilding wrapper stacks on each handle access
  - keep compat vendor-view defaults on the same one-time `RegistryOptions|ProviderBuildOverrides -> BuildContext` path used by native providers; OpenRouter registry reasoning is the current reference case, and rebuilding vendor defaults ad hoc inside handles or splitting global/provider lanes into different code paths would add hidden request-shaping overhead plus another drift surface
  - keep registry audio extras on the provider-owned client path instead of re-encoding multipart/streaming request shaping inside handles; OpenAI `audio_translate` is now the reference case, and if extras become hot enough to matter we should add a parallel typed-client cache rather than fork protocol translation logic
  - keep unsupported extras short-circuiting before transport on every public entry point; Groq translation is the current boundary case, and letting “extras exist” devolve into shaped-but-doomed requests would add hidden overhead and muddy the surface contract
  - keep speech extras on that same delegated path; xAI `tts_stream` is the current guard case, and the registry speech handle should not hardcode today’s unsupported default in a way that would diverge once a provider-owned client grows real streaming support
  - add targeted benchmarks or smoke timing checks for hot paths
- keep custom transport coverage symmetrical across JSON and streaming executors so accidental fallback to real HTTP is caught before it distorts hot-path measurements
- prefer provider-local metadata-key normalization wrappers over extra parse/serialize passes; MiniMaxi's `anthropic -> minimaxi` rekey path should stay a top-level map rewrite on already-materialized response objects, not a second protocol transformation layer
- keep public-path parity growth capture-only and no-network; Google Vertex embedding/imagen parity is the reference pattern, so surface expansion does not add response-fixture churn or hide hot-path cost behind real HTTP
- keep rerank-only provider migrations on the same provider-owned client path used by registry/facade entry points; Cohere and TogetherAI are the reference pattern, avoiding duplicate executor wrappers that would otherwise skew latency and maintenance cost
- keep split-endpoint providers on a single provider-owned config/client path as well; Amazon Bedrock is now the reference pattern, so chat and rerank do not drift into separate factory-only URL derivation branches that would hide transport overhead and maintenance cost

## R11 - OpenAI-compatible endpoint-shape over-assumption

- Severity: High
- Likelihood: Medium
- Area: OpenAI-compatible secondary capabilities, especially speech/transcription
- Failure mode: a provider is marked as `speech` / `transcription` capable just because it is "OpenAI-compatible", even when its real audio routes or base URLs do not match the shared compat audio family contract
- Signals:
  - provider docs mention audio only under `chat/completions` multimodal flows
  - TTS or STT lives on a different route family such as `/v1/tts`
  - TTS/STT uses a dedicated base URL instead of the compat base URL used for chat/embedding/image
- Mitigation:
  - require official provider-doc verification before enabling compat audio capability flags
  - keep provider enrollment narrow when only a subset of vendors matches the shared compat audio contract
  - add explicit negative tests for pending vendors so capability drift is caught early
  - promote split-route vendors to provider-owned audio specs instead of forcing them through the shared OpenAI audio transformer

## Review cadence

This register should be reviewed:

- before milestone implementation starts
- when a new provider migration begins
- before release notes are drafted

## Escalation rule

If a High-severity risk becomes active in real test failures, do not continue broad migration blindly.
Pause and either:

- narrow the milestone scope
- restore a compatibility bridge temporarily
- add a dedicated parity test before continuing

