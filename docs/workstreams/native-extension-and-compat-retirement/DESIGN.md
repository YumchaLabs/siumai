# Native Extension And Compat Retirement

Status: Active
Last updated: 2026-05-25

## Why This Lane Exists

The compatibility-surface breaking-convergence lane narrowed the public compatibility surface and
moved registry image, speech, and transcription extras behind `ProviderExtensionFactory`. It did
not finish retiring the deeper generic-client compatibility shape because method-style builders,
extension-only adapters, and low-level `ContentPart` root paths still depend on ADR-0007 and
ADR-0008 prerequisites.

This lane exists to convert those residual risks into provider-native, test-visible steps.

## Relevant Authority

- ADRs:
  - `docs/adr/0007-llmclient-demotion-policy.md`
  - `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
- Existing docs:
  - `docs/architecture/public-surface.md`
  - `docs/migration/migration-0.11.0-beta.7.md`
- Related workstreams:
  - `docs/workstreams/compatibility-surface-breaking-convergence/`
  - `docs/workstreams/fearless-public-surface-deepening/`
  - `docs/workstreams/fearless-registry-facade-construction-boundary/`

## Problem

Three architecture debts remain connected:

- `ProviderExtensionFactory` exposes native extension construction hooks, but default
  implementations still fall back to `compat_*_client_with_ctx` adapters.
- Method-style and generic-client construction keep `ProviderCompatibilityFactory` and
  `LlmClient` discoverable as a second construction model.
- ADR-0008 still blocks a full low-level `ContentPart` root namespace move until serde-facing
  payload parity, provider/protocol fixture coverage, and directional content namespaces are mature.

Treating these as separate cleanups risks moving one dependency while accidentally strengthening
another. Treating them as one lane keeps the target architecture simple: stable and extension
families should be native-first, and compatibility should remain explicit, narrow, and removable.

## Target State

- Built-in providers with native extension clients override `ProviderExtensionFactory` hooks instead
  of inheriting generic-client adapter fallbacks.
- Remaining `compat_*_client_with_ctx` call sites are classified as method-style migration,
  provider prerequisite, or intentionally deferred extension adapter work.
- Method-style/generic-client retirement has a source-enforced plan with concrete deletion
  preconditions instead of open-ended deprecation text.
- ADR-0008 root `ContentPart` movement has executable parity gates, and any safe preparatory
  namespace movement is completed without breaking serde payloads.

## In Scope

- Inventory and classify provider extension factory defaults and overrides.
- Add provider-owned native extension factory overrides where a native client already exists.
- Tighten registry architecture guards so new stable or native-extension paths do not regress into
  `ProviderCompatibilityFactory`.
- Document and test method-style/generic-client retirement preconditions.
- Add ADR-0008 parity gates or preparatory tests that make the root namespace move mechanically
  safer.

## Out Of Scope

- Removing `LlmClient` before ADR-0007 deletion conditions are satisfied.
- Breaking `ChatMessage` or `ChatResponse` serde shapes without fixture parity.
- Rewriting provider runtime behavior beyond construction-path ownership.
- Promoting extension-only capabilities into stable model families without a separate ADR.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Some built-in providers already have native image, speech, or transcription clients that can be returned directly from extension hooks. | Medium | Registry defaults currently adapt generic clients, while provider packages expose native family clients. | Keep the inventory as the first executable task and split provider-specific blockers. |
| Method-style/generic-client construction cannot be deleted in one pass. | High | ADR-0007 still treats `LlmClient` as a compatibility surface. | Strengthen deprecation gates and deletion criteria rather than forcing a breaking removal. |
| ADR-0008 root movement needs more test infrastructure before low-level paths move. | High | The previous lane recorded serde payload and provider/protocol parity blockers. | Build parity gates first and only move paths when the gates prove safety. |

## Architecture Direction

Provider construction should be explicit about the contract being built:

- stable model families use `ProviderFamilyFactory`;
- extension capabilities use `ProviderExtensionFactory`;
- generic `LlmClient` construction remains isolated as a compatibility migration path.

The same principle applies to content. Prompt-facing, response-facing, and legacy content shapes
should have separate namespaces, with root re-exports reserved for stable, non-legacy types.

## Closeout Condition

This lane can close when:

- provider extension fallback use is inventoried and reduced for at least one native-capable
  provider family;
- generic-client retirement criteria are guarded by tests or source checks;
- ADR-0008 root-move prerequisites have an executable gate or a safe preparatory slice;
- evidence gates pass for touched crates; and
- remaining removals are split into concrete follow-ons.
