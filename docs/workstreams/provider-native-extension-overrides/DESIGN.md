# Provider Native Extension Overrides

Status: Active
Last updated: 2026-05-25

## Why This Lane Exists

`ProviderExtensionFactory` is already the registry seam for extension-only capabilities such as
files, skills, music, image extras, speech extras, and transcription extras. The previous native
extension lane moved hybrid image extras to native provider clients, but several built-in providers
still inherit default extension methods that build a generic `LlmClient` and then adapt it back into
the same extension capability.

That fallback preserves a compatibility architecture in the middle of otherwise native provider
factories. It makes the registry harder to reason about because stable handles can request an
extension hook while construction still flows through a legacy generic-client path.

## Relevant Authority

- `docs/adr/0007-llmclient-demotion-policy.md`
- `docs/adr/0006-family-model-first-trait-policy.md`
- `docs/workstreams/native-extension-and-compat-retirement/`
- `docs/architecture/registry-without-builtins.md`

## Problem

Provider clients for Azure OpenAI, OpenAI, Anthropic, Gemini, xAI, and MiniMaxi already implement
one or more extension capability traits directly:

- file management for Azure OpenAI, OpenAI, Anthropic, Gemini, xAI, and MiniMaxi;
- skills for OpenAI and Anthropic;
- music generation for MiniMaxi.

The registry factory methods for those providers do not currently make that native ownership
visible. Calls can still flow through `compat_language_client_with_ctx(...)`, `as_*_capability()`,
and `ClientBacked*` adapters even when the factory can return the provider-owned typed client as the
extension trait object directly.

## Target State

- Built-in provider factories return typed provider clients directly from extension hooks when the
  typed client already implements the extension trait.
- Generic-client adapter defaults remain only as compatibility fallback for providers without a
  proven native extension hook.
- A source-level boundary test prevents the selected provider-native extension hooks from
  regressing into `compat_language_client_with_ctx(...)`, `as_*_capability()`, or `ClientBacked*`
  adapter glue.
- Speech and transcription extras remain out of this lane unless a provider-owned native object is
  proven separately.

## In Scope

- Registry factory overrides for provider-native file, skills, and music extension hooks.
- Focused architecture boundary tests in `siumai-registry`.
- Workstream evidence and index updates needed to track the slice.

## Out Of Scope

- Removing `LlmClient` or default `ProviderFactory` compatibility methods.
- Rewriting provider runtime behavior or request serialization.
- Promoting extension-only capabilities into stable model families.
- Speech/transcription extras whose current implementation still delegates through generic
  OpenAI-compatible clients.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| The selected provider clients implement the same `siumai_core::traits::*` capability traits that registry extension hooks return. | High | Provider crates import or re-export core traits and have direct impls for the selected clients. | Compile gates fail; split the provider out and keep only proven hooks. |
| Returning `Arc<ProviderClient>` as `Arc<dyn ExtensionTrait>` preserves behavior while removing adapter glue. | High | Existing family methods already use the same typed-client Arc projection pattern. | Add provider-specific native extension object construction instead. |
| Speech/transcription extras need a separate pass. | Medium | Several providers expose extra methods through generic-client delegates rather than dedicated native objects. | Revisit once provider-local native extra clients exist. |

## Architecture Direction

Provider factories should be explicit about the contract being constructed:

- model families use `ProviderFamilyFactory`;
- extension-only capabilities use `ProviderExtensionFactory`;
- generic `LlmClient` construction is reserved for compatibility entry points.

This lane deepens the extension seam by making provider-native hooks carry the extension contract
directly instead of tunneling through a wider generic-client interface.

## Closeout Condition

This lane can close when:

- the selected native extension hooks are implemented and guarded;
- focused registry tests and checks pass under the touched provider feature set;
- evidence is recorded; and
- any remaining extension defaults are classified as provider prerequisite work, not hidden active
  scope.
