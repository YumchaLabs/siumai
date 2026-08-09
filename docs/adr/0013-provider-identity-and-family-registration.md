# ADR-0013: Separate Provider Identity from Family Execution Scope

## Status

Accepted

## Date

2026-08-06

## Context

A configured provider may expose several model families and several API modes. Those surfaces do
not necessarily share a protocol, API mode, request policy, or fidelity claim:

- Groq combines OpenAI-compatible language execution with a native transcription API;
- Alibaba combines compatible language APIs with native embedding and video APIs;
- Cohere exposes embedding and rerank as separate model families;
- OpenAI, xAI, DeepSeek, and MiniMax expose more than one language API mode.

The former base `Provider::scope()` contract treated one `ProviderScope` as provider-wide identity.
Composite providers therefore had to choose an arbitrary preferred mode or whichever family happened
to be configured first. The former `ProviderRegistration` repeated the same mistake by storing one
scope and one model policy beside several family factories. A registration could consequently report
Chat Completions policy for a native transcription model or language policy for an embedding model.

Support metadata is a different concern again. A verified profile or support manifest records dated
evidence for named public claims. It does not determine whether an arbitrary future model ID can be
encoded, and a custom endpoint may be executable without carrying any official named claim.

## Decision

Siumai separates canonical provider identity, exact execution scope, dynamic family registration,
and support evidence.

### Provider identity

The base `Provider` trait exposes only `provider_id()`. `ProviderId` names canonical vendor or product
ownership and is the only provider-wide identity guaranteed by the trait.

The base trait does not expose platform, protocol, API mode, registrations, resources, or a capability
bag. Narrow family-provider traits describe which concrete model handles a provider can construct.

### Execution scope

`ProviderScope` describes one exact executable technical surface: provider, optional platform,
protocol, and API mode. It belongs to model descriptors, model-policy contexts, and family
registrations. It is not a summary of the whole configured provider.

Portable support evidence uses `SupportScope`, and provider-native evidence uses
`NativeSupportScope`. Those evidence types carry the additional dimensions required for a dated
claim and must not be confused with executable `ProviderScope`.

Public `scope()` accessors return the domain object rather than exposing the internal `Arc` used to
share immutable scope data.

`ModelOperation` uniquely determines `ModelFamily`. Policy contexts store the operation and derive
the family, so Registry and third-party policies cannot observe or construct a contradictory pair.

### Dynamic registration

A `ProviderRegistration` represents one canonical provider and contains at least one family binding.
Each family binding owns:

- its exact `ProviderScope`;
- its `ModelPolicy`;
- its erased model factory.

Bindings for different families may share a runtime while retaining different protocols and policies.
Merging registrations is allowed only for the same canonical provider and disjoint families. This
is explicit host-owned composition: matching provider identity does not prove that credentials,
endpoints, accounts, or runtime instances are identical. A combined registration can be projected
to one family when a route should expose a narrower allowlist.

One registration contains at most one default binding per family. When a provider exposes alternative
API modes for the same family, it publishes explicit mode-specific registrations. The facade may pick
one documented ergonomic default, while callers register alternate modes under separate host-owned
routes when they need both.

The facade's provider-registration adapter is fallible. A valid provider configured only for native
resources or jobs returns a typed `NoPortableFamilyRegistration` error instead of synthesizing an
empty registration or being omitted from the ergonomic adapter trait.

Registry consumes only the selected family binding. It does not infer capabilities from the provider
type, inspect concrete providers, or choose an API mode.

### Support evidence

Provider-owned profiles and support manifests remain separate introspection surfaces. A named claim
records exact scope, fidelity, stability, official source, and verification date. Registry and model
construction do not use that metadata as an allowlist.

The base `Provider` trait does not require `support_manifest()`. Providers with one portable profile
may expose that profile directly; composite providers may expose a manifest. A manifest takes an
explicit `ProviderId` and may contain no claims when a custom or native-only configuration makes no
named support assertion.

## Options considered

### Option A: Keep one provider-wide scope

Rejected because protocol and API mode are properties of an execution surface, not canonical provider
identity. Any selected default becomes incorrect for another family or mode.

### Option B: Keep one scope and policy per registration, then create one registration per family

Rejected as the only representation because a host commonly wants one route to expose disjoint
families from the same configured runtime. Forcing a route per family makes lookup less ergonomic
without resolving alternative modes within one family.

### Option C: Add a universal capability map to `Provider`

Rejected because a manually synchronized capability matrix duplicates policy and support evidence,
cannot express request-dependent behavior, and encourages downcasts or provider matching.

### Option D: Require every provider to expose a support manifest

Rejected because support evidence is advisory introspection rather than the executable provider
contract. It would burden custom providers and encourage Registry to mistake dated claims for current
account capability.

### Option E: Use identity-only providers and per-family registration bindings

Chosen because each fact has one owner, composite providers remain honest, Registry stays small, and
direct and routed model handles retain the same exact descriptors.

## Consequences

### Positive

- Composite providers no longer publish an arbitrary provider-wide protocol identity.
- Registry evaluates and constructs each model family with the correct scope and policy.
- Policy evaluation cannot express an invalid family/operation combination.
- Hosts can narrow a combined registration to one family before assigning a route.
- Direct and routed model descriptors can be validated for exact provider, model, family, platform,
  protocol, and API mode parity.
- Support evidence remains inspectable without becoming a closed model or capability allowlist.
- Third-party providers implement only the narrow family traits and registrations they need.

### Costs

- `Provider::scope()` and `Provider::platform()` are removed.
- Registration scope access requires a `ModelFamily`.
- Custom registration code starts with a `from_*` family constructor and adds disjoint families with
  `bind_*`; an empty public registration is no longer constructible.
- Facade registration returns a typed error for provider configurations without a portable family.
- Alternative modes for one family require explicit registrations and, when routed together, distinct
  route IDs.

## Migration

1. Replace provider-wide scope queries with `provider.provider_id()` or a concrete model's descriptor.
2. Replace the global registration constructor and `with_*` factory methods with `from_*` and
   `bind_*` family bindings.
3. Query registration scope, platform, protocol, or API mode with an explicit `ModelFamily`.
4. Call policy or Registry evaluation with a `ModelOperation`; its family is derived automatically.
5. Merge only disjoint family registrations for the same provider, use `for_family` to narrow a
   combined registration, and keep same-family alternate modes as separate registrations.
6. Handle `RegisterProviderError::NoPortableFamilyRegistration` when a provider may be configured
   only for native resources or jobs.
7. Expose provider-owned profiles or manifests for named support claims, but do not use them as
   request-time capability gates.

## References

- `docs/architecture/overview.md`
- `docs/architecture/public-api.md`
- `docs/architecture/registry.md`
- `docs/adr/0010-provider-plane-and-host-control-plane.md`
- `docs/plans/2026-08-07-001-refactor-provider-faithful-semantic-revival-plan.md`
