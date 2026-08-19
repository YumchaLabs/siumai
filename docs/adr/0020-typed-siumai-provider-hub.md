# ADR-0020: Typed Siumai Provider Hub

## Status

Accepted

## Date

2026-08-19

## Context

Siumai supports two complementary public paths: provider-owned APIs for exact native capabilities
and provider-neutral family traits for portable application code. ADR-0019 correctly established
the six root family modules as the canonical execution seam and rejected a capability-erased
universal client. It also rejected a `Siumai` entry point because the considered design combined
provider construction, model selection, capability discovery, and execution in one dynamic client.

That rejection treated every `Siumai` facade as if it had to be a universal `LlmClient`. It left a
real ergonomics gap: changing a directly configured provider still requires callers to learn each
provider constructor before reaching the otherwise portable family API. The historical
`Siumai::builder().openai().api_key(...)` outline made provider selection easy to discover, and the
desired experience can be restored without reviving the old implementation model.

The existing architecture now has the necessary boundaries: configured providers implement small
family-provider traits, family models carry exact identity and mode, root family modules own call
execution, Registry owns explicit runtime erasure, and provider crates own credentials, builders,
native resources, and configuration errors. A typed facade can compose those owners without
duplicating them.

## Decision

### Typed provider construction

The `siumai` facade exposes `Siumai::builder()` on one concrete zero-state entry type. Provider
features add zero-argument selectors such as `.openai()`, `.anthropic()`, and `.gemini()`. A
selector enters provider-specific required-input stages; it does not construct a universal
provider enum or store optional credentials.

The public quickstart keeps the historical shape:

```rust
let hub = Siumai::builder()
    .openai()
    .api_key(api_key)
    .build()?;
```

Credential transitions construct the provider-owned credential and real provider builder. A
provider-owned credential can be supplied through `.credential(...)`. Constructor inputs such as
a compatibility profile, project, location, or deployment are separate typed stages in the order
required by the real builder. `.build()` is available only after the documented stage sequence is
complete and returns the provider-owned configuration error synchronously without network I/O.

Advanced optional configuration remains on the real provider builder behind one consuming
`.configure_provider(...)` transition. The facade does not mirror provider endpoint, retry,
transport, default-option, project, region, or resource setters.

The provider-shape audit establishes an important distinction:

- OpenAI needs one `OpenAiCredential` to create `OpenAiProviderBuilder`.
- OpenAI-compatible providers need an `OpenAiCompatibleProfile` and
  `OpenAiCompatibleCredential`.
- Google Vertex Anthropic needs project, location, and `GoogleVertexCredential`.
- ElevenLabs needs an `ElevenLabsProfile` and `ElevenLabsCredential`.
- Alibaba creates `AlibabaProviderBuilder` from `AlibabaCredential`, but that builder cannot build
  a useful provider until at least one endpoint is selected. Its facade chain therefore requires
  `.configure_provider(...)` after the credential transition and before `.build()`. The closure
  operates on the real Alibaba builder, and `AlibabaConfigError` remains the authority that checks
  whether the selected endpoint configuration is semantically complete.

Facade stages do not discover environment credentials, retain a second plaintext secret, define a
universal missing-credential error, or add an argument-taking selector alias such as
`.openai(credential)`.

### Configured-provider hub

`Siumai<P>` represents one long-lived concrete configured provider. It contains no model slots,
capability matrix, route table, provider enum, `Any`, or downcast support. `Siumai::from_provider`
wraps an already configured provider without rebuilding it.

The hub exposes only family selectors supported by `P`'s existing provider traits. For example,
`.language(model)` returns a `LanguageClient<P, M>` and `.embedding(model)` returns an
`EmbeddingClient<P, M>`. Multiple clients can be bound from one hub without reconstructing the
provider. Model identifiers remain open, and binding is synchronous and network-free.

Each family client retains the same configured provider instance and the exact model produced by
that provider. It implements the corresponding existing family model trait, delegates descriptor
and route identity to the inner model, and can be passed directly to generic code. The facade does
not offer a public constructor that can pair an arbitrary unrelated provider and model.

### Call ownership and native access

Family clients provide method-style portable operations such as `.generate(...)`, `.embed(...)`,
and `.transcribe(...)`. These are shallow ergonomic delegates to the existing root family modules
and family call builders. The facade does not assemble requests independently, validate provider
options, dispatch transport calls, retry, collect streams, normalize usage, or define replacement
response and error types.

`Siumai<P>::provider()` exposes `&P`. Every family client exposes the same concrete provider through
`provider()` and its concrete model through `model()`. Provider-native files, batches, catalogs,
sessions, hosted tools, media jobs, Realtime, WebSocket, native responses, and model-specific
methods remain reachable through those typed owners rather than portable facade methods or a
native-capability enum.

### API modes, Registry, and erasure

A provider hub may define one documented canonical selector per family. Alternate API modes use
explicit methods on the concrete hub or the provider-owned API. OpenAI `.language(model)` means
Responses and `.chat_completions(model)` is the explicit alternative. Gemini `.language(model)`
means Interactions and `.generate_content(model)` is the explicit alternative. No selector infers
mode or capability from model-name patterns.

Registry remains the explicit family-specific runtime-erasure boundary. The builder does not
register providers, resolve aliases, inspect global state, or choose business routes. Once callers
erase a model behind a family trait object or Registry handle, they intentionally give up concrete
provider-native access; the facade does not recover it through downcasting.

### Compatibility boundary

This ADR supersedes ADR-0019's conclusion that Siumai must not expose a `Siumai` entry point. It
retains ADR-0019's accepted root family modules, family-specific call builders, complete response
types, exact-target provider options, Registry boundary, and rejection of a universal client.

The documented selector chains, family binding names, operation names, typed provider/model
accessors, and explicit alternate-mode selectors form the facade compatibility boundary. Public
intermediate stage names are incidental return types; callers construct them only through the
documented chain. No historical spelling aliases are added during this beta refactor.

## Options considered

### Option A: Keep root family modules as the only primary journey

Rejected. The modules remain the correct generic execution seam, but they do not provide one
discoverable place to select and configure a direct provider. Requiring every application to start
from provider-specific constructors leaves provider switching unnecessarily ceremonial.

### Option B: Restore the historical universal `LlmClient`

Rejected. A universal client would need provider enums, capability probing, model slots, error and
response normalization, hidden routing, or downcasts. It would compete with provider builders,
family traits, Registry, and native provider APIs.

### Option C: Add a dynamically typed `Siumai` wrapper

Rejected. Erasing the provider at construction would make provider-native resources and exact
configured-instance identity inaccessible or require an escape mechanism based on `Any`, strings,
or enums. Runtime erasure already has an explicit owner in Registry and family trait objects.

### Option D: Use credential-as-argument provider selectors

Rejected. `.openai(OpenAiCredential::api_key(key))` is compact but loses the historical
zero-argument selector rhythm, makes heterogeneous multi-input provider construction less
discoverable, and prevents the type state from guiding callers through required inputs one step at
a time.

### Option E: Add a typed provider hub and family clients

Chosen. It restores the direct-provider experience while making invalid construction states
unrepresentable, retaining concrete native access, and reusing the current execution boundaries.

## Consequences

### Positive

- Direct applications can switch providers while keeping the same portable family call shape.
- Required provider inputs are discoverable through type-directed stages instead of optional
  fields and late universal errors.
- One configured provider can create multiple exact family/model clients without rebuilding.
- Generic code continues to accept ordinary family traits rather than facade-specific traits.
- Provider-native capabilities remain directly available from concrete provider and model types.
- Root family modules, Registry, provider builders, transport, and runtime retain one clear owner
  each.

### Costs

- Each built-in provider feature needs a small facade adapter that reflects its real construction
  shape and supported family traits.
- Public type signatures include provider, model, and sometimes stage types; Rust error messages
  may be more verbose than a dynamic universal client.
- Providers with multi-step or caller-selected technical addressing need additional typed stages
  or a required real-builder configuration transition.
- The beta migration is breaking: callers must replace both the historical universal client and
  the current root-family-only direct journey with the final typed facade where appropriate.

## Verification

The facade has a focused `siumai_builder_contract` target registered explicitly because the crate
disables automatic test discovery. Its no-default lane proves that the zero-state builder exists;
provider-feature lanes prove the zero-argument selector, credential and provider-owned credential
paths, reusable multi-family hub, generic family acceptance, and concrete provider/model access.
Negative method-availability tests are added only after their positive surfaces exist so a missing
API cannot produce a false success.

Provider adapter tests verify the required stage order, real builder and configuration error,
canonical and alternate modes, exact configured-instance identity, and sanitized credential/error
diagnostics. Root family, Registry, provider fixture, rustdoc, feature, and migration gates continue
to verify their existing ownership contracts without a custom API-policy parser.

## References

- [ADR-0013](0013-provider-identity-and-family-registration.md)
- [ADR-0015](0015-validation-ownership-and-forward-compatibility.md)
- [ADR-0019](0019-facade-family-call-ownership.md)
- `docs/architecture/public-api.md`
- `docs/migration/siumai-next.md`
- `docs/plans/2026-08-19-1253-refactor-typed-siumai-facade-plan.md`
