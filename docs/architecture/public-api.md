# Public API and Extension Policy

- Status: Current repository contract
- Updated: 2026-08-06

## Public entry points

Applications may depend directly on an owning crate or use the `siumai` facade:

- direct provider crates are the authoritative surface for provider construction, protocol modes,
  typed options, metadata, and native resources;
- `siumai::providers::*` contains curated provider namespaces rather than blanket crate mirrors;
- `siumai::prelude::*` exports the provider-neutral family contracts used by application code;
- `siumai::registry` and `siumai::runtime` are optional facade integrations;
- protocol codecs, transports, MCP, and server adapters remain available from their owning
  packages instead of being relayed through broad facade namespaces.

The facade does not expose a universal client or provider capability downcasts. Direct and routed
models implement the same family traits, so an application can choose provider fidelity, local
routing, or both without maintaining two execution APIs.

## Provider construction

Provider builders configure credentials, endpoint policy, retry and transport limits, and
provider-wide typed defaults. They do not select an application route, discover models, or perform
network I/O during construction.

Provider methods construct lightweight family models synchronously from open model IDs. When a
provider supports multiple language protocols, the provider exposes named constructors such as
`chat_completions(model)` or `responses(model)` and a documented `language(model)` default.

The base `Provider` trait exposes only `provider_id()`. Inspect a concrete model descriptor or an
explicit family registration when platform, protocol, or API-mode identity matters. Those values
describe one execution surface and cannot truthfully summarize a composite provider.

Dynamic registration starts with one `ProviderRegistration::from_*` family binding and may add
disjoint families with `bind_*`. Each binding owns its exact scope, model policy, and erased factory.
An empty registration is not constructible. `ModelOperation` determines which family policy is
evaluated, and `for_family` can narrow a combined registration for a route-level allowlist.
Alternative modes for one family use the provider's mode-specific registrations and distinct
Registry routes. Facade `register_provider` returns a typed error when a valid provider configuration
has only provider-native resources or jobs and therefore no portable family registration.

## Typed provider extensions

Provider-specific request behavior uses types owned by the provider package. A typed call option
declares its provider namespace, model family, and API mode, validates before type erasure, and is
attached through `CallOptions`. It configures one invocation and participates in the provider-owned
precedence and merge policy.

Provider behavior attached to one message, content part, or tool definition uses a typed durable
annotation stored beside that semantic node. Annotations have no precedence or recursive merge
algorithm. Foreign history annotations remain inert; a provider reads only its exact namespace,
API mode, and node target. Provider-native state required for faithful replay remains a
bounded provenance-bearing opaque item rather than an optional annotation.

Use a common request field only when its semantics are stable across providers. Examples of
provider-owned behavior include prompt-cache controls, reasoning modes, hosted search, MCP or code
execution tools, service tiers, provider-specific log probabilities, and resource references.

Provider metadata follows the same ownership rule. Shared response and usage fields remain neutral;
typed provider metadata views expose provider-specific details without moving commercial provider
types into core.

Provider-owned profiles or support manifests expose dated claims for named surfaces. They are
introspection and maintenance evidence, not capability gates: custom endpoints may carry generic or
empty evidence while still constructing models that the selected protocol can encode safely.

## Stability

The public contract has three practical levels:

1. stable family contracts in `siumai-core` and the curated facade/prelude;
2. provider-owned stable APIs for documented provider capabilities and resources;
3. explicitly named experimental modules for sessions, jobs, or capabilities whose lifecycle is
   not yet a stable family primitive.

Compatibility namespaces, old generic builders, protocol relays, and source-layout aliases are not
a stability tier. Breaking releases delete them after the replacement path and migration guidance
exist.

## Features

Facade provider features activate only the selected optional provider dependency. Provider package
features represent real compile-time behavior, such as an optional protocol or message capability;
empty relay features are removed.

Applications that need the full provider surface should depend on the provider crate directly. A
facade feature must not activate unrelated providers, protocols, or job/session integrations.

## Errors, streams, and cancellation

Family calls return the canonical error type with operation/provider/model context, sanitized
public diagnostics, retry hints, and bounded provider details where available. Credentials, signed
URLs, raw headers, and unbounded response bodies never appear in ordinary `Debug` or display output.
`ErrorKind::ContextWindowExceeded` and `ErrorKind::Unavailable` distinguish exact provider signals
that callers commonly handle differently from invalid input or an unknown provider failure. A
provider message is never inspected heuristically to infer either category.

Language streaming begins only after the provider stream is established. The stream owns its
transport resources and cancellation child; dropping it releases those resources. Consumers must
observe a terminal completed, failed, or cancelled event. Protocol/server encoders reject events
after terminal and do not manufacture success on unexpected EOF. In-band provider errors are failed
terminals carrying the same typed `Error` contract as setup failures; raw provider error JSON is not
a second high-level failure channel.

## Documentation rule

Examples and README snippets use only current public paths and declare their required features.
Named provider support claims include an official source, verification date, family/API-mode or
native-surface scope, fidelity, and stability through a provider-owned profile or support manifest.
Model constants are completion hints and exact known-policy evidence, not a closed union of every
remotely available model.
