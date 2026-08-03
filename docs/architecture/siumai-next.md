# Siumai Next Architecture

- Status: Accepted
- Date: 2026-08-04
- Scope: next breaking release
- Supersedes: the ADR, alignment, and workstream corpus that predates this decision
- Implementation plan: `docs/plans/2026-08-04-001-refactor-siumai-next-revival-plan.md`

## Context

Siumai's unified entry point is useful, but its current implementation makes one
generic client responsible for unrelated model families, provider construction,
capability discovery, transport policy, routing, and compatibility. That shape
creates downcasts, duplicated protocol runtimes, model caches that hide expensive
construction, and tests that protect source layout instead of user-visible behavior.

Siumai Next keeps the ergonomic unified experience while rebuilding the layers
under it around Rust traits with narrow ownership. This is an intentional breaking
release. Compatibility shims are temporary migration state and must not survive the
refactor.

## Decision

### Stable model families

The stable provider-neutral surface contains exactly six invocation families:

| Family | Stable primitive |
|---|---|
| Language | one generate call or one established event stream |
| Embedding | one provider request containing one or more inputs |
| Rerank | one query and one candidate set |
| Image | one final-result generation call |
| Speech | one final-result synthesis call |
| Transcription | one final-result transcription call |

Realtime/Live sessions, streaming transcription, streaming translation, video,
and asynchronous media jobs remain experimental. Files, batches, assistants,
skills, hosted applications, and provider catalogs are provider resources rather
than model families.

### Unified facade

The `siumai` crate remains the ergonomic facade. It may construct configured
providers, assemble registrations, resolve routes, and call provider-neutral
runtime helpers. It does not define a universal client or a capability bag.

Direct and routed use terminate at the same family traits:

```rust,ignore
let provider = OpenAi::builder().api_key(key).build()?;
let direct = provider.responses("gpt-5.6");

let registry = Registry::builder()
    .register("primary", provider.registration())?
    .build();
let routed = registry.language_model("primary:gpt-5.6")?;
```

Configured providers own authentication, endpoint policy, transport, retry policy,
headers, and provider resources in an `Arc`-backed runtime. Model construction is
synchronous and cheap. Token refresh and other asynchronous credential work happen
when a request is executed.

### Public contracts

The following sketch fixes responsibilities, not final field spelling:

```rust,ignore
pub trait Model: Send + Sync {
    fn provider_id(&self) -> &ProviderId;
    fn model_id(&self) -> &ModelId;
    fn family(&self) -> ModelFamily;
}

#[async_trait]
pub trait LanguageModel: Model {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, Error>;

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error>;
}
```

`async_trait` is the single object-erasure boundary. Streams use one boxed
`Send + 'static` carrier. Concrete providers may expose concrete models and
inherent extension methods; Registry registrations expose only narrow family
constructors.

Provider options are typed and provider-owned. Common requests contain only
semantics shared by stable families. A checked raw JSON escape hatch is explicit,
namespaced, and must reject protected transport and authentication fields.

### Identity and routing

`ProviderId`, `RouteId`, and `ModelId` are distinct open identifiers:

- `ProviderId` identifies canonical provider ownership.
- `RouteId` identifies one configured provider instance and its default API mode.
- `ModelId` is opaque provider input and may contain `:`.

Registry references use `route:model`; parsing splits only at the first colon.
Registry snapshots are immutable. Replacing a route creates a new snapshot, while
previously resolved model handles keep their captured provider runtime.

### Package ownership

The target dependency direction is:

```text
siumai facade  -> registry, runtime, provider crates
server / MCP   -> runtime, core
runtime        -> core
registry       -> core
provider-*     -> core, transport, protocol-*
protocol-*     -> core
transport      -> core
core           -> third-party data/async primitives only
```

The concrete package rules are:

1. `siumai-core` owns durable neutral data, errors, family traits, lifecycle types,
   policy vocabulary, tool contracts, and call options.
2. `siumai-transport` owns HTTP/WebSocket clients, endpoint safety, retry,
   observability sanitization, framing, limits, backpressure, and cancellation.
3. `siumai-protocol-*` owns wire DTOs, codecs, and stateful decoders.
4. `siumai-provider-*` owns configured runtimes, concrete models, model policy,
   protocol selection, provider resources, and model lifecycle data.
5. `siumai-registry` depends on core only. Built-in registrations are assembled by
   the facade, never by Registry.
6. `siumai-runtime` owns generation helpers, structured output, history projection,
   tool loops, approvals, snapshots, and run budgets.
7. MCP, server, and bridge packages are optional integrations above runtime and
   protocol layers.

`config/architecture/dependency-policy.json` is the machine-readable dependency
ratchet. Transitional allowances name the unit that removes them. The strict target
policy becomes the release gate after the new vertical slices replace legacy paths.

### Provider support claims

Support is described by two independent dimensions for a scoped
`{provider, platform, family, api_mode}` tuple:

- Fidelity: `native`, `verified-compatible`, or `generic-compatible`.
- Public stability: `stable` or `experimental`.

Native protocol fidelity can still expose an experimental Siumai API. A named
built-in profile requires an official source, verification date, typed dialect
policy where behavior differs, and offline request/response/stream/error fixtures.
Unknown model IDs remain usable; lifecycle catalogs are advisory rather than
closed Rust enums. See `docs/providers/support-policy.md`.

### Stream lifecycle

`stream()` returns outer errors only before establishment. Once established, a
stream emits exactly one terminal event: `Completed`, `Failed`, or `Cancelled`.
Protocol EOF without a terminal event becomes `Failed(UnexpectedEof)`. Consumer
drop requests cancellation but cannot fabricate an event for a dropped consumer.

Protocol decoders are stateful and protocol-specific. Each consumes raw events,
emits zero or more canonical events, and has exactly one finish path. Shared
transport framing is not a universal provider decoder.

### Runtime and tool boundary

Plain `generate` and `stream` perform one model call and never execute local code.
Only an explicit tool-loop API may execute a local `ToolBinding`. Model-visible
`ToolSpec`, execution ownership, authenticated trust context, one-time approval,
execution receipts, and snapshots are separate concepts.

Side-effecting execution records `Prepared`, `Dispatched`, `Completed`, or
`Indeterminate`. A crash after dispatch but before a completed checkpoint never
silently replays work. The library does not claim general exactly-once execution.

### Deletion policy

Replacement code and behavioral evidence land before the path they replace is
deleted. Once replaced, the following are removed rather than deprecated forever:

- `LlmClient`, `ClientWrapper`, capability bags/downcasts, and compatibility facets;
- completion as a stable family and fake version-marker traits;
- Registry provider factories, provider settings, model caches, and central model catalogs;
- duplicate transport, decoder, structured-output, and tool-loop implementations;
- mirrored AI SDK public namespaces and generated static model mega-lists;
- source-scanning tests, stale examples, superseded ADRs, alignment inventories,
  and completed workstream journals.

Wire fixtures are not deleted merely because their harness uses old types. They are
retained, reclassified, and migrated unless an explicit retired-behavior record
explains why the behavior no longer exists.

## Rust and verification baseline

The workspace uses Rust 2024 and Cargo resolver 3. The initial 1.85 probe failed
because the existing source intentionally uses let chains and const library methods
that stabilized later across a large part of the workspace. Mechanical rewrites in
code scheduled for deletion would add risk without product value. Rust 1.88 is the
declared MSRV and has been verified against the initial core/spec surface. The full
workspace MSRV lane becomes mandatory before release; lowering the MSRV remains
possible if the final reduced source and retained dependencies support it naturally.

Tests prove observable contracts:

- public usage through compiled examples and external fake implementations;
- dependency direction through `cargo metadata` JSON;
- protocol behavior through offline wire fixtures and state-machine tests;
- provider behavior through shared family contracts and provider-specific fixtures;
- feature isolation through explicit Cargo build lanes.

Tests must not enforce method order, exact source strings, module filenames, or
parity with a TypeScript directory tree.

## Consequences

The next release is source-incompatible and requires a migration guide. The result
has fewer stable concepts, keeps provider-specific behavior honest, permits current
and future model IDs without releases, and preserves the convenience that makes a
unified library valuable.
