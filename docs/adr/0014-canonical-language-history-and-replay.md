# ADR-0014: Canonical Language History and Provider Replay Domains

## Status

Accepted

## Date

2026-08-08

## Context

Language providers expose three related but different representations:

- provider wire input and output;
- portable request and response semantics;
- durable conversation history used by an agent or runtime.

The former language boundary did not enforce one representation for executable tool input. An
OpenAI-compatible function argument could appear as an encoded JSON string in one path and as a
parsed JSON value in another. Provider-executed tool activity could also enter the portable runtime
tool loop even though the host was not its execution owner.

Request and response content shared one carrier despite having different role and direction rules.
Response-only citations and refusals could consequently be copied into an invalid assistant request
message. Provider-native output needed for exact continuation was retained as opaque content, but
its provenance did not distinguish separate custom endpoints, accounts, workspaces, projects, or
deployments that happened to use the same protocol.

These ambiguities forced every downstream agent SDK to normalize tool arguments, reconstruct valid
history, and decide whether provider-native state was safe to replay.

## Decision

Siumai keeps one canonical provider-neutral language contract and makes its construction,
projection, and replay invariants explicit.

### Portable executable tool calls

`ToolCall` represents only a caller-executed portable function call. Its input is one checked
`ToolInput` containing the parsed semantic JSON value. A protocol decoder parses encoded argument
text exactly once before constructing the call. The value may be an object, array, scalar, or JSON
string when that string is the actual tool input; an encoded object is never retained as an
ambiguous semantic string.

Tool call identity, name, and encoded input size are bounded. Fields are private, construction and
deserialization use the same validation path, and `Debug` does not expose semantic input. A
provider-executed program, custom-text call, hosted tool, MCP operation, computer action, or
otherwise provider-owned operation remains on the provider-native output surface or in bounded
`ProviderOpaque` replay data. It cannot be constructed as a portable `ToolCall`. A structured local
function call issued by a provider-hosted program remains executable by the caller; its portable
call is paired with native replay metadata that preserves the program caller linkage.

Malformed or oversized local tool JSON is a typed protocol/input failure before runtime execution.
Provider-native raw input remains opaque because parsing it would invent caller-execution semantics.

### Direction-aware messages and history projection

`Message` remains the single provider-neutral request carrier, but role/content combinations are
validated before encoding. Role-safe constructors are the preferred public path. Response-only
content does not become valid request content merely because both directions use `ContentPart`.

`LanguageResponse::project_assistant_history()` is the canonical response-to-request transition. It
preserves assistant-replayable text, reasoning, media, local tool calls, and provider-native opaque
items. It reports citations, refusals, and response-side tool results as structured omissions rather
than silently inserting invalid request parts. The projection may contain no assistant message when
the response contains no replayable content.

Runtime uses this projection for tool loops, structured-output repair, reports, and durable
snapshots. Omission records are part of the durable step record so projection loss remains
inspectable.

### Explicit replay domains

`ProviderScope` owns an optional `ReplayDomain` in addition to provider, platform, protocol, and API
mode. A domain contains:

- an `Official` or `Custom` audience with a stable non-secret identifier;
- an optional non-secret caller scope for a material account, workspace, project, or deployment
  boundary.

Two scopes may replay provider-native state only when provider, platform, protocol, API mode, and
the complete replay domain match. Missing domains always fail closed. Model IDs and Registry route
IDs do not participate: a provider protocol may allow continuation across models, and a route is a
host alias rather than provider replay identity. A protocol may impose an additional model equality
rule when its wire contract requires one.

`ProviderProvenance` is checked, immutable, and requires both a protocol and a replay domain.
Deserialization uses the same constructor, so persisted data cannot bypass these requirements.

Official provider builders supply a provider-owned audience when the account boundary is not
material or is unknown. Callers using multiple accounts on that audience must set distinct caller
scopes. Providers that already receive a material non-secret account coordinate may require an
explicit caller scope. Custom endpoints always require a caller-declared custom audience. Endpoint
ownership and replay audience kind must agree; transport endpoint policy does not prove provider
ownership.

Replay identity is never derived from a URL, hostname, credential, authorization header, signed
value, private provider payload, region catalog, or mutable availability metadata. Technical
project, workspace, location, and deployment inputs still belong to provider builders when required
for addressing or signing, but they are not serialized into replay identity unless the caller
explicitly supplies a suitable non-secret label.

## Options considered

### Option A: Keep permissive core values and normalize in every consumer

Rejected because every runtime would need its own JSON normalization, role repair, replay matching,
redaction, and parity rules. Different consumers would execute the same provider output
differently.

### Option B: Expose only protocol-native language APIs

Rejected as the primary surface because it removes Siumai's portable value and makes shared runtime,
Registry, middleware, and conformance behavior provider-specific. Native APIs remain first-class
alongside the canonical projection.

### Option C: Derive replay identity from endpoint or credentials

Rejected because URLs and credentials may be sensitive, mutable, signed, or unstable. Hashing does
not repair the ownership problem and would make durable data depend on secret rotation or endpoint
spelling.

### Option D: Bind replay to Registry route and model ID

Rejected because routes are host-controlled aliases and model IDs are not universally part of a
provider's replay contract. Both would reject safe continuations without preventing same-route
account changes.

### Option E: Use checked canonical values and explicit replay domains

Chosen because the lowest layer with enough information enforces each invariant once, while native
wire fidelity remains available without entering the portable execution loop.

## Consequences

### Positive

- Direct, incremental, and terminal local tool calls share one parsed JSON representation.
- Provider-owned tool activity cannot be executed accidentally by the portable runtime.
- Invalid request roles fail before transport, and response-to-history loss is explicit.
- Provider-native replay fails before network submission when endpoint or caller domains differ.
- Custom endpoints remain usable without storing URLs or credentials in durable state.
- Runtime snapshots contain the execution scope and projection omissions needed for an auditable
  continuation.

### Costs

- `ToolCall` fields are private and provider-owned construction is removed.
- Code that built messages with arbitrary role/content combinations must use role-safe constructors
  or handle validation errors.
- Custom compatible endpoints must declare a `ReplayDomain::custom(...)`.
- Multi-account official configurations must provide distinct non-secret caller scopes when opaque
  replay could cross those accounts.
- Runtime snapshot schema version 5 is intentionally incompatible with earlier development
  snapshots.

## Migration

1. Construct caller-executed calls with `ToolCall::local` and read their input through accessors.
2. Use role-safe `Message` constructors and call `LanguageRequest::validate()` at custom boundaries.
3. Use `LanguageResponse::project_assistant_history()` instead of copying response parts directly.
4. Give every custom endpoint an explicit custom replay domain; add caller scope where account or
   deployment separation is material.
5. Recreate pre-version-5 runtime snapshots rather than rewriting provider provenance manually.

## References

- `docs/architecture/overview.md`
- `docs/migration/siumai-next.md`
- `docs/adr/0010-provider-plane-and-host-control-plane.md`
- `docs/adr/0013-provider-identity-and-family-registration.md`
- `docs/plans/2026-08-07-001-refactor-provider-faithful-semantic-revival-plan.md`
