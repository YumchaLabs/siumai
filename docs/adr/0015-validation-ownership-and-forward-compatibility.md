# ADR-0015: Validation Ownership and Forward-Compatible Provider Intent

## Status

Accepted

## Date

2026-08-10

## Context

Siumai must protect portable execution invariants without turning fast-moving provider product
metadata into a second, stale source of truth. The previous public seams mixed these two concerns:

- model catalogs and support policies could influence whether a model was constructible or whether
  an explicit option reached the wire;
- provider option merging exposed host-specific origins such as route, model, runtime step, and
  call to every provider implementation;
- generic raw-option validation rejected field names without knowing whether the concrete protocol
  could safely encode them;
- Responses settlement selected strictness from provider claims instead of separating wire dialect
  normalization from portable semantic reconciliation;
- portable language results represented provider failures and cancellations as successful response
  values in some paths.

These checks were individually well intentioned, but several belonged to volatile product policy,
host orchestration, or a lower technical layer. Keeping them in shared execution APIs made future
model identifiers and provider fields harder to use, duplicated provider behavior, and encouraged
callers to treat dated support evidence as an execution authority.

Siumai still needs strict validation for local tool execution, message roles, resource bounds,
transport authority, replay identity, stream settlement, and wire shapes. The decision is therefore
not to become permissive. It is to place each invariant in the lowest durable layer that has enough
information to enforce it completely.

## Decision

Siumai adopts a caller-first provider intent model with explicit validation ownership.

### Validation ownership

| Concern | Owner | Required behavior |
| --- | --- | --- |
| Portable value invariants, role safety, local tool ownership, canonical JSON input, and shared resource bounds | `siumai-core` | Reject invalid states independently of provider identity. |
| Provider wire schemas, event ordering, canonical decoding, dialect normalization, and stream settlement | `siumai-protocol-*` | Enforce technical protocol invariants and preserve bounded native observations. |
| Provider option relationships, supported API modes, feature-derived headers, signing inputs, and native resources | `siumai-provider-*` or the owning compatibility engine | Encode explicit intent or return a typed structural error before transport. |
| Endpoint safety, authentication authority, redirects, retries, replay safety, body/frame limits, and deadlines owned by network execution | `siumai-transport` | Prevent provider body options from changing transport authority. |
| Account choice, region availability, pricing, quota, compliance, aliases, lifecycle allowlists, health, routing, and fallback | Host application | Remain outside provider execution and Registry lookup. |
| Official sources, verification dates, model constants, lifecycle hints, fidelity, and support manifests | Provider-owned introspection and repository evidence | Inform maintainers and callers explicitly; never gate ordinary execution. |

A validation rule must not move upward merely because the higher layer is easier to modify. A rule
that depends on provider wire meaning stays with the codec or provider. A rule that depends on host
business policy stays with the host.

### Open model identifiers and explicit intent

Model identifiers remain open. A catalog-unknown, private, proxied, or future identifier uses the
protocol baseline when the request is structurally encodable. Dated model knowledge may offer
constants and explicit introspection, but it does not decide whether Registry can construct a model
or whether a call may be sent.

An explicitly selected typed provider option has one of two outcomes:

1. it is represented on the final provider wire; or
2. the owning provider returns a typed configuration error explaining the structural conflict.

It is never silently removed because the model name is absent from an allowlist or appears in a
lifecycle catalog. A model-dependent execution branch may remain only when current official wire
evidence proves that the model identity changes encoding or interpretation. Such a branch requires
paired fixtures for the known dialect and the unknown-model baseline. Product eligibility,
commercial availability, and lifecycle-only branches are deleted from execution.

Feature-derived technical requirements remain valid. For example, a beta header may be selected
because the request uses a beta feature, not because a model-name allowlist predicts that feature.

### Provider options and raw forward compatibility

Provider option merging consumes ordered, exact-target patches rather than public host-origin
labels. The host may still offer route, model, runtime-step, and call ergonomics, but those labels
are private assembly concerns. The provider owns defaults and provider-specific merge semantics.

Ordinary typed options infer their provider, family, and API-mode target. Routing-aware optional
patches are inert when their exact target is not selected. Sensitive typed options and all raw body
options bind to an opaque configured-provider instance identity so credentials or replay-sensitive
values cannot cross two configurations that share public labels.

Raw options are a bounded provider-body escape hatch, not a transport escape hatch. Core validates
their target envelope and resource budget. The owning provider or codec rejects canonical request
fields, protected fields, invalid shapes, and stable relationship violations before overlaying the
remaining value. Unknown fields and future string enum values remain representable. A codec without
an explicit reviewed raw-body policy rejects raw options.

Byte limits are stated at the boundary where they can be enforced. A bounded byte constructor may
reject before JSON materialization. A `serde_json::Value` convenience constructor can only perform
an early-abort accounting walk after caller allocation and must not claim otherwise.

### Responses dialects and semantic reconciliation

Provider claims and model catalogs do not select stream strictness. A provider-owned wire dialect
descriptor normalizes documented omissions, aliases, and identity fallbacks before one shared
semantic reconciler runs.

The reconciler remains strict for portable text/refusal semantics and caller-executed item kind,
ownership, identity, call ID, tool name, and canonical JSON input. Duplicate terminals,
events-after-terminal, malformed executable input, in-band failures, unexpected EOF, and duplicate
or backwards incremental ordering remain typed errors.

Provider-native phases, transient statuses, terminal sequence drift, optional metadata, and other
bookkeeping do not fail a portable result merely because two observations differ. Replay-required
native state is classified explicitly per item kind. A conflict makes exact native replay
unavailable; it is never silently merged into portable execution authority. Public diagnostics
contain bounded structural summaries only, while sensitive native data remains redacted and behind
an explicit native or sensitive-response boundary.

### Portable language settlement

A portable language response represents only completed or incomplete generation. Provider failure
and cancellation are typed call outcomes, not successful response statuses.

Before stream establishment, failure remains the outer setup error. After establishment, the
stream settles exactly once with a completed result, incomplete result, failure, or cancellation.
Unexpected EOF is a failure unless the protocol explicitly defines it as successful completion.

Direct and established-stream failures may carry the same bounded `PartialLanguageOutput`: only
observational text, reasoning, refusal content, and usage. It carries no tool execution authority,
provider metadata, or replay authority. Provider-native APIs remain the path for complete failed,
cancelled, queued, or in-progress resources.

Usage remains unknown when absent. Ordered usage observations declare whether they are snapshots or
deltas so runtime can replace or add them without double counting. Deadline ownership is explicit;
the owner that settles a timeout or cancellation also cancels the source and suppresses any later
provider terminal.

## Options considered

### Option A: Keep product policy and strict validation in shared execution APIs

Rejected because model catalogs, lifecycle data, and provider product surfaces change faster than
the library. Treating them as runtime truth blocks valid private or future models and makes support
evidence responsible for protocol behavior.

### Option B: Make provider options and responses globally permissive

Rejected because it would allow invalid local tool input, protected field overrides, ambiguous
stream settlement, and unsafe replay state to escape the layer that understands them.

### Option C: Copy the reference AI SDK option maps and model heuristics directly

Rejected because dynamic provider maps and warning-based omission fit its TypeScript API but do not
provide the typed, fail-closed invariants expected from Rust. Siumai uses the reference SDK for
concepts and fixtures, not as its public architecture.

### Option D: Use caller-first intent with layer-owned structural validation

Chosen because it keeps the unified API useful, preserves provider-native fidelity, and removes
volatile product policy without weakening portable execution, transport, or resource safety.

## Consequences

### Positive

- Future and private model identifiers remain usable without waiting for a Siumai catalog release.
- Explicit provider intent reaches the wire or fails locally with a typed structural error.
- Provider implementations no longer depend on host-specific option-origin types.
- Raw provider fields can evolve without gaining endpoint, authentication, or transport authority.
- Responses compatibility is explained by technical dialect normalization rather than dated
  support claims.
- Portable consumers receive one settlement and usage model across direct and streaming paths.
- Support profiles remain valuable evidence without becoming hidden runtime policy.

### Costs

- Removing `ModelPolicy`, public option origins, and legacy terminal states is a breaking API change.
- Every raw-consuming provider mode must own and test a small fail-closed body policy.
- Configured provider instances need opaque identities for sensitive option targeting.
- Protocol implementations must classify native replay fields and terminal dialect differences
  explicitly.
- Runtime snapshots using the previous terminal shape cannot resume and require a schema version
  increment during this beta.

## Migration and verification

1. Move lifecycle, commercial, and allowlist decisions into explicit host policy before Registry
   execution.
2. Replace `Registry::evaluate` and provider policy callbacks with direct family resolution plus
   optional provider-owned support introspection.
3. Insert typed options through target-inferred call builders; use exact optional targets only for
   routing-aware fallbacks and instance-bound raw options for future provider fields.
4. Handle direct language failure through the typed call failure contract and established-stream
   failure through its single terminal outcome.
5. Recreate pre-refactor runtime snapshots rather than attempting to infer the new terminal and
   option-target semantics.

Verification uses focused deterministic fixtures. Representative baselines already prove that an
explicit OpenAI `max` reasoning effort reaches both Responses and Chat wire for a private model ID,
without an implicit summary, and that Anthropic beta requirements derive from requested features
for an unknown model. Further units add targeted dialect, target-isolation, settlement, usage, and
deletion tests. No provider-by-model-by-option matrix or live credentialed test is required.

## References

- `docs/architecture/overview.md`
- `docs/adr/0010-provider-plane-and-host-control-plane.md`
- `docs/adr/0013-provider-identity-and-family-registration.md`
- `docs/adr/0014-canonical-language-history-and-replay.md`
- `docs/providers/support-policy.md`
- `docs/plans/2026-08-10-001-refactor-validation-ownership-and-forward-compatibility-plan.md`
