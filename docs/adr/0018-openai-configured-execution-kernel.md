# ADR-0018: OpenAI Configured Execution Kernel

## Status

Accepted

## Date

2026-08-16

## Context

The configured OpenAI-compatible engine and the official OpenAI provider previously duplicated
stateless request-plan construction, HTTP execution, bounded provider-error capture, SSE byte
framing, terminal ordering, unexpected-EOF handling, and stream cancellation. The existing
`extension::v1` codec policies could reuse request and response semantics, but their portable
language output erased the official provider's native Responses resource and stream-frame carriers.

Wrapping the official provider in `OpenAiCompatibleProvider` would remove meaningful ownership:
official credentials, typed options, annotations, native resources, replay status, Realtime,
Responses WebSocket, and support evidence are not generic compatibility concerns. Keeping two
execution loops, however, made security and lifecycle fixes easy to apply to one route but miss the
other.

Reference SDKs demonstrate the value of small shared request and stream helpers, but their dynamic
provider module layouts are not an appropriate Rust ownership model. Official provider documents
and Siumai's provider-specific fixtures remain the wire authority.

## Decision

`siumai-openai-compatible::extension::v2` is the semver-covered provider-author contract for one
stateless OpenAI-family HTTP/SSE execution kernel.

A provider prepares and validates:

- the selected `ProviderTransport`;
- a relative request target and non-credential headers;
- a JSON body bounded against that transport before it is retained;
- replay safety, warnings, and sanitized error context;
- a direct decoder or SSE decoder whose associated output preserves provider-owned data.

The kernel alone constructs the immutable transport request plan and owns HTTP execution, bounded
non-success capture, OpenAI error metadata classification, SSE framing, terminal-in-batch
discipline, explicit decoder finish, unexpected EOF, contextualized setup errors, and child
cancellation on stream drop.

The kernel does not select or construct credentials, endpoints, signing, retry, timeouts, provider
identity, replay domains, support evidence, options, annotations, dialects, resources, or native
wire semantics. Decoder hooks cannot mutate transport authority.

The official OpenAI provider depends downward on this kernel for stateless Chat Completions and
Responses direct and SSE calls. It remains a branded provider rather than a wrapper around the
generic compatible provider. Its request preparation and native result/frame decoding stay in
`siumai-provider-openai`; background Responses, resources, Realtime, and Responses WebSocket do not
move into the kernel. The facade `openai` feature does not re-export or activate the facade
`openai-compatible` feature.

`extension::v1` remains available for current branded compatible codec policies. Migration to v2
is incremental and justified only when a consumer needs the lower execution seam.

## Options considered

### Option A: Keep separate official and compatible execution loops

Rejected. The duplicated transport and stream lifecycle is security-sensitive and has no branded
semantic value.

### Option B: Build official OpenAI through `OpenAiCompatibleProvider`

Rejected. It would erase native fidelity and transfer official resources, evidence, identity, and
session semantics to the wrong owner.

### Option C: Generalize a workspace-wide provider execution trait

Rejected. Other protocols do not yet demonstrate the same request, error, framing, and terminal
contract. A universal trait would encode speculation and weaken provider ownership.

### Option D: Share one narrow stateless kernel with provider-owned associated outputs

Chosen. It removes duplicate mechanics without collapsing branded preparation or native results.

## Consequences

### Positive

- Compatible and official OpenAI-family calls receive the same bounded error and SSE lifecycle
  fixes.
- Official native Responses results and frames remain lossless and paired with one provider call.
- The provider-author seam is usable without access to private workspace types.
- Authentication, endpoints, replay, retry, and timeout authority remain in transport and provider
  construction rather than decoder hooks.

### Costs

- `siumai-provider-openai` gains a direct dependency on `siumai-openai-compatible`.
- Provider adapters must explicitly translate their prepared request and native decoder into the
  v2 contract.
- The v1 and v2 extension seams coexist until existing branded compatibility profiles have a real
  reason to migrate.

## Verification

Kernel fixtures cover bounded JSON preparation, bounded sensitive provider errors, diagnostics,
fragmentation, terminal plus `[DONE]`, duplicate or trailing terminal data, per-event cancellation,
unexpected EOF, and drop cleanup. Official-route fixtures cover Chat direct and usage-only streaming,
Responses direct/native preservation, native-to-portable projection, buffered-frame cancellation,
and unexpected EOF. Compatible-route fixtures cover their configured direct and SSE adapters.
Feature checks verify that official OpenAI does not activate the facade compatible-provider feature.

## References

- `docs/adr/0015-validation-ownership-and-forward-compatibility.md`
- `docs/architecture/overview.md`
- `docs/migration/siumai-next.md`
- `docs/plans/2026-08-15-2359-refactor-deepen-runtime-and-openai-lifecycles-plan.md`
