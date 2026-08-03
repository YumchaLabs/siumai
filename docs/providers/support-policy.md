# Provider Support Policy

This document defines what Siumai means when it claims support for a provider. It
does not list model IDs; provider-owned catalogs carry dated lifecycle advice and
unknown IDs remain valid input.

## Claim scope

Every claim is scoped by:

```text
provider + platform + family + api_mode [+ region/deployment]
```

A broad provider name alone is not a support claim. For example, a native language
implementation does not imply native image or realtime support, and a cloud-hosted
deployment may differ from the provider's first-party platform.

## Fidelity

| Value | Meaning |
|---|---|
| `native` | Siumai implements the provider's native control plane and preserves its relevant wire semantics. |
| `verified-compatible` | Siumai uses a compatibility protocol with a named, tested dialect profile for this provider and scope. |
| `generic-compatible` | Users may supply an endpoint and credentials through a generic compatibility builder; Siumai makes no named provider claim. |

## Public stability

| Value | Meaning |
|---|---|
| `stable` | The Siumai family contract is supported by the normal semver policy. |
| `experimental` | The protocol may be native, but the Siumai session/job/stream contract may change in a breaking release. |

Fidelity and stability are independent. Realtime can be both `native` and
`experimental`.

## Evidence for named profiles

A built-in `native` or `verified-compatible` claim must include:

1. An official provider documentation URL.
2. A `verified_at` date and platform/API-mode scope.
3. A provider-owned policy for meaningful dialect differences.
4. Offline success, error, and applicable stream fixtures.
5. Model lifecycle entries only where an official source exists.

Lifecycle entries use `active`, `deprecated`, `retired`, or `rolling-alias`, and may
name a replacement. They are advisory. CI validates their structure and staleness;
it does not scrape websites or generate Rust from another SDK's model union.

## Generic compatibility

The generic OpenAI-compatible builder remains an escape hatch for private gateways
and unlisted services. It requires an explicit endpoint policy and credential
audience, exposes only protocol-baseline behavior, and does not inherit a named
provider's fidelity claim.

## Maintenance workflow

Provider changes are audited against official documentation first. The local
Vercel AI SDK checkout is secondary evidence for fixtures and edge cases. A
maintainer updates the provider policy, lifecycle data, and behavior fixtures in
one change; a static mega-list of remote gateway models is not accepted.
