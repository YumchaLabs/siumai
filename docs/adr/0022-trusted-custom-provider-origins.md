# ADR-0022: Trusted Custom Provider Origins

## Status

Accepted

## Date

2026-10-25

## Context

[Issue #33](https://github.com/YumchaLabs/siumai/issues/33) identified that the existing endpoint
policies cannot express a public self-hosted HTTP
service. They also require applications to classify ordinary developer-configured endpoints as
loopback, private, link-local, or shared address space. Transport security is valuable, but a
developer-authorized provider origin and a URL returned by an untrusted provider are different
trust inputs.

## Decision

Provide `EndpointConfig::custom(url)` backed by `EndpointPolicy::TrustedCustom`. Calling it
authorizes one exact developer-configured HTTP(S) origin on any network; it does not authorize a
class of request URLs or disable validation globally. Scheme, normalized host, effective port,
relative target confinement, credential audience, bounded I/O, and redirect rejection remain
transport-owned. Construction remains synchronous and network-free.

The host must never treat unvalidated tenant, request, tool, or model output as trusted endpoint
configuration. The hostname may resolve to any address, including link-local metadata ranges.
Applications that need destination restrictions retain the existing `PublicCustom` and exact
`LocalExplicit` policies. `Official` retains its audited HTTPS production-origin proof.

`OpenAiCompatibleProfile::custom(...)` exposes this path without inventing named-provider
fidelity, model support, a replay domain, credentials, or an API mode.

```text
trusted configured origin -- exact audience --> provider execution
provider response URL     -- independent guard --> resource download
```

Resource URLs cannot inherit `TrustedCustom`; a configured provider returning another local URL
does not authorize fetching it. Proxy endpoints retain their own explicit policy allowlist, and
CONNECT destinations still require a policy that enforces public HTTPS. Explicit WebSocket
configuration may use the shared trusted-origin policy, but only with `ws`/`wss` and its own
credential audience; an HTTP endpoint does not infer or create a session endpoint. MCP retains its
independent endpoint policy.

## Alternatives

- Documentation alone preserves the public-HTTP gap and ordinary configuration burden.
- A global `allow_http` or unchecked HTTP client override weakens unrelated destinations and
  obscures credential/resource ownership.
- An orthogonal matrix of TLS and address-grant flags adds more configuration than this use case
  needs. A single explicit trust decision plus existing restrictive policies is smaller.

## Consequences and verification

This is an additive API; existing constructors keep their behavior. Prefer the new path for
trusted self-hosted application configuration, not for a multi-tenant endpoint selector. HTTP can
carry provider credentials only by the caller's explicit configuration and remains cleartext.

Offline tests must prove public/private/loopback HTTP construction, real authenticated direct
execution with a fixed resolver, unchanged restrictive-policy rejection, audience confinement,
redirect rejection, and independent resource/proxy/WebSocket guards. The self-hosted example must
compile and run without a server or provider call. This decision amends the endpoint section of
the current transport contract without changing resource-download or forward-proxy authority.
