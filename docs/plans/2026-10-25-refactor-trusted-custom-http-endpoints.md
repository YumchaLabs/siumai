# Trusted custom HTTP endpoints

## Goal

Resolve issue #33 without making ordinary self-hosted provider configuration depend on IP-range
classification. Developer-configured HTTP(S) origins may be public, private, loopback, shared, or
link-local. This authorization does not transfer to URLs returned by the provider.

## Design

- `EndpointConfig::custom(url)` selects `EndpointPolicy::TrustedCustom` for one exact origin.
- The host owns the trust decision for this configuration. It must not pass unvalidated tenant,
  tool, model, or request-supplied URLs into this constructor.
- Keep `Official`, `PublicCustom`, and `LocalExplicit` as unchanged restrictive policies. Do not
  implement a global switch, address auto-detection, or a new policy-combination framework.
- Preserve URL shape validation, exact credential audience, relative request targets, direct
  connector ownership, no redirects, admission, retries, and bounded I/O.
- `ResourceUrl` cannot inherit `TrustedCustom`; its existing public/local rules remain intact.
- CONNECT proxy configuration and destinations remain public-HTTPS/local-proxy scoped as before.
  A trusted custom provider origin is not automatically eligible for CONNECT.
- Shared WebSocket endpoint validation must still accept only `ws`/`wss` under an explicitly
  selected trusted policy. Existing WebSocket and MCP constructors do not change.
- `OpenAiCompatibleProfile::custom(...)` is the ordinary generic HTTP(S) entry; explicit replay
  identity, credentials, and API mode remain provider-owned inputs.

```text
trusted application configuration -> EndpointConfig::custom -> provider HTTP execution
untrusted response URL           -> ResourceUrl             -> isolated guarded downloader
```

## Alternatives

1. Keep restrictive constructors and add documentation: smallest change, but still cannot express
   a public HTTP endpoint and keeps network categories on the ordinary application path.
2. Add a global `allow_http` or disable validation: confuses endpoint authorization with resource
   trust and weakens unrelated requests. Rejected.
3. Add one trusted-origin construction path and retain explicit restrictive policies: selected.
   Less application-facing configuration, with an explicit host-owned trust obligation.

## Work and ownership

- Main: transport policy, URL/DNS/peer tests, proxy/resource/WebSocket guards, transport integration
  tests, ADR/current architecture/migration documentation, combined review and serial validation.
- Compatible-provider worker: generic profile convenience, facade credential convenience, focused
  provider/facade contracts, self-hosted guide and offline example, root/facade README links.
- Read-only security reviewer: check trust propagation and the negative boundaries.

Prior model-advisory changes in this checkout remain out of scope and must be preserved.

## Acceptance and validation

- Public-host and public-literal HTTP, LAN HTTP, loopback HTTP, and HTTPS construct without DNS.
- A deterministic resolver can route a configured public hostname to the local fixture without
  relaxing the public-only policy or changing the logical credential audience.
- Ordinary authenticated HTTP execution and facade generation/SSE work with the new path.
- Invalid schemes, user information, fragments, foreign targets, and peer mismatches still fail.
- New trusted configuration cannot enable private resource downloads, unsafe proxy policy,
  non-public CONNECT destinations, or unintended WebSocket schemes.
- Existing exact local grants and default public/official HTTPS checks pass unchanged.
- Run targeted nextest first, then the workspace/facade feature lanes, Clippy, doctests, rustdoc
  links, formatting, and diff checks serially with the existing target directory.
- No live provider calls, external publication, credentials, or new validation scripts.

## Risks

- A trusted custom hostname may resolve to any address, including metadata ranges. This is an
  intentional authorization, not an SSRF defense; untrusted endpoint input requires host validation
  or an existing restrictive policy.
- HTTP exposes prompts, responses, and credentials in cleartext. Prefer TLS whenever available.
- Reusing the policy enum can accidentally widen other consumers. Explicit allowlists and negative
  tests must protect resource and proxy boundaries.
