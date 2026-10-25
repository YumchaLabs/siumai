# OpenAI-compatible endpoints

## Trusted provider configuration (beta.12)

`OpenAiCompatibleProfile::custom(...)` and the facade's `.unauthenticated()` convenience method
are available starting with `0.11.0-beta.12`. Enable the `openai-compatible` facade feature
explicitly; `all-providers` does not include it. For `0.11.0-beta.11`, use the compatibility
example below.

Use `custom` for a developer-controlled or independently validated provider origin. It accepts
HTTP or HTTPS without requiring the caller to classify its network addresses: public HTTP
domains or IP literals, LAN services, loopback, and HTTPS gateways all use the same constructor.
This is explicit trust in one configured origin, not a global `allow_http` switch.

Use a generic profile rather than treating a self-hosted service or relay as the official
OpenAI API:

```rust,no_run
use siumai::providers::openai_compatible::{
    OpenAiCompatibleApiMode, OpenAiCompatibleProfile,
};
use siumai::{ProviderId, ReplayDomainId, Siumai};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile = OpenAiCompatibleProfile::custom(
        ProviderId::new("self-hosted")?,
        "http://127.0.0.1:11434/v1",
        ReplayDomainId::new("self-hosted")?,
        OpenAiCompatibleApiMode::ChatCompletions,
    )?;
    let ai = Siumai::builder()
        .openai_compatible()
        .profile(profile)
        .unauthenticated()
        .build()?;
    let client = ai.language("your-served-model")?;
    let response = client.generate("Hello").await?;
    println!("{:?}", response.output_text());
    Ok(())
}
```

Replace the model ID with one actually served by the endpoint. Siumai does not download models,
discover a local inventory, or require a known model constant. The endpoint must implement the
selected wire API; a native-only service is not made OpenAI-compatible by its URL.

Replace the URL with the origin you explicitly trust, such as `http://192.168.1.20:8000/v1`
or `https://gateway.example/v1`; no separate local-network switch is needed. Never pass an
unchecked tenant or user URL to this trusted constructor. A URL supplied by a tenant is not
trusted merely because it is syntactically valid.

If the service requires authentication, replace `.unauthenticated()` with `.api_key(...)` or
`.credential(...)`. Do not supply a dummy key to an unauthenticated service. Existing credential
paths and defaults are unchanged.

The same client exposes `client.stream("Hello").await?`. Select
`OpenAiCompatibleApiMode::Responses` only when the server implements that API; Siumai does not
infer the API mode from the model ID or fall back between modes.

Construction and model acquisition are synchronous and network-free. The profile stays generic
and experimental with platform `custom-endpoint`, even for a loopback URL or a URL naming an
official provider. It does not inherit provider fidelity, lifecycle advice, or a claim that a
particular server implements the full OpenAI API.

The caller supplies a non-secret `ReplayDomainId`; Siumai does not invent one from the URL or
provider name. The resulting replay audience is custom, never official. Choose replay domains
that keep unrelated account, deployment, or history boundaries separate.

## Trust and transport boundaries

- Prefer HTTPS whenever possible. HTTP exposes prompts, responses, and any credentials in
  cleartext, including on public networks. Siumai does not upgrade HTTP to TLS for you.
- Only pass developer-controlled or independently validated configuration to `custom`. It
  deliberately permits arbitrary network addresses, including link-local metadata services;
  it is not an SSRF filter for unchecked tenant or user URLs.
- `custom` delegates to transport's `EndpointConfig::custom` and `EndpointPolicy::TrustedCustom`.
  URL validation and exact-origin credential binding remain in force. Unsupported schemes and
  URL user information are rejected; relative API requests cannot change the configured origin.
- Resource downloads remain a separate, unauthenticated Direct-only boundary. Provider-returned
  or user-supplied resource URLs never inherit this trusted provider policy.
- Redirects and ambient process proxy discovery remain disabled. Trusted-custom configuration
  does not authorize CONNECT routing or proxy destinations. Existing explicit proxy grants and
  public-HTTPS provider destination restrictions remain unchanged; CONNECT is not an arbitrary
  HTTP tunneling escape hatch.

See the [transport contract](../architecture/transport-contract.md) for the transport-owned rules.

## Published beta.11 compatibility

For `siumai = "=0.11.0-beta.11"`, retain the existing strict constructors and explicit credential:

```rust
use siumai::providers::openai_compatible::{
    OpenAiCompatibleApiMode, OpenAiCompatibleCredential, OpenAiCompatibleProfile,
};
use siumai::{ProviderId, ReplayDomainId, Siumai};

let profile = OpenAiCompatibleProfile::local_explicit(
    ProviderId::new("self-hosted")?,
    "http://127.0.0.1:11434/v1",
    ReplayDomainId::new("self-hosted")?,
    OpenAiCompatibleApiMode::ChatCompletions,
)?;
let ai = Siumai::builder()
    .openai_compatible()
    .profile(profile)
    .credential(OpenAiCompatibleCredential::unauthenticated())
    .build()?;
let _client = ai.language("your-served-model")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`local_explicit` authorizes HTTP(S) on IPv4/IPv6 loopback only. For a LAN destination use
`custom_endpoint` with `EndpointConfig::private_network_explicit` and an explicit
`ReplayDomain::custom(...)`. Other local grants remain separate and exact. `public_custom`
continues to require HTTPS and public addresses; it does not accept public HTTP. These strict
paths remain available unchanged in beta.12 as restrictive alternatives to `custom`.
For strict paths, `SchemeNotAllowed` and `AddressNotAllowed` still report violations of the
selected policy; neither is a reason to trust unchecked input through the new constructor.

## Offline verification

The [offline example](../../siumai/examples/self_hosted_http.rs) uses `custom` to
construct the complete facade without connecting to a server:

```text
cargo run -p siumai --example self_hosted_http --no-default-features --features openai-compatible -j 1
```

The facade contract suite separately exercises generation and SSE against an in-process HTTP
fixture, including the absence of an Authorization header for an unauthenticated credential.
