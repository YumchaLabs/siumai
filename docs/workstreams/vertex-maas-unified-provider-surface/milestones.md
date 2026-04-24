# Vertex MaaS Unified Provider Surface - Milestones

Last updated: 2026-04-10

## VMP-M0 - Scope locked

Acceptance criteria:

- AI SDK Vertex MaaS reference files are identified
- Siumai's missing public/runtime surface is written down explicitly
- the workstream chooses a canonical provider id

Status: completed

## VMP-M1 - First-class provider identity exists

Acceptance criteria:

- `vertex-maas` is a built-in provider id
- `google-vertex-maas` and `vertex.maas` aliases resolve to the same surface
- native metadata and provider catalog output describe the provider correctly

Current state:

- `vertex-maas` is now registered as a built-in provider id
- registry alias resolution now includes `google-vertex-maas` and `vertex.maas`
- native metadata and provider catalog now expose the curated MaaS text/completion/embedding story

Status: completed

## VMP-M2 - Runtime/base-url/auth semantics align

Acceptance criteria:

- runtime reuses shared OpenAI-compatible execution
- `project + location` can derive the OpenAPI base URL
- Google Bearer auth works without forcing a fake non-empty API key

Current state:

- `VertexMaasProviderFactory` now reuses the shared compat runtime
- base URL derives from `project + location`, env fallback, or explicit override
- compat auth checks now allow preexisting `Authorization` headers, and Vertex MaaS supports
  token provider / explicit header / static bearer / ADC fallback

Status: completed

## VMP-M3 - Public facade and regression coverage exist

Acceptance criteria:

- `Provider::vertex_maas()` and `Siumai::builder().vertex_maas()` are public
- contract tests pin base URL and auth behavior
- public-path tests pin builder/provider/registry equivalence

Current state:

- the unified builder/provider surfaces are public
- registry contract tests now pin project/location precedence, `global` fallback, auth injection,
  and completion/embedding-family support
- public-path parity and import tests now cover the unified Vertex MaaS surface
- public `provider_ext::vertex_maas` model constants plus provider-catalog curated model reuse now
  keep the facade and registry on the same audited MaaS subset
- `provider_ext::vertex_maas::{GoogleVertexMaasProviderSettings, VERSION}` now mirrors the
  package-level settings/version surface for the audited `project/location/baseURL/headers/fetch`
  subset

Status: completed

## VMP-M4 - Stable provider typing is cleaned up around the Google Vertex family

Acceptance criteria:

- `vertex-maas` is not the only Google Vertex surface with stable typing
- `vertex` and `anthropic-vertex` also stop degrading to `Custom(...)`
- retry/validator/provider-catalog layers agree on those identities

Current state:

- `ProviderType::{Vertex, AnthropicVertex, VertexMaas}` now all exist
- provider catalog now resolves native metadata for `vertex`, `anthropic-vertex`, and
  `vertex-maas`
- retry/default validator helpers now also understand the Google Vertex wrapper family

Status: completed
