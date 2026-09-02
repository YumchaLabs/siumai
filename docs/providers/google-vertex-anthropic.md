# Claude on Google Vertex AI Provider Support

- Provider identity: `google`
- Technical platform: `vertex-ai`
- Language mode: Anthropic Messages through Vertex `rawPredict` and `streamRawPredict`
- Evidence refreshed: 2026-08-14
- Provider crate: `siumai-provider-google-vertex`

The provider accepts open Claude model identifiers. Its dated model catalog describes verified
models and exact known quirks, but it does not decide account, organization, project, region, or
private-model eligibility.

## Prompt caching

The Google Cloud prompt-caching guide verified on 2026-08-14 documents explicit
`cache_control` breakpoints, a default five-minute TTL, and an opt-in one-hour TTL encoded as
`"ttl": "1h"`. Siumai projects typed one-hour cache intent at message, content, and tool targets for
known, future, and private model IDs. Provider or project eligibility failures remain Vertex API
responses.

The Anthropic Messages codec continues to enforce cache breakpoint count, duplicate annotations,
TTL encoding, and TTL ordering independently of model eligibility.

## Structured outputs and strict tools

The Google Cloud structured-outputs guide verified on 2026-08-14 documents JSON outputs through
`output_config.format`, strict tool use through `tools[].strict`, and organization-policy controls
for the feature. Siumai therefore projects explicit structured-output and strict-tool intent for
known, future, and private model IDs instead of maintaining a positive model allowlist.

Protocol invariants remain local. Constrained output must be strict and use an object JSON schema;
malformed tool annotations and unencodable request shapes still fail before submission.

## Evidence

| Exact slice | Official source | Verified |
|---|---|---|
| Prompt-cache breakpoints and one-hour TTL | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/prompt-caching | 2026-08-14 |
| Structured JSON output, strict tools, and organization-policy controls | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/structured-outputs | 2026-08-14 |

Deterministic offline fixtures cover one-hour cache projection at message, content, and tool
targets for a private future model; structured output and strict tools for one known and one private
future model; and non-strict constrained-output rejection before transport submission.
