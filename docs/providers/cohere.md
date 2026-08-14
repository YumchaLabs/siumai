# Cohere Provider Support

- Provider identity: `cohere`
- Technical platform: `public-api`
- Portable families: Embedding and Rerank
- Primary API mode: native v2
- Embedding evidence verified: 2026-08-14
- Provider crate: `siumai-provider-cohere`

Known model constants and the provider profile catalog are dated ergonomic hints, not execution
allowlists. Private, proxied, and future model identifiers remain valid when their requests can be
encoded by the v2 wire contract.

## Embedding output dimensions

Cohere's v2 Embed reference documents `output_dimension` as an optional integer with the exact
values `256`, `512`, `1024`, and `1536`. It also describes the current product availability as
Embed v4 and newer. Siumai treats the four-value domain as a stable wire invariant while leaving
model availability to Cohere and the host application.

The provider therefore:

- encodes an explicit valid dimension for known, private, or future model identifiers;
- rejects values outside the documented four-value domain before transport;
- rejects a conflict between portable `EmbeddingRequest::dimensions` and typed Cohere options;
- verifies that every returned float vector matches the requested dimension.

This boundary deliberately does not infer eligibility from a model-name prefix or copy Cohere's
current catalog into runtime validation. Cohere remains authoritative and may reject a valid wire
value for a model or account that does not offer the product capability.

## Evidence

| Exact slice | Official source | Verified |
|---|---|---|
| v2 Embed request, `output_dimension` values, and current product guidance | [Cohere Embed v2 reference](https://docs.cohere.com/v2/reference/embed) | 2026-08-14 |

Deterministic offline fixtures cover future-model dimension encoding, invalid values, canonical
versus typed conflicts, and response vector-length mismatches. They do not perform live,
credentialed, or billable calls.
