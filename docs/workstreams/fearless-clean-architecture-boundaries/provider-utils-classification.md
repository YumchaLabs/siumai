# FCAB-100 Provider-Utils Classification

Date: 2026-05-21

## Decision

FCAB-100 deepens the `siumai-provider-utils` crate from the first FCAB-090 slice into the canonical
home for AI SDK-style helpers that depend only on `siumai-spec` contracts plus generic runtime
libraries. `siumai-core::utils` no longer keeps source-compatible alias modules for those helpers;
it only owns core runtime utilities and explicit compatibility helpers.

The boundary rule is:

- move helpers when their implementation only needs spec-level data shapes or generic crates;
- keep helpers in `siumai-core` when they directly depend on core stream handles, family stream
  parts, or runtime policy contracts;
- do not create shallow provider-specific utility mirrors in provider crates.

## `siumai-provider-utils` Implementations

These modules are implemented in `siumai-provider-utils`; matching `siumai-core::utils::*` alias
modules have been removed:

| Module | Classification | Reason |
| --- | --- | --- |
| `builder_helpers` | Provider-utils | API key, model, and base URL defaults are shared provider-builder glue and depend only on generic error/common-param contracts. |
| `chat_request` | Provider-utils | Request-default merging is provider dispatch normalization, not core runtime policy. |
| `data` | Provider-utils | Base64, data URL, image file-to-data-URI, cosine-similarity, and JSON deep equality helpers are adapter utilities over spec data carriers. |
| `download` | Provider-utils | AI SDK-style safe download and data URL handling sits at provider/protocol adapter boundaries and depends on `DownloadError`, not core runtime types. |
| `error_message` | Provider-utils | Display normalization is a generic provider utility helper. |
| `headers` | Provider-utils | Header normalization/extraction/combination is protocol boundary glue. |
| `id` | Provider-utils | AI SDK-style ID generation is a generic helper and does not require core contracts. |
| `json_instruction` | Provider-utils | Prompt JSON-instruction injection works over spec message/schema data. |
| `json_parse` | Provider-utils | JSON parse/safe-parse helpers use spec JSON/schema contracts. |
| `mime` | Provider-utils | MIME detection and media-type extension helpers are request/response adapter utilities. |
| `option` | Provider-utils | `Arrayable` and nullable filtering helpers are generic AI SDK parity utilities. |
| `provider_options` | Provider-utils | Provider option parsing validates open provider option maps with spec schemas. |
| `provider_reference` | Provider-utils | Provider reference detection/resolution works over spec file-reference carriers. |
| `reasoning` | Provider-utils | Reasoning effort/budget lowering is reusable provider adapter mapping over spec reasoning data. |
| `runtime` | Provider-utils | Runtime version and user-agent suffix helpers are generic provider-utils metadata. |
| `serial_job` | Provider-utils | Serial async job execution is a reusable adapter helper and does not depend on family traits. |
| `standards` | Provider-utils | `ToolNameMapping` maps spec-level provider-defined tool ids to provider-native names for protocol/provider adapters and does not require core runtime contracts. |
| `settings` | Provider-utils | Environment setting/API-key loaders mirror AI SDK provider-utils behavior. |
| `url` | Provider-utils | URL joining/support checks are shared route/request adapter helpers. |
| `utf8_decoder` | Provider-utils | Streaming byte-to-text chunk decoding is a generic protocol helper. |
| `validate_types` | Provider-utils | Runtime type validation uses spec schema contracts and belongs with parse helpers. |

Provider, protocol, and registry crates should import these helpers through their internal
`crate::provider_utils` alias or directly from `siumai-provider-utils` when no internal alias exists.
They should not import moved helpers from `siumai_core::utils::*` or `crate::utils::*`.

## Remaining `siumai-core::utils` Owners

| Module | Classification | Reason |
| --- | --- | --- |
| `cancel` | Stable core runtime utility | It owns `CancelHandle` stream wiring for `ChatStream` and `ChatStreamHandle`. Moving it to `siumai-provider-utils` would either introduce a dependency cycle or force a hollow generic abstraction over core stream handles. Provider crates may continue to use this through their private `crate::utils::cancel` core alias. |
| `streaming_tool_call` | Explicit compatibility helper | It depends on `LanguageModelV4StreamPart`, shared provider metadata, and core stream-part assembly. The facade exposes it only through `siumai::compat::*` / `prelude::compat::*`, not the root or unified prelude. |

## Adjacent Core Runtime Modules

FCAB-100 intentionally does not move these broader modules:

| Module family | Classification | Reason |
| --- | --- | --- |
| `execution` | Core runtime / experimental integration | Owns provider-agnostic execution traits, middleware contracts, and executors over core family types. |
| `streaming` | Stable core stream contract | Owns public stream part/event types and generic SSE parsing; protocol-specific stream state belongs in `siumai-protocol-*`. |
| `retry` / `retry_api` | Core runtime policy | Retry classification and backoff integration are runtime policy over `LlmError`, not provider-utils adapter helpers. |
| `encoding` | Experimental core integration | Token/text encoding helpers are exposed as advanced core runtime utilities and are not provider-specific. |
| `tooling` | Stable/advanced core tooling runtime | Tool execution contracts carry core request options and cancellation handles. |

## Guards

FCAB-100 adds or refreshes core boundary guards for this classification:

- `provider_utils_crate_owns_high_churn_provider_helpers`
- `core_utils_remaining_owned_modules_are_classified`
- `provider_protocol_crates_do_not_import_moved_provider_utils_from_core`
- `core_production_source_does_not_import_registry_facade_provider_or_protocol_crates`

Together these guards prove that the new provider-utils crate owns the spec-only implementations,
`siumai-core` only keeps aliases for migrated helpers, the remaining core-owned utility modules are
classified, and provider/protocol/registry code does not regress to old core utility import paths.
