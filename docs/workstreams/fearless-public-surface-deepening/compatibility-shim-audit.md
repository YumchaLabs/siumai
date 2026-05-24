# Fearless Public Surface Deepening - Compatibility Shim Audit

Last updated: 2026-05-25

## Scope

This audit classifies the remaining public compatibility shims after the facade and streaming
deepening slices. It is intentionally narrower than the older architecture audits: this document
answers whether any remaining shim can be deleted in the current public-surface deepening lane.

ADR-0007 and ADR-0008 are authoritative for this decision:

- ADR-0007 keeps `LlmClient` available for backward compatibility, bridge adapters, registry
  internals that have not migrated to family-native execution, and compatibility-focused
  capability discovery.
- ADR-0008 keeps legacy `ContentPart` available in the current beta line while classifying it as a
  compatibility carrier and deferring any move under `compat` to a later breaking slice.

## Classification Rules

- `keep now`: the shim is still required for source compatibility, migration examples, or an
  extension-only gap that has not reached a native family surface.
- `future breaking lane`: the shim is intentionally retained now, but has a clear deletion or
  narrowing condition that should be handled in a breaking-change workstream.
- `delete now`: safe to remove in this lane without contradicting ADR-0007, ADR-0008, migration
  docs, or public compile tests.

## Summary

| Surface | Current paths | Classification | Reason | Existing guard / evidence | Follow-up |
| --- | --- | --- | --- | --- | --- |
| Facade method-style builder entry | `siumai::compat::{Siumai, SiumaiBuilder, Provider}`, `siumai::prelude::compat::{Siumai, SiumaiBuilder, Provider}` | keep now; future breaking lane | Source-compatible path for historical method-style construction. Root `siumai::Provider`, `siumai::provider::*`, and `siumai::builder::*` shims are already removed; the explicit compat path remains time-bounded no earlier than `0.12.0`. | `root_provider_builder_entry_is_compatibility_classified`, `public_surface_compat_imports_compile`, `public_surface_compat_prelude_imports_compile`, migration beta.7 docs. | Reassess after examples and downstream migration no longer need method-style construction. |
| Legacy builder internals | `siumai::compat::builder::*`, especially `siumai::compat::builder::BuilderBase` | keep now; future breaking lane | Provider builders still need a provider-agnostic base snapshot during migration. The removed root builder shim already prevents accidental stable-prelude coupling. | `root_provider_builder_entry_is_compatibility_classified`, `public_surface_compat_imports_compile`. | Narrow the module or move direct builder-base guidance to provider crates in a breaking lane. |
| Generic client compatibility types | `siumai_core::compat::client::{LlmClient, ClientWrapper}`, `siumai::compat::client::{LlmClient, ClientWrapper}`, `siumai::experimental::client::{LlmClient, ClientWrapper}`, `siumai_registry::compat::client::{LlmClient, ClientWrapper}` | keep now; future breaking lane | ADR-0007 explicitly allows `LlmClient` for backward compatibility, bridge adapters, registry internals, and compatibility capability discovery. The physical owner is already `siumai_core::compat::client`. | `llm_client_is_physically_scoped_under_compat_module`, `facade_generic_client_paths_are_explicit_compatibility_exports`, `registry_generic_client_imports_are_compat_scoped`, `public_surface_extensions_imports_compile`. | Remove or hard-deprecate only after major providers and registry handles migrate off generic-client dispatch. |
| Lower-level generic client aliases | `siumai_core::client`, `siumai_core::core::client` | keep now; future breaking lane | These are migration aliases, not implementation owners. Deleting them now would create a low-level public break before ADR-0007 removal conditions are met. | `llm_client_is_physically_scoped_under_compat_module`; beta.7 migration docs name `siumai_core::client` as a lower-level migration alias. | Move to an explicit deprecation/removal workstream after family-native provider coverage is complete. |
| Registry generic-client factory facet | `ProviderCompatibilityFactory`, `ProviderFactory::compat_*_client(...)`, `ProviderFactory::compat_*_client_with_ctx(...)`, `siumai_registry::registry::factory::build_*_client(...)` | keep now; future breaking lane | Stable registry handles have been split to family facets, but legacy method-style construction and extension-only surfaces still need an explicit generic-client seam. | `provider_factory_trait_is_family_first_before_compat_methods`, `registry_root_keeps_small_custom_factory_surface`, registry source guards around `compat_*_client`. | Delete only when method-style `Siumai` and extension-only generic-client adapters are replaced by native family or extension factories. |
| Legacy chat content carrier | `siumai_core::compat::content::*`, `siumai::compat::content::*`, `siumai::content::compat::*`, `siumai::prelude::compat::content::*` | keep now; future breaking lane | ADR-0008 keeps `ContentPart` available during the beta line while new code moves to directional prompt/output content. | `legacy_content_part_has_explicit_compat_namespace`, `stable_unified_prelude_does_not_export_legacy_content_part`, `public_surface_legacy_content_part_uses_explicit_compat_namespace`, `public_surface_directional_content_namespaces_compile`. | Later breaking slice may move or deprecate old paths after directional adapters and fixture parity are complete. |
| Broad migration type namespace | `siumai::compat::types::*`, `siumai::prelude::compat::types::*` | keep now; future breaking lane | Historical `siumai::types::*` has been removed; the broad import survives only under explicit compatibility namespaces for migration. | `broad_facade_types_path_is_explicit_compat_only`, `public_surface_compat_imports_compile`, beta.7 migration docs. | Narrow after downstream users have migrated to `prelude::unified`, family modules, provider extensions, or focused compat imports. |
| Deprecated AI SDK spelling aliases | `siumai::compat::{CallSettings, Experimental_*, experimental_filter_active_tools, step_count_is}`, `siumai::prelude::compat::{...}` | keep now; future breaking lane | Deprecated upstream spelling parity remains useful for source-compatible migration, but stable `prelude::unified` should expose the non-experimental names. | `stable_unified_prelude_excludes_compatibility_construction_aliases`, `public_surface_compat_imports_compile`, `public_surface_compat_prelude_imports_compile`. | Remove after a documented deprecation window for explicit compat paths. |
| Streaming tool-call compatibility helpers | `siumai_core::utils::streaming_tool_call::*`, re-exported by `siumai::compat::*` and `siumai::prelude::compat::*` | keep now; future breaking lane | The root/prelude aliases are already removed. The remaining implementation is classified as an explicit compatibility helper over core stream-part types. | `core_utils_remaining_owned_modules_are_classified`, `streaming_tool_call_helpers_are_explicit_compat_only`, public compat import tests. | Move implementation to a protocol or dedicated compatibility crate only with a public migration note. |
| Provider extension legacy parameters | `siumai::provider_ext::<provider>::legacy_params::*` | keep now; future breaking lane | Provider-owned legacy params are intentionally scoped under provider extensions so stable family APIs do not learn deprecated parameter bags. | `provider_ext_legacy_params_are_explicitly_scoped`, provider public-surface compile tests, beta.7 migration docs. | Remove per provider when provider-specific option migration is complete. |
| OpenAI-compatible provider/protocol compat modules | `siumai::protocol::openai::compat::*`, provider-owned OpenAI-compatible config/client aliases, adapter/spec modules | keep now | These are provider/protocol compatibility surfaces, not facade root shims. They remain the owner for OpenAI-compatible wire adapters and vendor families. | OpenAI-compatible facade architecture guards, protocol/provider route guards, provider-ext documentation guards. | Continue to keep provider/protocol ownership explicit; do not move these into `siumai-core` or facade root. |

## Delete Now

delete now: none.

No remaining audited shim satisfies the deletion criteria in this lane. The high-risk broad root
paths were already removed in earlier lanes (`siumai::Provider`, `siumai::provider::*`,
`siumai::builder::*`, `siumai::types::*`, `siumai_registry::LlmClient`, and direct root streaming
tool-call aliases). The remaining shims are explicit compatibility namespaces or migration aliases
protected by ADR-0007 / ADR-0008.

## Future Breaking Lane Candidates

1. Remove or hard-deprecate method-style facade construction after public examples and downstream
   usage move to registry handles and config-first provider clients.
2. Delete `siumai_core::client` and `siumai_core::core::client` once generic-client compatibility
   imports have moved to `siumai_core::compat::client`.
3. Narrow `siumai::compat::types::*` and `siumai::prelude::compat::types::*` into smaller named
   compatibility groups.
4. Move legacy `ContentPart` paths fully under compatibility namespaces once ADR-0008's future
   breaking-slice conditions are met.
5. Retire `ProviderCompatibilityFactory` and `compat_*_client*` methods after registry and
   extension-only surfaces have native family or extension factories.
6. Move `StreamingToolCall*` helper implementation out of `siumai-core::utils` if a protocol-owned
   or dedicated compatibility owner becomes available.
