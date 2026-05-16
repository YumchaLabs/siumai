# Fearless Boundary Hardening - TODO

Last updated: 2026-05-14

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Planning And Boundary Policy

- [x] Create the boundary-hardening workstream design document.
- [x] Create the boundary-hardening milestone document.
- [x] Create the boundary-hardening TODO document.
- [x] Record the fearless-removal policy: unnecessary compatibility and redundant code may be
  removed when a canonical path is tested and documented.
- [x] Link this workstream from the docs index.
- [x] Add or update source guards that prevent new stable-family code from depending on
  compatibility-only `LlmClient` downcasts.
  - `siumai-registry::registry::entry::boundary_tests` now rejects `compat_*_client_with_ctx`
    calls in stable handle primary execution paths while leaving explicitly extension-only bridges
    available.
- [x] Add or update source guards that prevent protocol/provider-specific modules from returning
  to `siumai-core`.
  - `siumai-protocol-openai::openai_compat_boundary_test` now prevents any direct
    `siumai_core::standards::openai` imports and verifies `siumai-core/src/standards/openai` does
    not own Rust protocol files.

## Track B - Core Boundary Hardening

- [x] Audit `siumai-core/src/standards/openai/*` and decide which remaining wire-format helpers
  belong in `siumai-protocol-openai`.
- [x] Move provider/protocol-specific OpenAI wire structs or conversion helpers out of
  `siumai-core` when they are not genuinely provider-agnostic.
- [x] Remove the `siumai-core::standards::openai` module after the protocol crate owns the remaining
  OpenAI message, wire type, finish reason, and usage helpers.
- [x] Audit `siumai-core/src/utils/*` for provider-specific helpers and move them to
  `auth`, `protocol`, or provider-owned modules.
  - The deprecated `siumai_core::utils::vertex` forwarding module was removed; Vertex URL helpers
    later moved from `siumai_core::auth::vertex` to provider-owned
    `siumai-provider-google-vertex::auth::vertex`, with facade compatibility kept through
    `siumai::experimental::auth::vertex`.
- [x] Remove deprecated utility re-exports after their migration targets are documented and tested.
- [x] Keep `siumai-core` focused on family traits, generic runtime, retry, middleware, streaming
  carriers, and provider-agnostic helpers.

## Track C - Provider And Protocol Re-Export Tightening

- [x] Audit broad `pub use siumai_core::{...}` re-exports in provider crates.
- [x] Audit broad `pub use siumai_core::{...}` re-exports in protocol crates.
- [x] Replace broad re-exports with explicit imports where the exports only preserve internal
  migrated module paths.
- [x] Keep public compatibility exports only when they are documented and time-bounded.
  - The legacy `siumai-provider-openai-compatible` crate still re-exports
    `siumai-protocol-openai::*`, but it no longer receives `siumai-core` mirror exports through the
    protocol crate.
- [x] Add source guards for provider/protocol crates once the broad re-export pattern is narrowed.
  - `openai_compat_boundary_test` rejects public `pub use siumai_core::{...}` and
    `pub use siumai_core::builder::*` mirrors in protocol/provider crate roots.

## Track D - Registry Compatibility Reduction

- [x] Audit remaining compatibility adapters and identify which are still required.
  - Stable registry handles now use family factory methods for primary execution.
  - The default `ProviderFactory` family methods now reject providers without native family
    construction instead of silently wrapping `compat_*_client*` outputs.
  - The old default bridge module was removed; extension-only generic client surfaces remain
    behind explicit `compat_*_client*` methods.
- [x] Remove compatibility adapters for stable families once all declared provider families have
  native family factory paths.
  - Registry entry tests now cover native family paths and default-rejection behavior for stable
    families.
- [x] Keep extension-only compatibility downcasts isolated until file, skill, music, or other
  extension handles have explicit construction paths.
  - Current extension-only bridges are isolated to language file/skill/music, image extras, speech
    streaming/voice listing, and transcription streaming/translation/language listing.
- [x] Tighten `ProviderFactory` docs and tests so `compat_*_client*` methods cannot be treated as
  primary implementation points.
  - `ProviderFactory` now exposes family-first methods and explicit `compat_*_client*` bridges
    only; deprecated generic `*_model` and `*_model_with_ctx` wrapper methods were removed.
  - Registry boundary tests prevent stable handle primary execution from drifting back to those
    compatibility methods.
- [x] Remove deprecated generic factory wrappers after migration notes and public compatibility
  windows allow it.
  - Tests and parity checks now use `compat_*_client*` for provider-owned generic client
    construction when an extension-only surface still needs it.

## Track E - Facade Surface Hardening

- [x] Audit `siumai/src/lib.rs` for root-level helper exports that should live in family modules.
- [x] Keep `siumai::prelude::unified::*` focused on stable family APIs and documented helper names.
  - Compatibility construction aliases now live under explicit `siumai::compat` and
    `siumai::prelude::compat` paths instead of the stable unified prelude.
- [x] Remove deprecated aliases from the stable prelude when they are not part of a documented
  compatibility promise.
  - Deprecated `experimental_generate_*` helper aliases are no longer exported from
    `prelude::unified`; explicit family/root paths remain available where documented.
- [x] Keep `siumai::experimental::*` as an explicit advanced boundary, not a default import path.
- [x] Add or update facade boundary tests for any narrowed public surface.
  - `facade_architecture_boundary_test::stable_unified_prelude_excludes_compatibility_construction_aliases`
    guards the stable prelude against regressing to compatibility-first exports.

## Track F - Documentation And Migration Hygiene

- [x] Update migration docs when a public compatibility alias or builder path is removed.
  - `docs/architecture/public-surface.md` now records the explicit compat prelude and the removed
    `experimental::utils::vertex` facade path.
- [x] Update examples that still use compatibility-first construction where a registry/config-first
  replacement exists.
- [x] Keep README and architecture docs from recommending `Siumai::builder()` or `LlmClient` as
  default paths.
- [x] Record each intentional removal in the relevant migration guide or changelog entry.
  - Intentional removals in this workstream are recorded in `milestones.md`; older compatibility
    audit entries remain historical context.

## Track G - Validation

- [x] Run focused `cargo nextest` checks for each affected crate after every deletion slice.
- [x] Run provider/protocol tests when moving protocol helpers between crates.
- [x] Run registry architecture boundary tests after changing factories or handles.
- [x] Run facade architecture boundary tests after changing facade exports.
- [x] Run `cargo fmt --check` or package-scoped format checks before each commit candidate.
- [x] Run `git diff --check` before closing the workstream.
