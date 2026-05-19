# ContentPart Compatibility Namespace Break — Handoff

Status: Closed
Last updated: 2026-05-19

## Current State

This workstream was opened as the follow-up requested after
`docs/workstreams/fearless-module-deepening/` closed.

Relevant prior work:

- ADR-0008 classifies legacy `ContentPart` as compatibility-only and defines the preconditions for
  a later breaking namespace move.
- `fearless-content-part-boundary-split` added the initial direct-construction audit and proof
  migrations.
- `fearless-module-deepening` extracted request-side bridge legacy `ContentPart` construction into
  `siumai-bridge/src/request/legacy_content.rs`.

CPN-010 is complete: the new lane is documented, task order is defined, and CPN-020 was the first
executable task.

CPN-020 is complete: the explicit legacy content compatibility namespace now exists at
`siumai::compat::content::*` and `siumai::prelude::compat::content::*`, with spec/core/facade
re-export layers and migration/public-surface docs updated. Old-path aliases remain available for
this slice; CPN-030 owns the breaking stable-prelude cleanup decision.

CPN-030 is complete: `siumai::prelude::unified::*` no longer exports the legacy dual-use
`ContentPart` carrier. Provider-extension examples that still need legacy content metadata now opt
into `siumai::compat::content::ContentPart`; architecture docs and guards state the rule.

CPN-040 is complete: `siumai-spec/src/types/ai_sdk/response_compat_projection.rs` now owns the
response-side legacy `ContentPart` -> generated-output projection. Public helper names are
unchanged through `ai_sdk::mod.rs` re-exports, and source guards prove `generate_text.rs` no longer
owns direct `ContentPart::*` response mapping.

CPN-050 is intentionally deferred/split: bridge/protocol response paths still serialize
provider-native `ChatResponse`/`ContentPart` shapes directly. Forcing them through
`GenerateTextContentPart` would create lossy round-trips and broaden this workstream into parser
redesign. The audit now points to the named adapter as the shipped seam.

CPN-060 is complete: migration/public-surface docs and source guards now state the final
`ContentPart` import rule, replacement directional content families, and named response adapter
behavior.

CPN-070 is complete: the workstream is closed with fresh final gate evidence.

## Final State

- Workstream status: closed.
- `WORKSTREAM.json`: `status=closed`, `active_task=null`, `next_task=null`.
- Final gates:
  - `cargo fmt --check -p siumai-spec -p siumai-core -p siumai`
  - `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast`
  - `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast`
  - `cargo nextest run -p siumai-bridge --features openai,anthropic,google response --no-fail-fast`
  - `cargo nextest run -p siumai --features openai,anthropic,google,xai,groq --test public_surface_imports_test --no-fail-fast`

## Decisions

- Preserve the serialized shape of `ContentPart`, `ChatMessage`, and `ChatResponse`.
- Old-path aliases stayed available in CPN-020 so the explicit compatibility namespace could land
  first with migration docs and tests.
- `prelude::unified` no longer exports legacy `ContentPart`; migration code must use
  `siumai::compat::content::ContentPart`.
- The response-side compatibility adapter lives in `siumai-spec::types::ai_sdk` and remains
  public through existing helper re-exports.
- Protocol/bridge response parser migration is split until a generated-output lane is a natural
  target, not a lossy detour.
- Prefer moving recommended facade imports and docs first, then deepen response adapters.

## Blockers

- None known.

## Next Recommended Action

1. Commit this workstream if the user approves.
2. Suggested commit message: `refactor: isolate legacy content compatibility surface`.
3. Future optional follow-on: a separate protocol/parser generated-output migration lane, only if a
   provider-native response model can adopt generated-output parts without lossy round-trips.
