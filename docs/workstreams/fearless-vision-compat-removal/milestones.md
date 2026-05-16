# Fearless Vision Compatibility Removal - Milestones

Last updated: 2026-05-16

## VCR-M1 - Workstream Opened

Status: complete

Exit criteria:

- Workstream docs exist.
- The docs index links the workstream.
- The deprecated dedicated vision surface is scoped as a removal lane.

## VCR-M2 - Guard Added

Status: complete

Exit criteria:

- Source guard coverage prevents `VisionCapability`, `VisionCapabilityProxy`, or
  `Siumai::vision_capability()` from returning to production public surfaces.
- Historical docs and migration notes remain allowed.

## VCR-M3 - Compatibility Code Removed

Status: complete

Exit criteria:

- Core no longer exports a dedicated `VisionCapability` trait.
- `LlmClient` no longer has an `as_vision_capability()` downcast.
- Registry/facade compatibility wrappers no longer expose a vision proxy.
- Deprecated vision-only request/response aliases are removed or explicitly deferred.

## VCR-M4 - Migration Docs Updated

Status: complete

Exit criteria:

- Migration docs point image understanding users to multimodal chat.
- Migration docs point image creation users to image-family APIs.
- Public import tests no longer pin removed vision names.

## VCR-M5 - Workstream Closed

Status: complete

Exit criteria:

- Focused checks pass.
- `git diff --check` passes.
- Handoff and evidence docs record the final state.

Notes:

- `VisionCapability`, `VisionCapabilityProxy`, `Siumai::vision_capability()`, the `LlmClient`
  vision downcast, and the vision-only request/response aliases were removed.
- The beta.7 migration guide points users to multimodal chat for image understanding and
  image-family APIs for image creation.
