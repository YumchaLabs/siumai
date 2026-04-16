# Fireworks Unified Provider Surface - Milestones

Last updated: 2026-04-07

## FWP-M0 - Scope locked

Acceptance criteria:

- the AI SDK Fireworks provider surface is identified
- the historical Siumai drift is written down explicitly
- the target architecture chooses unified registry aggregation over a new text provider crate

Status: completed

## FWP-M1 - Canonical provider identity is corrected

Acceptance criteria:

- `fireworks` remains the built-in provider id
- native metadata/provider catalog describe Fireworks as an image-capable unified provider
- public facade exposes the canonical provider entry point

Status: completed

## FWP-M2 - Unified runtime surface exists

Acceptance criteria:

- one built Fireworks client exposes chat/completion/embedding/transcription plus provider-owned
  image support
- sync, async, and legacy Fireworks image backends route through the correct provider-owned URLs
- image default-model fallback no longer inherits the text default

Status: completed

## FWP-M3 - Request semantics are aligned

Acceptance criteria:

- provider-owned image requests preserve audited warning/override semantics
- Fireworks compat chat requests normalize the audited provider-specific option shape
- Kontext edits are restricted to the supported Fireworks image models

Status: completed

## FWP-M4 - Docs and handoff are complete

Acceptance criteria:

- dedicated Fireworks workstream docs exist
- structural-alignment docs mention Fireworks as a closed provider-surface gap
- unreleased changelog sections capture the user-visible/runtime implications

Status: completed

## FWP-M5 - Ongoing parity maintenance

Acceptance criteria:

- Fireworks typed option/model coverage stays aligned with the audited AI SDK package
- curated model metadata stays in sync with the chosen reference set

Status: in progress
