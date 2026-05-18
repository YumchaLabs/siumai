# Provider Surface Second Pass

Status: Closed
Last updated: 2026-05-18

## Why This Lane Exists

The AI SDK provider-interface convergence program is closed, but the repository still has a few
residual shape and hygiene questions:

- some provider-surface lanes are intentionally deferred rather than mirrored as AI SDK parity
  targets
- some historical lanes are still `unknown` in the index, which makes reuse and follow-on
  selection noisier than it should be
- `stream-metadata-parity-hardening` still has concrete open follow-up items and may be the next
  real code slice instead of a dead historical note

This lane exists to make that second pass explicit. It is not a reopen of the parent AIPC program.
It is a narrow program lane for classifying the remaining provider-surface drift, separating
intentional deferrals from real follow-ons, and only then opening bounded implementation slices.

## Relevant Authority

- `docs/workstreams/ai-sdk-provider-interface-convergence/DESIGN.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/HANDOFF.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/PARITY_INVENTORY.md`
- `docs/workstreams/INDEX.md`
- `docs/workstreams/stream-metadata-parity-hardening/DESIGN.md`
- `docs/workstreams/stream-metadata-parity-hardening/TODO.md`
- `docs/workstreams/google-vertex-typed-option-surface-alignment/design.md`
- `docs/workstreams/minimaxi-unified-provider-surface/design.md`
- `docs/workstreams/ollama-unified-provider-surface/design.md`
- `docs/workstreams/workstream-status-hygiene/DESIGN.md`
- ADR-0001, ADR-0002, ADR-0003, ADR-0005, ADR-0006, ADR-0007, ADR-0008

## Problem

After the provider-by-provider alignment work, the remaining risk is no longer broad provider
coupling. The remaining risk is residue:

- the index still has historical unknowns
- deferred provider lanes can be mistaken for unfinished parity work
- a real stream/metadata follow-on can be buried under historical notes
- without a second-pass lane, future work can drift back into the wrong parent program or reopen a
  closed lane for mechanical cleanup

## Target State

- `docs/workstreams/INDEX.md` matches the actual directory and status layout.
- Residual workstreams are classified as closed, deferred, historical, or concrete follow-on.
- `stream-metadata-parity-hardening` is either resumed as the next real follow-on or explicitly
  split/closed with machine-readable status.
- `minimaxi` and `ollama` remain intentional deferred surfaces unless a real parity target appears.
- Any new code work is split into a bounded child lane instead of being absorbed into this
  coordination lane.

## In Scope

- `docs/workstreams/INDEX.md`
- residual workstream metadata and handoff docs
- classification of `stream-metadata-parity-hardening`
- confirmation of deferred provider-surface decisions for MiniMaxi and Ollama
- child workstream split decisions for any concrete drift discovered here

## Out Of Scope

- Reopening the closed AIPC parent program for mechanical cleanup
- Broad provider code rewrites without a concrete drift finding
- Forcing AI SDK parity onto providers that are intentionally deferred
- Historical lane rewrites that do not change reuse decisions

## Architecture Direction

Treat the workstream index as the shallow navigation interface for historical lanes. Treat deferred
provider surfaces as intentional, not incomplete. Treat concrete drift as a child lane, not a
reason to widen this program.

This lane should help future agents answer three questions quickly:

1. Is this lane real work or historical context?
2. Is this provider-surface difference intentional or a gap?
3. If it is a real gap, what is the smallest bounded lane that can own it?

## Closeout Condition

This lane can close when:

- the index reflects the actual residual lane set,
- the remaining provider-surface candidates are classified,
- any concrete drift has been split into a bounded child lane,
- and the documentation makes the intentional deferrals obvious.

Closeout result: satisfied. No concrete provider drift was found in this second pass. The residual
lanes were classified with machine-readable status, and future work should start from a narrow
behavioral gap rather than this coordination lane.
