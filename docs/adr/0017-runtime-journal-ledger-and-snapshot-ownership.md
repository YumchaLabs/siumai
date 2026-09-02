# ADR-0017: Runtime Journal, Deferred Ledger, and Snapshot Ownership

## Status

Accepted

## Date

2026-08-15

## Context

Durable runtime state previously had several writers. The step engine appended execution events,
the durable adapter reconstructed recovery events, snapshot helpers assembled overlapping state,
and provider-deferred observations entered the report before the model stream reached an
authoritative terminal. Public constructors also allowed external code to create snapshot graphs
that the runtime itself could never produce.

That distribution made partial mutation possible. A later invalid tool call could leave earlier
prepared events and budget mutations behind, a failed stream could retain resumable provider work,
and recovery could bypass the same successor and snapshot-size checks used by ordinary execution.
`RunStore` adapters were also at risk of becoming a second runtime state machine instead of a narrow
lease and compare-and-swap port.

Durable snapshots are sensitive, authority-bearing replay state. Sanitized diagnostics and public
inspection are useful, but public mutation is not part of the storage contract.

## Decision

Runtime uses four private lifecycle owners:

1. A completed-step planner freezes and validates the full model result before semantic state is
   committed. Consumed model attempts and provider-reported usage settle independently and exactly
   once.
2. `ToolJournal` is the only writer of tool execution transitions. It owns sequence, time, attempt,
   recovery, retry eligibility, and the read-only execution-log projection.
3. `ProviderDeferredLedger` owns exact `ProviderScope + correlation_id` identity, ordered in-place
   updates, monotonic resolution, and pending provider-state projection. Stream observations are
   staged per call and commit only after an authoritative completed terminal.
4. `CheckpointWriter` is the only snapshot candidate writer. It assembles and validates a
   candidate, validates `previous -> candidate`, measures bounded compact JSON, enforces the runtime
   snapshot budget, renews the lease, and then invokes store CAS.

Snapshot v8 and durable execution ABI v7 activate these ownership changes together. Version 8
removes `dispatch_id`, serializes the validated journal and deferred ledger, and rejects v7 at the
version envelope without a migration shim. Snapshot, resume, terminal, pending, journal-event, and
fingerprint state remains publicly serializable and inspectable through kinds and accessors, while
runtime-only constructors and mutators are crate-private.

`RunStore` remains a storage port. It owns lease fencing, run identity, revisions, terminal-write
rejection, and atomic replacement. It does not validate runtime successor semantics. External
stores must reject oversized serialized input before typed deserialization and supply
confidentiality, integrity and authenticity, tenant/run isolation, access control, and rollback or
revision protection.

## Options considered

### Option A: Keep public state constructors and validate at CAS time

Rejected. It preserves multiple state writers and makes invalid graphs a supported public input.

### Option B: Move successor and schema rules into every `RunStore`

Rejected. Storage adapters do not have enough runtime context and would duplicate a state machine
across backends.

### Option C: Add compatibility shims for snapshot v7

Rejected. Reconstructing journal ownership, deferred resolution, or removed dispatch identity would
require guesses about effects and replay authority.

### Option D: Use private lifecycle owners and one deliberate schema break

Chosen. It keeps each invariant in the lowest layer that can enforce it completely and makes the
store boundary narrow enough for external adapters.

## Consequences

### Positive

- Tool planning is semantically atomic while model-call accounting remains exact.
- Failed, cancelled, and unexpectedly closed streams cannot create resumable provider work.
- Crash recovery and stable retry use the same journal and checkpoint gates as ordinary execution.
- Every checkpoint candidate is size-bounded before CAS.
- Public snapshot APIs support inspection and storage without exposing state mutation.

### Costs

- Snapshot v7 and durable ABI v6 runs cannot resume under the new runtime.
- Downstream code that pattern-matched or constructed snapshot enums must migrate to kinds and
  accessors.
- External stores must treat serialized snapshots as sensitive authority-bearing data and enforce
  their own pre-deserialization resource and security boundaries.

## Migration and verification

Drain or discard beta-era durable runs before deploying the new ABI. Recreate trusted runs through
`DurableToolLoop`; do not rewrite old JSON into v8.

Deterministic tests cover completed-step rollback, journal transition legality, exact deferred
identity and resolution, mixed local/provider progression, v8 strict decoding, v7 envelope
rejection, zero-CAS invalid and oversized candidates, exact byte limits, external pre-decode bounds,
and sanitized diagnostics. External compile contracts use only public snapshot accessors and
`RunStore` adapter constructors.

## References

- `docs/adr/0014-canonical-language-history-and-replay.md`
- `docs/adr/0015-validation-ownership-and-forward-compatibility.md`
- `docs/architecture/overview.md`
- `docs/migration/siumai-next.md`
