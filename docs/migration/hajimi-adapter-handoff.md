# Hajimi Adapter Handoff

This document maps the containment code that a downstream Hajimi-style agent adapter may currently
carry to the Siumai invariants in the `0.11.0-beta.9` breaking line. It is a compatibility handoff,
not a dependency or runtime contract: Hajimi remains host-owned and may keep stricter budgets,
approvals, cancellation, deadlines, and business policy.

## Containment mapping

| Downstream containment | Siumai contract to consume | Handoff action |
|---|---|---|
| Normalize an encoded terminal tool-argument string against a streamed call | `ToolInput`/`ToolCall::local` canonical JSON plus checked provider decoding | Remove protocol-specific string/object repair from the adapter. Accept the checked parsed value for caller-owned tools. |
| Keep provider-executed tool input opaque | `ToolExecutionOwner` and provider-native opaque items | Keep provider-owned tools out of local `ToolSet` execution. Replay native items through the provider API. |
| Parse raw SSE error strings, JSON, and status codes | Typed stream failure with `LlmError` category, retryability, delay, safe message, and bounded diagnostics | Stop classifying provider payloads in the agent loop. Match the typed error and use diagnostics only through an explicit redacted boundary. |
| Reject a successful EOF that has no terminal response | Exactly-once stream settlement: terminal response or typed error | Remove the consumer-side EOF state machine. An established Siumai stream cannot complete successfully without settlement. |
| Reconcile incremental tool events with the terminal snapshot | Protocol normalization and semantic parity for shared executable items | Treat the terminal response as canonical. Retain only host-specific UI reconciliation for items that intentionally exist in one view. |
| Repair late Chat usage after `finish_reason` | Buffered terminal metadata and usage-preserving Chat stream conversion | Remove the usage workaround. Read usage from the final response after the trailing usage-only chunk. |
| Force reasoning options for private model identifiers | Typed provider options with fail-closed explicit-effort handling | Set effort through the public typed option. It is encoded or rejected; it is never silently filtered by a model allowlist. |
| Reject `max` through a downstream JSON workaround | `OpenAiReasoningEffort::Max` and independent reasoning summary controls | Remove the compatibility workaround and map `Max` directly where the selected protocol supports it. |
| Encode host-defined JSON as a portable `Custom` content part | Direction-specific request/output contracts and namespaced provider extensions | Do not use `Custom` as an arbitrary host transport. Keep host context in the host envelope or an explicit provider extension. |

## What remains Hajimi policy

Siumai does not own the agent's tool approvals, cancellation policy, iteration budget, storage
format, context-window rollover policy, or redaction policy for host logs. Hajimi may apply stricter
limits than Siumai's protocol bounds and may reject a valid provider result for host-level reasons.
Those decisions should be represented as host errors, not confused with provider protocol failures.

## Local checkout note

The local `../rust/hajimi` checkout inspected on 2026-08-09 still declares `siumai =
0.11.0-beta.4` and does not contain the separately described `hajimi-provider-siumai` adapter crate.
Its tests therefore exercise an older integration surface and are not a release gate for this
Siumai line. Do not update that checkout or its lockfile as part of Siumai release work.

When a downstream adapter is migrated to this line, use its own credential-free tests first. A live
canary may be run only with explicit operator authorization and an approved endpoint; relay capacity
or upstream timeouts are operational evidence, not deterministic compatibility failures.
