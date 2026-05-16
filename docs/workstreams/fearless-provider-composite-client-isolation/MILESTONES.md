# Fearless Provider Composite Client Isolation - Milestones

Last updated: 2026-05-16

Status: completed

## M1 - Framed

- Workstream docs exist and are linked from the docs index.
- The lane has a bounded target: isolate hybrid provider composite clients as compat-only adapters.

## M2 - Naming Makes Intent Visible

- DeepInfra, Fireworks, and TogetherAI private composite clients use compat-oriented names.
- Debug output also describes those wrappers as compatibility composite clients.

## M3 - Family Paths Are Guarded

- Stable family factory methods do not construct provider composite clients.
- Stable family factory methods do not self-call `compat_*_client_with_ctx(...)`.
- Stable family factory methods do not use `LlmClient` capability downcasts.

## M4 - Audit And Migration Are Updated

- The compatibility audit documents the compat-only state and deletion condition.
- Migration/changelog notes explain that this is an internal architecture hardening step.

## M5 - Closed Or Split

- Focused gates pass.
- Remaining work is split conceptually into a future compatibility-wrapper deletion lane, not kept
  open here.
