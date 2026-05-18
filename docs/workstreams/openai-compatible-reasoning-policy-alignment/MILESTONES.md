# OpenAI-Compatible Reasoning Policy Alignment - Milestones

Status: Completed
Last updated: 2026-05-18

## M0 - Scope And Evidence Freeze

Exit criteria:

- Workstream docs describe the problem, target state, scope, and non-goals.
- AI SDK reference behavior is summarized without copying implementation details.

Status: Complete.

## M1 - Shared Reasoning Policy

Exit criteria:

- Reasoning/thinking request parameter mapping lives behind one policy module.
- Policy tests cover the provider families currently handled by duplicated code.

Status: Complete.

## M2 - Builder And Config Integration

Exit criteria:

- Builder and config fluent APIs call the shared policy.
- Duplicate provider-id reasoning mapping is removed from call sites.
- Existing behavior remains covered by regression tests.

Status: Complete.

## M3 - Verification And Closeout

Exit criteria:

- Formatting and focused tests pass.
- Workstream docs record final evidence and status.
- Remaining architecture candidates are explicit follow-ons.

Status: Complete.
