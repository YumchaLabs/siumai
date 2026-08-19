# Siumai Documentation

This directory separates durable repository documentation from task-scoped plans and historical
research. A file's location alone does not make it current authority; use the status and ownership
rules below before relying on it.

## Current authority

| Area | Document | Status |
|---|---|---|
| Repository working rules | `AGENTS.md` | Current |
| Contributor guide | `../CONTRIBUTING.md` | Current provider and validation checklist |
| Architecture | `architecture/overview.md` | Current ownership and dependency contract |
| Public API | `architecture/public-api.md` | Current facade, provider, option, feature, and stability policy |
| Registry | `architecture/registry.md` | Current local-routing contract |
| Transport and security | `architecture/transport-contract.md` | Accepted contract; verify implementation status per crate |
| Provider/host ownership | `adr/0010-provider-plane-and-host-control-plane.md` | Accepted |
| Provider identity and registration | `adr/0013-provider-identity-and-family-registration.md` | Accepted |
| Typed Siumai provider hub | `adr/0020-typed-siumai-provider-hub.md` | Accepted; supersedes ADR-0019's rejection of a `Siumai` entry |
| Provider support claims | `providers/support-policy.md` | Current claim policy |
| Gemini provider support | `providers/google.md` | Current dated Gemini Interactions and product-surface evidence |
| MiniMax support | `providers/minimax.md` | Current dated provider/API-mode and resource evidence |
| Breaking API migration | `migration/siumai-next.md` | User-facing guide for the next breaking release |
| Release process | `releasing.md` | Current manual workflow and known release-plz restart limitation |

Task plans may describe temporary migration work and acceptance criteria. While a plan is active,
checkpoint notes may be recorded there for traceability. Once a task completes, the plan is a
task-scoped record, not standing repository authority; durable outcomes belong in architecture,
ADRs, provider evidence, migration guidance, or the changelog.

## Document ownership

- `architecture/` explains current or accepted system boundaries and public API shape. Each retained
  document must state whether it is current, accepted-but-not-fully-implemented, or historical.
- `adr/` records accepted durable decisions and their trade-offs. Superseded decisions should be
  removed when their remaining rationale has been incorporated into current architecture docs.
- `providers/` owns support-claim policy, dated official evidence, fidelity/stability declarations,
  and the provider/family/API-mode support matrix.
- `migration/` is user-facing guidance for released breaking changes, not an internal task journal.
- `plans/` contains active task authority only. Completed plans may be removed once their durable
  decisions and migration guidance have been captured elsewhere.

## Historical material

Obsolete alignment reports, workstream journals, and source-layout snapshots are intentionally kept
in Git history instead of the active documentation tree. Do not recreate task journals as standing
repository documentation. Move durable decisions into ADRs, current boundaries into architecture
documents, and dated provider evidence into provider support documentation.

When code, tests, and documentation disagree, identify the owning contract before editing. Prefer
accepted ADRs and current architecture/provider documents over historical parity notes, and keep
the implementation, executable contracts, and user-facing documentation consistent.
