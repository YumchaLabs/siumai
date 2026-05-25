# Native Extension And Compat Retirement — TODO

Status: Closed
Last updated: 2026-05-25

Status legend:

- `[ ]` pending
- `[~]` in progress
- `[x]` complete
- `[-]` intentionally deferred or split

## M0 — Scope And Evidence Freeze

- [x] NECR-010 [owner=planner] [deps=none] [scope=docs/workstreams/native-extension-and-compat-retirement]
  Goal: Open the follow-on lane for native extension factory convergence, method-style generic-client
  retirement planning, and ADR-0008 root `ContentPart` move preparation.
  Validation: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`,
  `WORKSTREAM.json`, and `HANDOFF.md` exist and agree.
  Review: planner self-review.
  Evidence: workstream docs.
  Handoff: DONE. Workstream docs opened, indexed, and validated. First executable task is
  NECR-020.

## M1 — Extension Factory Inventory

- [x] NECR-020 [owner=codex] [deps=NECR-010] [scope=siumai-registry/src,siumai-registry/tests,docs/workstreams/native-extension-and-compat-retirement]
  Goal: Inventory `ProviderExtensionFactory` defaults, provider overrides, native extension clients,
  and remaining `compat_*_client_with_ctx` fallback paths.
  Validation:
  - focused source inventory in this workstream;
  - existing registry architecture boundary tests still pass.
  Review: registry construction-boundary review.
  Evidence: inventory section in `EVIDENCE_AND_GATES.md` or a dedicated decision note.
  Handoff: DONE. `ProviderFactory` extension defaults were inventoried. DeepInfra, Fireworks, and
  TogetherAI were selected as the first image-extras native override targets because each already
  has a native image client implementing `ImageExtras`.

## M2 — Native Extension Factory Convergence

- [x] NECR-030 [owner=codex] [deps=NECR-020] [scope=siumai-registry/src,siumai-registry/tests,provider factory sources]
  Goal: Add the first safe provider-owned native extension factory override where the provider
  already has a native extension client, reducing reliance on generic-client adapter fallback.
  Validation:
  - `cargo check -p siumai-registry --tests --no-default-features --features openai`
  - focused registry factory architecture tests.
  Review: provider construction-path review.
  Evidence: source guard or test proving the selected extension path does not call
  `compat_*_client_with_ctx`.
  Handoff: DONE. DeepInfra, Fireworks, and TogetherAI now override
  `image_extras_with_ctx(...)` to return native image extras clients through
  `ProviderExtensionFactory`, with a boundary test guarding against generic-client adapter fallback.

## M3 — Method-Style / Generic-Client Retirement Plan

- [x] NECR-040 [owner=codex] [deps=NECR-020] [scope=siumai-registry/src,siumai-core/src,docs/adr,docs/architecture,docs/migration,tests]
  Goal: Turn method-style and generic-client retirement from prose into source-enforced criteria and
  a deletion plan.
  Validation:
  - focused source guard for compatibility-only construction paths;
  - docs identify exact preconditions for deleting `ProviderCompatibilityFactory` and core aliases.
  Review: ADR-0007 compliance review.
  Evidence: guard test and doc updates.
  Handoff: DONE. `ProviderCompatibilityFactory` is now guarded as method-style-only production
  infrastructure, and ADR-0007 plus public migration docs define deletion gates.

## M4 — ADR-0008 Root ContentPart Move Preparation

- [x] NECR-050 [owner=codex] [deps=NECR-020] [scope=siumai-spec,siumai-core,siumai,docs/adr,tests]
  Goal: Add executable parity gates or a safe preparatory namespace slice for the low-level
  `ContentPart` root move without breaking serde-facing payload compatibility.
  Validation:
  - focused content projection boundary tests;
  - public import tests for any namespace movement.
  Review: ADR-0008 compliance review.
  Evidence: parity gate or decision note.
  Handoff: DONE. Added a serde fixture gate that locks the current root/compat `ContentPart`
  payload equivalence inside `ChatMessage` and `ChatResponse`; low-level root movement remains
  blocked until broader provider/protocol fixture parity exists.

## M5 — Closeout

- [x] NECR-060 [owner=planner] [deps=NECR-030,NECR-040,NECR-050] [scope=docs/workstreams/native-extension-and-compat-retirement]
  Goal: Close the lane or split any remaining provider-specific removals into narrower follow-ons.
  Validation:
  - documented final gate matrix in `EVIDENCE_AND_GATES.md`
  - `git diff --check -- docs/workstreams/native-extension-and-compat-retirement`
  Review: final self-review or `review-workstream`.
  Evidence: updated workstream docs.
  Handoff: DONE. Lane closed with follow-ons split to broader provider/protocol fixture parity,
  additional provider-native extension overrides, and any eventual breaking low-level root
  `ContentPart` namespace movement.
