# FCAB-130 Review — Family Taxonomy Finalization

Date: 2026-05-21
Reviewer: Codex

## Workstream Compliance

- Blocking findings: none.
- The task required finalizing family taxonomy, treating Video as stable if source/docs support it,
  and keeping Music extension-only unless an ADR is opened.
- The source already supported Video as stable through `VideoModel`, `VideoModelV4`, `siumai::video`,
  `prelude::unified` exports, `VideoModelHandle`, and `ProviderRegistryHandle::video_model(...)`.
- The docs now consistently name seven stable families: Language, Embedding, Image, Rerank, Speech,
  Transcription, and Video.
- ADR 0001 and ADR 0006 now record an FCAB-130 amendment rather than silently changing the public
  contract.
- Music remains extension-only; no `MusicModel`, `MusicModelHandle`, or registry `music_model(...)`
  path was added.
- Workstream TODO, MILESTONES, EVIDENCE, HANDOFF, and seam inventory were updated.

## Code Quality

- Blocking findings: none.
- The new source guard checks both documentation and actual source seams, so a future regression
  cannot merely update one side of the taxonomy.
- The guard correctly distinguishes Video's stable family model surface from the lower-level
  `VideoGenerationCapability` adapter/compat surface.
- Music remains available through explicit extension/capability paths, avoiding premature stable API
  design for a family that has not gone through an ADR.
- No runtime behavior changed; the task is a taxonomy/docs/source-guard convergence slice.

## Missing Gates

- No task-local missing gates.
- Broader workspace smoke/integration gates remain FCAB-140 scope.

## Residual Risk

- Historical beta.6 documentation still mentions the beta.6 family shape, but it now includes an
  explicit note that beta.7+ promotes Video and keeps Music extension-only.
- Future Music family work would be a public contract expansion and should be split into its own ADR
  and workstream.

## Verdict

FCAB-130 is ready to mark complete. Proceed to FCAB-140 integration and closeout validation.
