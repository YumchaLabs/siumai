# Shared Type Surface Alignment - Milestones

Last updated: 2026-04-21

## M1 - Shared audit baseline

Status: done

- audit `repo-ref/ai/packages/ai/src/types/index.ts`
- audit the upstream shared metadata and usage files
- separate true shared/public data structures from provider-owned typed options

## M2 - Stable carrier widening

Status: done

- add `siumai-spec/src/types/ai_sdk.rs`
- widen `ResponseMetadata` with optional `headers`
- widen `Warning` with `Deprecated { setting, message }`
- add conversion helpers from existing runtime carriers

## M3 - Public facade exposure

Status: done

- re-export the shared names from `siumai::types::*`
- re-export the audited shared names from `siumai::prelude::unified::*`
- add compile-guard coverage for the new surface

## M4 - Follow-up alignment

Status: deferred

- design prompt/request helper types that are not yet stable enough to expose
- tighten runtime body/header capture where providers expose richer metadata
- continue auditing other shared AI package helper/data structures against `repo-ref/ai`
