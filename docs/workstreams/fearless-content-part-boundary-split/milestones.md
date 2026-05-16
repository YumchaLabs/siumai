# Fearless ContentPart Boundary Split - Milestones

Last updated: 2026-05-16

## CPB-M1 - Workstream Opened

Status: complete

Exit criteria:

- Workstream docs exist.
- The docs index links the workstream.
- The deferred legacy `ContentPart` problem is described as a separate compatibility-breaking lane.

## CPB-M2 - Direct Construction Guard Tightened

Status: complete

Exit criteria:

- New production direct `ContentPart` construction cannot appear without an audit classification.
- Guarded adapter paths are documented.
- Focused facade/spec boundary tests pass.

## CPB-M3 - Request Adapter Slice Shipped

Status: complete

Exit criteria:

- At least one production request path stops constructing legacy `ContentPart` directly.
- The replacement adapter rejects response-side metadata reads.
- Focused tests prove the request payload is unchanged.

## CPB-M4 - Response Adapter Slice Shipped

Status: complete

Exit criteria:

- At least one production response path stops constructing legacy `ContentPart` directly.
- The replacement adapter avoids request-side provider options on response output.
- Focused tests prove the response projection is unchanged.

## CPB-M5 - Compatibility Decision Recorded

Status: complete

Exit criteria:

- The refreshed scan is classified.
- The next breaking decision for legacy `ContentPart` is recorded in docs or an ADR.
- Migration docs point users at canonical request and response content shapes.

Notes:

- ADR-0008 records legacy `ContentPart` as a compatibility carrier that should stay in place for
  the current beta line and move only in a later breaking slice once directional adapters and docs
  cover the main request/response paths.
- The beta.7 migration guide now lists canonical request and response content imports.
